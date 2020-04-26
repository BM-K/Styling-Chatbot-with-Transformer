import time
import torch
import argparse
from torch import nn
from metric import acc, train_test
from Styling import styling, make_special_token
from get_data import data_preprocessing, tokenizer1
from generation import inference

SEED = 1234

# argparse 정의
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=40) # max_len 크게 해야 오류 안 생김.
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=22)
parser.add_argument('--warming_up_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--embedding_dim', type=int, default=160)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--train', action="store_true")
group = parser.add_mutually_exclusive_group()
group.add_argument('--per_soft', action="store_true")
group.add_argument('--per_rough', action="store_true")
args = parser.parse_args()

# 시간 계산 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 학습
def train(model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    tr_acc = 0
    model.train()

    for step, batch in enumerate(iterator):
        optimizer.zero_grad()

        enc_input, dec_input , enc_label = batch.text, batch.target_text, batch.SA
        dec_output = dec_input[:, 1:]
        dec_outputs = torch.zeros(dec_output.size(0), args.max_len).type_as(dec_input.data)

        # emotion 과 체를 반영
        enc_input, dec_input, dec_outputs = \
            styling(enc_input, dec_input, dec_output, dec_outputs, enc_label, args, TEXT, LABEL)

        y_pred = model(enc_input, dec_input)

        y_pred = y_pred.reshape(-1, y_pred.size(-1))
        dec_output = dec_outputs.view(-1).long()

        # padding 제외한 value index 추출
        real_value_index = [dec_output != 1] # <pad> == 1

        # padding 은 loss 계산시 제외
        loss = criterion(y_pred[real_value_index], dec_output[real_value_index])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = acc(y_pred, dec_output)

        total_loss += loss
        iter_num += 1
        tr_acc += train_acc

        train_test(step, y_pred, dec_output, real_value_index, enc_input,
                   args, TEXT, LABEL)

    return total_loss.data.cpu().numpy() / iter_num, tr_acc.data.cpu().numpy() / iter_num

# 테스트
def test(model, iterator, criterion):
    total_loss = 0
    iter_num = 0
    te_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            enc_input, dec_input, enc_label = batch.text, batch.target_text, batch.SA
            dec_output = dec_input[:, 1:]
            dec_outputs = torch.zeros(dec_output.size(0), args.max_len).type_as(dec_input.data)

            # emotion 과 체를 반영
            enc_input, dec_input, dec_outputs = \
                styling(enc_input, dec_input, dec_output, dec_outputs, enc_label, args, TEXT, LABEL)

            y_pred = model(enc_input, dec_input)

            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_outputs.view(-1).long()

            real_value_index = [dec_output != 1]  # <pad> == 1

            loss = criterion(y_pred[real_value_index], dec_output[real_value_index])

            with torch.no_grad():
                test_acc = acc(y_pred, dec_output)
            total_loss += loss
            iter_num += 1
            te_acc += test_acc

    return total_loss.data.cpu().numpy() / iter_num, te_acc.data.cpu().numpy() / iter_num

def main(TEXT, LABEL, train_loader, test_loader):

    # for sentiment analysis. load .pt file
    from KoBERT.Bert_model import BERTClassifier
    from kobert.pytorch_kobert import get_pytorch_kobert_model
    bertmodel, vocab = get_pytorch_kobert_model()
    sa_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    sa_model.load_state_dict(torch.load('bert_SA-model.pt'))

    # print argparse
    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__)-1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    from model import Transformer, GradualWarmupScheduler

    # Transformer model init
    model = Transformer(args, TEXT, LABEL)
    if args.per_soft:
        sorted_path = 'sorted_model-soft.pth'
    else:
        sorted_path = 'sorted_model-rough.pth'

    # loss 계산시 pad 제외.
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL.vocab.stoi['<pad>'])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=args.num_epochs)

    # pre-trained 된 vectors load
    model.src_embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.trg_embedding.weight.data.copy_(LABEL.vocab.vectors)
    model.to(device)
    criterion.to(device)

    # overfitting 막기
    best_valid_loss = float('inf')

    # train
    if args.train:
        for epoch in range(args.num_epochs):
            torch.manual_seed(SEED)
            scheduler.step(epoch)
            start_time = time.time()

            # train, validation
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            valid_loss, valid_acc = test(model, test_loader, criterion)

            # time cal
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #torch.save(model.state_dict(), sorted_path) # for some overfitting
            #전에 학습된 loss 보다 현재 loss 가 더 낮을시 모델 저장.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss},
                    sorted_path)
                print(f'\t## SAVE valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} ##')

            # print loss and acc
            print(f'\n\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
            print(f'\t==Train Loss: {train_loss:.3f} | Train_acc: {train_acc:.3f}==')
            print(f'\t==Valid Loss: {valid_loss:.3f} | Valid_acc: {valid_acc:.3f}==\n')

    # inference
    print("\t----------성능평가----------")
    checkpoint = torch.load(sorted_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = test(model, test_loader, criterion) # 아
    print(f'==test_loss : {test_loss:.3f} | test_acc: {test_acc:.3f}==')
    print("\t-----------------------------")
    while (True):
        inference(device, args, TEXT, LABEL, model, sa_model)
        print("\n")

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # TEXT 는 사람의 말, LABEL 은 챗봇 답변을 의미하는 Field.
    TEXT, LABEL, train_loader, test_loader = data_preprocessing(args, device)
    main(TEXT, LABEL, train_loader, test_loader)
