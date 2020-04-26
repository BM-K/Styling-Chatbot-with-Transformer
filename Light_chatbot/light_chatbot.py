import argparse
import re

import torch
from konlpy.tag import Mecab
from torch import nn
from torchtext import data
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset

from Styling import styling, make_special_token
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
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--per_soft', type=bool, default=False)
parser.add_argument('--per_rough', type=bool, default=True)
args = parser.parse_args()

def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        _acc = (yhat == y).float()[y != 1].mean() # padding은 acc에서 제거
    return _acc

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

# tokenizer
def tokenizer1(text):
    result_text = re.sub(r'[-=+.,#/\:$@*\"※&%ㆍ!?』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
    a = Mecab().morphs(result_text)
    return ([a[i] for i in range(len(a))])

# 데이터 전처리 및 loader return
def data_preprocessing(args, device):

    # ID는 사용하지 않음. SA는 Sentiment Analysis 라벨(0,1) 임.
    ID = data.Field(sequential=False,
                    use_vocab=False)

    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=tokenizer1,
                      batch_first=True,
                      fix_length=args.max_len,
                      dtype=torch.int32
                      )

    LABEL = data.Field(sequential=True,
                       use_vocab=True,
                       tokenize=tokenizer1,
                       batch_first=True,
                       fix_length=args.max_len,
                       init_token='<sos>',
                       eos_token='<eos>',
                       dtype=torch.int32
                       )

    SA = data.Field(sequential=False,
                    use_vocab=False)

    train_data, test_data = TabularDataset.splits(
        path='.', train='chatbot_0325_ALLLABEL_train.txt', test='chatbot_0325_ALLLABEL_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('target_text', LABEL), ('SA', SA)], skip_header=True
    )

    # TEXT, LABEL 에 필요한 special token 만듦.
    text_specials, label_specials = make_special_token(args)

    TEXT.build_vocab(train_data, max_size=15000, specials=text_specials)
    LABEL.build_vocab(train_data, max_size=15000, specials=label_specials)

    train_loader = BucketIterator(dataset=train_data, batch_size=args.batch_size, device=device, shuffle=True)
    test_loader = BucketIterator(dataset=test_data, batch_size=args.batch_size, device=device, shuffle=True)
    # BucketIterator(dataset=traing_data check)

    return TEXT, LABEL, test_loader

def main(TEXT, LABEL):
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL.vocab.stoi['<pad>'])

    # print argparse
    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    from model import Transformer

    model = Transformer(args, TEXT, LABEL)
    if args.per_soft:
        sorted_path = 'sorted_model-soft.pth'
    else:
        sorted_path = 'sorted_model-rough.pth'
    model.to(device)

    checkpoint = torch.load(sorted_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = test(model, test_loader, criterion)  # 아
    print(f'==test_loss : {test_loss:.3f} | test_acc: {test_acc:.3f}==')
    print("\t-----------------------------")
    while (True):
        inference(device, args, TEXT, LABEL, model)
        print("\n")

    return 0

if __name__=='__main__':
    print("-준비중-")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    TEXT, LABEL, test_loader = data_preprocessing(args, device)
    main(TEXT, LABEL)