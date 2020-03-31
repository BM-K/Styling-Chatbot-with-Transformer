import argparse
import torch
from torch import nn
from tqdm import tqdm
from transformers import AdamW
#from transformers.optimization import WarmupLinearSchedule
import time
import random
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
bertmodel, vocab = get_pytorch_kobert_model()
import KoBERT.dataset_ as dataset
# print(vocab.to_tokens(517))
# print(vocab.to_tokens(5515))
# print(vocab.to_tokens(517))
# print(vocab.to_tokens(492))
# print("----------------------------------------------")
# print(vocab.to_tokens(3610))
# print(vocab.to_tokens(7096))
# print(vocab.to_tokens(4214))
# print(vocab.to_tokens(1770))
# print(vocab.to_tokens(517))
# print(vocab.to_tokens(46))
# print(vocab.to_tokens(4525))
# print(vocab.to_tokens(3610))
# print(vocab.to_tokens(6954))
#
# exit()

device = torch.device("cuda:0")
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def train(model, iter_loader, optimizer, loss_fn):
    train_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(iter_loader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        #scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
    return loss.data.cpu().numpy(), train_acc/(batch_id + 1)

def test(model, iter_loader, loss_fn):
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(iter_loader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            test_acc += calc_accuracy(out, label)
    return loss.data.cpu().numpy(), test_acc/(batch_id + 1)

def bert_inference(model, src):
    model.eval()
    with torch.no_grad():
        src_data = dataset.infer(args, src)
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(src_data):
            token_ids = torch.tensor([token_ids]).long().to(device)
            segment_ids = torch.tensor([segment_ids]).long().to(device)
            valid_length = valid_length.tolist()
            valid_length = torch.tensor([valid_length]).long()

            out = model(token_ids, valid_length, segment_ids)

            max_vals, max_indices = torch.max(out, 1)

            label = max_indices.data.cpu().numpy()
            if label == 0:
                return 0
            else:
                return 1
    return -1

import csv
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    if args.do_test:
        max_list = max_indices.data.cpu().numpy().tolist()
        f = open('chat_Q_label_0325.txt', 'a', encoding='utf-8')
        wr = csv.writer(f, delimiter='\t')
        for i in range(len(max_list)):
            wr.writerow(str(max_list[i]))
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Argparse init
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--warmup_ratio', type=int, default=0.1)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--do_train', type=bool, default=False)
parser.add_argument('--do_test', type=bool, default=False)
parser.add_argument('--train', type=bool, default=True)
args = parser.parse_args()

def main():
    from Bert_model import BERTClassifier
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    train_dataloader, test_dataloader = dataset.get_loader(args)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)
    best_valid_loss = float('inf')

    # for idx, (key, value) in enumerate(args.__dict__.items()):
    #     if idx == 0:
    #         print("\nargparse{\n", "\t", key, ":", value)
    #     elif idx == len(args.__dict__) - 1:
    #         print("\t", key, ":", value, "\n}")
    #     else:
    #         print("\t", key, ":", value)

    if args.do_train:

        for epoch in range(args.num_epochs):
            start_time = time.time()

            print("\n\t-----Train-----")
            train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn)
            valid_loss, valid_acc = test(model, test_dataloader, loss_fn)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'bert_SA-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('bert_SA-model.pt'))

    if args.do_test:
        test_loss, test_acc = test(model, test_dataloader, loss_fn)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # while(1):
    #     se = input("input : ")
    #     se_list = [se, '-1']
    #     bert_inference(model, [se_list])

if __name__ == "__main__":
    main()