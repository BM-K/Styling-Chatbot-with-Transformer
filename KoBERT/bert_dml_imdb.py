import torch
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

klloss = [0]
num = [0]

torch.backends.cudnn.deterministic = True

from torchtext import datasets

init_token = bert_tokenizer.cls_token
eos_token = bert_tokenizer.sep_token
pad_token = bert_tokenizer.pad_token
unk_token = bert_tokenizer.unk_token
init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = bert_tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = bert_tokenizer.convert_tokens_to_ids(unk_token)
BATCH_SIZE = 32

max_input_length = bert_tokenizer.max_model_input_sizes['bert-base-uncased']
def tokenize_and_cut(sentence):
    tokens = bert_tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchtext import data

TEXT = data.Field(
                  use_vocab=False,
                  tokenize=tokenize_and_cut,
                  preprocessing=bert_tokenizer.convert_tokens_to_ids,
                  init_token=init_token_idx,
                  eos_token=eos_token_idx,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx)

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split()
LABEL.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device,
    shuffle=True)

import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=0 if n_layers < 2 else dropout)
        self.init_weights()

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(embedding_dim, output_dim)
    def forward(self, text):
        # text = [sent len, batch size]
       # with torch.no_grad():
        embedded = self.bert(text)[1]
        embedded = self.dropout(embedded)
        output = self.classifier(embedded)
        # embedded = [sent len, batch size, emb dim]

        #_, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        #if self.rnn.bidirectional:
        #    hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        #else:
        #    hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        #output = self.out(hidden)

        # output = [batch size, out dim]

        return output

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for name, param in bert_model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

import torch.nn.functional as F
from torch.autograd import Variable

def train(mutual_model, iterator, optimizer, criterion):
    batch_list = []
    batch_label = []
    epoch_loss = []
    epoch_acc = []
    for i in range(model_num):
        epoch_loss.append(0)
        epoch_acc.append(0)

    for batch in iterator:
        text = batch.text
        batch_list.append(text)
        batch_label.append(batch.label)
        outputs = []
        for i in range(model_num):
            model = mutual_model[i]
            model.train()
            predictions = model(text).squeeze(1)
            outputs.append(predictions)

        for i in range(model_num):
            ce_loss = criterion(outputs[i], batch.label)
            kl_loss = 0
            for j in range(model_num):
                if i != j:
                    kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                       F.softmax(Variable(outputs[j])))
            klloss[0] = klloss[0] + kl_loss.item()
            num[0] = num[0] + 1

            loss = ce_loss + kl_loss / (model_num - 1)

            acc = binary_accuracy(outputs[i], batch.label)
                  
            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()

            epoch_loss[i] += loss.item()
            epoch_acc[i] += acc.item()

    for i in range(model_num):
        epoch_loss[i] /= len(iterator)
        epoch_acc[i] /= len(iterator)
    return epoch_loss, epoch_acc, batch_list, batch_label


def evaluate(mutual_model, iterator, criterion):
    ev_ba_list = []
    ev_ba_label = []
    epoch_loss = []
    epoch_acc = []
    for i in range(model_num):
        epoch_loss.append(0)
        epoch_acc.append(0)

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            ev_ba_list.append(text)
            ev_ba_label.append(batch.label)

            outputs = []
            for i in range(model_num):
                model = mutual_model[i]
                model.eval()
                predictions = model(text).squeeze(1)
                outputs.append(predictions)

            for i in range(model_num):
                ce_loss = criterion(outputs[i], batch.label)
                kl_loss = 0
                for j in range(model_num):
                    if i != j:
                        kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                           F.softmax(Variable(outputs[j])))
                loss = ce_loss + kl_loss / (model_num - 1)

                acc = binary_accuracy(outputs[i], batch.label)

                epoch_loss[i] += loss.item()
                epoch_acc[i] += acc.item()

    for i in range(model_num):
        epoch_loss[i] /= len(iterator)
        epoch_acc[i] /= len(iterator)
    return epoch_loss, epoch_acc, ev_ba_list, ev_ba_label

def in_train(model, iterator, optimizer, criterion, batch_list, batch_label):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for idx, batch in enumerate(batch_list):
        optimizer.zero_grad()

        text = batch

        predictions = model(text).squeeze(1)

        loss = criterion(predictions, batch_label[idx])

        acc = binary_accuracy(predictions, batch_label[idx])

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def in_evaluate(model, iterator, criterion, ev_ba_list, ev_ba_label):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(ev_ba_list):
            text = batch

            predictions = model(text).squeeze(1)

            loss = criterion(predictions, ev_ba_label[idx])

            acc = binary_accuracy(predictions, ev_ba_label[idx])

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
N_EPOCHS = 10
model_num = 2
mutual_model = []
mutual_optim = []

best_valid_loss = []
criterion = nn.BCEWithLogitsLoss()

in_model = BERTGRUSentiment(bert_model,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT)

in_best_valid_loss = float('inf')
in_optimizer = torch.optim.Adam(in_model.parameters(), lr=0.001)
in_model.cuda()

for i in range(model_num):
    model = BERTGRUSentiment(bert_model,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT)
    model.cuda()
    mutual_model.append(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mutual_optim.append(optimizer)
    best_valid_loss.append(float('inf'))

loss_kl = nn.KLDivLoss(reduction='batchmean')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc, batch_list, batch_label =\
        train(mutual_model, train_iterator, mutual_optim, criterion)
    valid_loss, valid_acc, ev_ba_list, ev_ba_label =\
        evaluate(mutual_model, valid_iterator, criterion)

    in_train_loss, in_train_acc = in_train(in_model, train_iterator, in_optimizer, criterion,
                                           batch_list, batch_label)
    in_valid_loss, in_valid_acc = in_evaluate(in_model, valid_iterator, criterion,
                                              ev_ba_list, ev_ba_label)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    for i in range(model_num):
        if valid_loss[i] < best_valid_loss[i]:
            best_valid_loss[i] = valid_loss[i]
            torch.save(mutual_model[i].state_dict(), f'mutual_model{i}.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    for i in range(model_num):
        print(f"Model{i}")
        print(f'\tTrain Loss: {train_loss[i]:.3f} | Train Acc: {train_acc[i] * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss[i]:.3f} |  Val. Acc: {valid_acc[i] * 100:.2f}%')

    if in_valid_loss < in_best_valid_loss:
        in_best_valid_loss = in_best_valid_loss
        torch.save(in_model.state_dict(), 'in-model.pt')
    print("독립")
    print(f'\tTrain Loss: {in_train_loss:.3f} | Train Acc: {in_train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {in_valid_loss:.3f} |  Val. Acc: {in_valid_acc * 100:.2f}%')

for i in range(model_num):
    mutual_model[i].load_state_dict(torch.load(f'mutual_model{i}.pt'))
    test_loss, test_acc, ev_ba_list, ev_ba_label =\
        evaluate(mutual_model, test_iterator, criterion)
    print(f'Test<m{i}> Loss: {test_loss[i]:.3f} | Test<m{i}> Acc: {test_acc[i] * 100:.2f}%')
    if i+1 == model_num:
        print("독립")
        in_model.load_state_dict(torch.load('in-model.pt'))
        in_test_loss, in_test_acc = in_evaluate(in_model, test_iterator, criterion,
                                                ev_ba_list, ev_ba_label)
        print(f'Test Loss: {in_test_loss:.3f} | Test Acc: {in_test_acc * 100:.2f}%')

print("kl loss :", klloss[0]/num[0])
