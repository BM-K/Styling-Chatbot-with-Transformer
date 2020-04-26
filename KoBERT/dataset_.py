import torch
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

bert_model, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


class inferBERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i]


def get_loader(max_len: int, batch_size: int):
    dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("ratings_test.txt", field_indices=[1, 2], num_discard_samples=1)

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, drop_last=False, shuffle=False)

    return train_dataloader, test_dataloader


def infer(max_len: int, src):
    SRC_data = inferBERTDataset(src, 0, tok, max_len, True, False)
    return SRC_data

# num=0
# print(len(data_test))
# f = open('chat_Q_label.txt', 'r', encoding='utf-8')
# rdr = csv.reader(f, delimiter='\t')
# for idx, lin in enumerate(rdr):
#     num+=1
# print(num)
# exit()
