import torch
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.vocab import Vectors
from konlpy.tag import Mecab
import re
from Styling import styling, make_special_token

# tokenizer
def tokenizer1(text):
    result_text = re.sub('[-=+.,#/\:$@*\"※&%ㆍ!?』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
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

    vectors = Vectors(name="kr-projected.txt")

    # TEXT, LABEL 에 필요한 special token 만듦.
    text_specials, label_specials = make_special_token(args)

    TEXT.build_vocab(train_data, vectors=vectors, max_size=15000, specials=text_specials)
    LABEL.build_vocab(train_data, vectors=vectors, max_size=15000, specials=label_specials)

    train_loader = BucketIterator(dataset=train_data, batch_size=args.batch_size, device=device, shuffle=True)
    test_loader = BucketIterator(dataset=test_data, batch_size=args.batch_size, device=device, shuffle=True)

    return TEXT, LABEL, train_loader, test_loader
