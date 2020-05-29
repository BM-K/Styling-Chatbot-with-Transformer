import torch

# acc 출력
def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float()[y != 1].mean() # padding은 acc에서 제거
    return acc

# 학습시 모델에 넣는 입력과 모델의 예측 출력.
def train_test(step, y_pred, dec_output, real_value_index, enc_input, args, TEXT, LABEL):

    if 0 <= step < 3:
        _, ix = y_pred[real_value_index].data.topk(1)
        train_Q = enc_input[0]
        print("<<Q>> :", end=" ")
        for i in train_Q:
            if TEXT.vocab.itos[i] == "<pad>":
                break
            print(TEXT.vocab.itos[i], end=" ")

        print("\n<<trg A>> :", end=" ")
        for jj, jx in enumerate(dec_output[real_value_index]):
            if LABEL.vocab.itos[jx] == "<eos>":
                break
            print(LABEL.vocab.itos[jx], end=" ")

        print("\n<<pred A>> :", end=" ")
        for jj, ix in enumerate(ix):
            if jj == args.max_len:
                break
            if LABEL.vocab.itos[ix] == '<eos>':
                break
            print(LABEL.vocab.itos[ix], end=" ")
        print("\n")
