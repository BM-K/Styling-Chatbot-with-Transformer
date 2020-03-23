import torch
import csv
import hgtk
from konlpy.tag import Mecab
import random

mecab = Mecab()
empty_list = []
positive_emo = ['ㅎㅎ', '~']
negative_emo = ['...', 'ㅠㅠ']

def mecab_token_pos_flat_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

def make_cute_word(target):
    # mecab 를 통해 문장을 구분 (example output : ['오늘/MAG', '날씨/NNG', '좋/VA', '다/EF', './SF'])
    ko_sp = mecab_token_pos_flat_fn(target)

    keyword = []
    EF_idx = -1

    # word 에 종결어미 'EF'가 포함 되어 있을 경우 index 와 keyword 추출.
    for idx, word in enumerate(ko_sp):
        if word.find('EF') > 0:
            keyword.append(word.split('/'))
            EF_idx = idx
            break

    # 'EF'가 없을 시 return.
    if keyword == []:
        return '', -1
    else:
        keyword = keyword[0]

    # hgtk 를 사용하여 keyword 를 쪼갬. (ex output : 하ᴥ세요)
    h_separation = hgtk.text.decompose(keyword[0])
    total_word = ''

    for idx, word in enumerate(h_separation):
        total_word += word

    # 'EF' 에 종성 'ㅇ' 를 붙여 Styling
    total_word = replaceRight(total_word, "ᴥ", "ㅇᴥ", 1)

    # 다 이어 붙임. ' 하세요 -> 하세용 ' 으로 변환.
    h_combine = hgtk.text.compose(total_word)

    return h_combine, EF_idx

def make_special_token():
    # 감정을 나타내기 위한 special token
    target_special_voca=['ㅎㅎ', '~', '...', 'ㅠㅠ']

    # train data set 의 chatbot answer 에서 'EF' 를 뽑아 종성 'ㅇ' 을 붙인 special token 생성
    with open('chatbot_0319_label_train.txt', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f, delimiter='\t')
        for idx, line in enumerate(rdr):
            target = line[2] # chatbot answer
            exchange_word, _ = make_cute_word(target)
            target_special_voca.append(str(exchange_word))
    target_special_voca = list(set(target_special_voca))

    # '<posi> : positive, <nega> : negative' 를 의미
    return ['<posi>', '<nega>'], target_special_voca

# python string 함수 replace 를 오른쪽부터 시작하는 함수.
def replaceRight(original, old, new, count_right):
    repeat = 0
    text = original

    count_find = original.count(old)
    if count_right > count_find:  # 바꿀 횟수가 문자열에 포함된 old보다 많다면
        repeat = count_find  # 문자열에 포함된 old의 모든 개수(count_find)만큼 교체한다
    else:
        repeat = count_right  # 아니라면 입력받은 개수(count)만큼 교체한다

    for _ in range(repeat):
        find_index = text.rfind(old)  # 오른쪽부터 index를 찾기위해 rfind 사용
        text = text[:find_index] + new + text[find_index + 1:]

    return text

# transformer 에 input 과 output 으로 들어갈 tensor Styling 변환.
def styling(enc_input, dec_input, dec_output, dec_outputs, enc_label, args, TEXT, LABEL):
    pad_tensor = torch.tensor([LABEL.vocab.stoi['<pad>']]).type(dtype=torch.int32).cuda()

    temp_enc = enc_input.data.cpu().numpy()
    batch_sentiment_list = []

    # encoder input : 나는 너를 좋아해 <posi> <pad> <pad> ... - 형식으로 바꿔줌.
    for i in range(len(temp_enc)):
        for j in range(args.max_len):
            if temp_enc[i][j] == 1 and enc_label[i] == 0:
                temp_enc[i][j] = TEXT.vocab.stoi["<nega>"]
                batch_sentiment_list.append(0)
                break
            elif temp_enc[i][j] == 1 and enc_label[i] == 1:
                temp_enc[i][j] = TEXT.vocab.stoi["<posi>"]
                batch_sentiment_list.append(1)
                break

    enc_input = torch.tensor(temp_enc, dtype=torch.int32).cuda()

    for i in range(len(dec_outputs)):
        dec_outputs[i] = torch.cat([dec_output[i], pad_tensor], dim=-1)

    temp_dec = dec_outputs.data.cpu().numpy()

    dec_outputs_sentiment_list = [] # decoder 에 들어가 감정표현 저장.

    # decoder outputs : 저도 좋아용 ㅎㅎ <eos> <pad> <pad> ... - 형식으로 바꿔줌.
    for i in range(len(temp_dec)): # i = batch size
        temp_sentence = ''
        sa_ = batch_sentiment_list[i]
        if sa_ == 0:
            sa_ = random.choice(negative_emo)
        elif sa_ == 1:
            sa_ = random.choice(positive_emo)
        dec_outputs_sentiment_list.append(sa_)

        for ix, token_i in enumerate(temp_dec[i]):
            temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
        exchange_word, idx = make_cute_word(temp_sentence)

        if exchange_word == '':
            for j in range(len(temp_dec[i])):
                if temp_dec[i][j] == LABEL.vocab.stoi['<eos>']:
                    temp_dec[i][j] = LABEL.vocab.stoi[sa_]
                    temp_dec[i][j+1] = LABEL.vocab.stoi['<eos>']
                    break
            continue
        temp_dec[i][idx] = LABEL.vocab.stoi[exchange_word]

        if LABEL.vocab.itos[temp_dec[i][idx+1]] == '<eos>':
            temp_dec[i][idx+1] = LABEL.vocab.stoi[sa_]
            temp_dec[i][idx+2] = LABEL.vocab.stoi['<eos>']
        else:
            for j in range(len(temp_dec[i])):
                if temp_dec[i][j] == LABEL.vocab.stoi['<eos>']:
                    temp_dec[i][j] = LABEL.vocab.stoi[sa_]
                    temp_dec[i][j + 1] = LABEL.vocab.stoi['<eos>']
                    break

    dec_outputs = torch.tensor(temp_dec, dtype=torch.int32).cuda()

    temp_dec_input = dec_input.data.cpu().numpy()
    # decoder input : <sos> 저도 좋아용 ㅎㅎ <eos> <pad> <pad> ... - 형식으로 바꿔줌.
    for i in range(len(temp_dec_input)):
        temp_sentence = ''
        for ix, token_i in enumerate(temp_dec_input[i]):
            temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
        exchange_word, idx = make_cute_word(temp_sentence)

        if exchange_word == '':
            for j in range(len(temp_dec_input[i])):
                if temp_dec_input[i][j] == LABEL.vocab.stoi['<eos>']:
                    temp_dec_input[i][j] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
                    temp_dec_input[i][j+1] = LABEL.vocab.stoi['<eos>']
                    break
            continue
        temp_dec_input[i][idx] = LABEL.vocab.stoi[exchange_word]

        if LABEL.vocab.itos[temp_dec_input[i][idx+1]] == '<eos>':
            temp_dec_input[i][idx+1] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
            temp_dec_input[i][idx+2] = LABEL.vocab.stoi['<eos>']
        else:
            for j in range(len(temp_dec_input[i])):
                if temp_dec_input[i][j] == LABEL.vocab.stoi['<eos>']:
                    temp_dec_input[i][j] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
                    temp_dec_input[i][j+1] = LABEL.vocab.stoi['<eos>']
                    break

    dec_input = torch.tensor(temp_dec_input, dtype=torch.int32).cuda()

    return enc_input, dec_input, dec_outputs
