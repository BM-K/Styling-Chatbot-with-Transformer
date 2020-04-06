import torch
import csv
import hgtk
from konlpy.tag import Mecab
import random

mecab = Mecab()
empty_list = []
positive_emo = ['ㅎㅎ', '~']
negative_emo = ['...', 'ㅠㅠ']
asdf = []

# mecab 을 통한 형태소 분석.
def mecab_token_pos_flat_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

# rough 를 위한 함수. 대명사 NP (저, 제) 를 찾아 나 or 내 로 바꿔준다.
def exchange_NP(target, args):
    keyword = []
    ko_sp = mecab_token_pos_flat_fn(target)
    for idx, word in enumerate(ko_sp):
        if word.find('NP') > 0:
            keyword.append(word.split('/'))
            _idx = idx
            break
    if keyword == []:
        return '', -1, False

    if keyword[0][0] == '저':
        keyword[0][0] = '나'
    elif keyword[0][0] == '제':
        keyword[0][0] = '내'
    else:
        return keyword[0], _idx, False

    return keyword[0][0], _idx, True

# 단어를 soft or rough 말투로 바꾸는 과정
def make_special_word(target, args, search_ec):
    # mecab 를 통해 문장을 구분 (example output : ['오늘/MAG', '날씨/NNG', '좋/VA', '다/EF', './SF'])
    ko_sp = mecab_token_pos_flat_fn(target)

    keyword = []

    # word 에 종결어미 'EF' or 'EC' 가 포함 되어 있을 경우 index 와 keyword 추출.
    for idx, word in enumerate(ko_sp):
        if word.find('EF') > 0:
            keyword.append(word.split('/'))
            _idx = idx
            break
        if search_ec:
            if ko_sp[-2].find('EC') > 0:
                keyword.append(ko_sp[-2].split('/'))
                _idx = len(ko_sp) -1
                break
            else:
                continue

    # 'EF'가 없을 시 return.
    if keyword == []:
        return '', -1
    else:
        keyword = keyword[0]

    if args.per_rough:
        return keyword[0], _idx

    # hgtk 를 사용하여 keyword 를 쪼갬. (ex output : 하ᴥ세요)
    h_separation = hgtk.text.decompose(keyword[0])
    total_word = ''

    for idx, word in enumerate(h_separation):
        total_word += word

    # 'EF' 에 종성 'ㅇ' 를 붙여 Styling
    total_word = replaceRight(total_word, "ᴥ", "ㅇᴥ", 1)

    # 다 이어 붙임. ' 하세요 -> 하세용 ' 으로 변환.
    h_combine = hgtk.text.compose(total_word)

    return h_combine, _idx

# special token 을 만드는 함수
def make_special_token(args):
    # 감정을 나타내기 위한 special token
    target_special_voca=[]

    banmal_dict = get_rough_dic()

    # train data set 의 chatbot answer 에서 'EF' 를 뽑아 종성 'ㅇ' 을 붙인 special token 생성
    with open('chatbot_0325_ALLLABEL_train.txt', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f, delimiter='\t')
        for idx, line in enumerate(rdr):
            target = line[2] # chatbot answer
            exchange_word, _ = make_special_word(target, args, False)
            target_special_voca.append(str(exchange_word))
    target_special_voca = list(set(target_special_voca))

    banmal_special_voca = []
    for i in range(len(target_special_voca)):
        try:
            banmal_special_voca.append(banmal_dict[target_special_voca[i]])
        except KeyError:
            if args.per_rough:
                print("not include banmal dictionary")
            pass

    # 임의 이모티콘 추가.
    target_special_voca.append('ㅎㅎ')
    target_special_voca.append('~')
    target_special_voca.append('ㅠㅠ')
    target_special_voca.append('...')
    target_special_voca = target_special_voca + banmal_special_voca

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

    # 부드러운 성격
    if args.per_soft:
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
                if LABEL.vocab.itos[token_i] == '<sos>' or LABEL.vocab.itos[token_i] == '<eos>' or LABEL.vocab.itos[token_i] == '<pad>':
                    continue
                temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
            temp_sentence = temp_sentence + '.'  # 마침표에 유무에 따라 형태소 분석이 달라짐.
            exchange_word, idx = make_special_word(temp_sentence, args, True)

            if exchange_word == '':
                for j in range(len(temp_dec[i])):
                    if temp_dec[i][j] == LABEL.vocab.stoi['<eos>']:
                        temp_dec[i][j] = LABEL.vocab.stoi[sa_]
                        temp_dec[i][j+1] = LABEL.vocab.stoi['<eos>']
                        break
                continue

            for j in range(len(temp_dec[i])):
                if LABEL.vocab.itos[temp_dec[i][j]] == '<eos>':
                    temp_dec[i][j - 1] = LABEL.vocab.stoi[exchange_word]
                    temp_dec[i][j] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
                    temp_dec[i][j + 1] = LABEL.vocab.stoi['<eos>']
                    break
                elif temp_dec[i][j] != LABEL.vocab.stoi['<eos>'] and j + 1 == len(temp_dec[i]):
                    print("\t-ERROR- No <EOS> token")
                    exit()

        dec_outputs = torch.tensor(temp_dec, dtype=torch.int32).cuda()

        temp_dec_input = dec_input.data.cpu().numpy()
        # decoder input : <sos> 저도 좋아용 ㅎㅎ <eos> <pad> <pad> ... - 형식으로 바꿔줌.
        for i in range(len(temp_dec_input)):
            temp_sentence = ''
            for ix, token_i in enumerate(temp_dec_input[i]):
                if LABEL.vocab.itos[token_i] == '<sos>' or LABEL.vocab.itos[token_i] == '<eos>' or LABEL.vocab.itos[token_i] == '<pad>':
                    continue
                temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
            temp_sentence = temp_sentence + '.'  # 마침표에 유무에 따라 형태소 분석이 달라짐.
            exchange_word, idx = make_special_word(temp_sentence, args, True)

            if exchange_word == '':
                for j in range(len(temp_dec_input[i])):
                    if temp_dec_input[i][j] == LABEL.vocab.stoi['<eos>']:
                        temp_dec_input[i][j] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
                        temp_dec_input[i][j+1] = LABEL.vocab.stoi['<eos>']
                        break
                continue

            for j in range(len(temp_dec_input[i])):
                if LABEL.vocab.itos[temp_dec_input[i][j]] == '<eos>':
                    temp_dec_input[i][j-1] = LABEL.vocab.stoi[exchange_word]
                    temp_dec_input[i][j] = LABEL.vocab.stoi[dec_outputs_sentiment_list[i]]
                    temp_dec_input[i][j+1] = LABEL.vocab.stoi['<eos>']
                    break
                elif temp_dec_input[i][j] != LABEL.vocab.stoi['<eos>'] and j+1 == len(temp_dec_input[i]):
                    print("\t-ERROR- No <EOS> token")
                    exit()

        dec_input = torch.tensor(temp_dec_input, dtype=torch.int32).cuda()

    # 거친 성격
    elif args.per_rough:
        banmal_dic = get_rough_dic()

        for i in range(len(dec_outputs)):
            dec_outputs[i] = torch.cat([dec_output[i], pad_tensor], dim=-1)

        temp_dec = dec_outputs.data.cpu().numpy()

        # decoder outputs : 나도 좋아  <eos> <pad> <pad> ... - 형식으로 바꿔줌.
        for i in range(len(temp_dec)):  # i = batch size
            temp_sentence = ''
            for ix, token_i in enumerate(temp_dec[i]):
                if LABEL.vocab.itos[token_i] == '<eos>':
                    break
                temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
            temp_sentence = temp_sentence+'.' # 마침표에 유무에 따라 형태소 분석이 달라짐.
            exchange_word, idx = make_special_word(temp_sentence, args, True)
            exchange_NP_word, NP_idx, exist = exchange_NP(temp_sentence, args)

            if exist:
                temp_dec[i][NP_idx] = LABEL.vocab.stoi[exchange_NP_word]

            if exchange_word == '':
                continue
            try:
                exchange_word = banmal_dic[exchange_word]
            except KeyError:
                asdf.append(exchange_word)
                print("not include banmal dictionary")
                pass

            temp_dec[i][idx] = LABEL.vocab.stoi[exchange_word]
            temp_dec[i][idx+1] = LABEL.vocab.stoi['<eos>']
            for k in range(idx+2, args.max_len):
                temp_dec[i][k] = LABEL.vocab.stoi['<pad>']

            # for j in range(len(temp_dec[i])):
            #     if LABEL.vocab.itos[temp_dec[i][j]]=='<eos>':
            #         break
            #     print(LABEL.vocab.itos[temp_dec[i][j]], end='')
            # print()

        dec_outputs = torch.tensor(temp_dec, dtype=torch.int32).cuda()

        temp_dec_input = dec_input.data.cpu().numpy()
        # decoder input : <sos> 나도 좋아 <eos> <pad> <pad> ... - 형식으로 바꿔줌.
        for i in range(len(temp_dec_input)):
            temp_sentence = ''
            for ix, token_i in enumerate(temp_dec_input[i]):
                if ix == 0 :
                    continue # because of token <sos>
                if LABEL.vocab.itos[token_i] == '<eos>':
                    break
                temp_sentence = temp_sentence + LABEL.vocab.itos[token_i]
            temp_sentence = temp_sentence + '.'  # 마침표에 유무에 따라 형태소 분석이 달라짐.
            exchange_word, idx = make_special_word(temp_sentence, args, True)
            exchange_NP_word, NP_idx, exist = exchange_NP(temp_sentence, args)
            idx = idx + 1  # because of token <sos>
            NP_idx = NP_idx + 1

            if exist:
                temp_dec_input[i][NP_idx] = LABEL.vocab.stoi[exchange_NP_word]

            if exchange_word == '':
                continue

            try:
                exchange_word = banmal_dic[exchange_word]
            except KeyError:
                print("not include banmal dictionary")
                pass

            temp_dec_input[i][idx] = LABEL.vocab.stoi[exchange_word]
            temp_dec_input[i][idx + 1] = LABEL.vocab.stoi['<eos>']

            for k in range(idx+2, args.max_len):
                temp_dec_input[i][k] = LABEL.vocab.stoi['<pad>']

            # for j in range(len(temp_dec_input[i])):
            #     if LABEL.vocab.itos[temp_dec_input[i][j]]=='<eos>':
            #         break
            #     print(LABEL.vocab.itos[temp_dec_input[i][j]], end='')
            # print()

        dec_input = torch.tensor(temp_dec_input, dtype=torch.int32).cuda()

    return enc_input, dec_input, dec_outputs

# 반말로 바꾸기위한 딕셔너리
def get_rough_dic():
    my_exword = {
        '돌아와요': '돌아와',
        '으세요': '으셈',
        '잊어버려요': '잊어버려',
        '나온대요': '나온대',
        '될까요': '될까',
        '할텐데': '할텐데',
        '옵니다': '온다',
        '봅니다': '본다',
        '네요': '네',
        '된답니다': '된대',
        '데요': '데',
        '봐요': '봐',
        '부러워요': '부러워',
        '바랄게요': '바랄게',
        '지나갑니다': "지가간다",
        '이뻐요': "이뻐",
        '지요': "지",
        '사세요': "사라",
        '던가요': "던가",
        '모릅니다': "몰라",
        '은가요': "은가",
        '심해요': "심해",
        '몰라요': "몰라",
        '라요': "라",
        '더라고요': '더라고',
        '입니다': '이라고',
        '는다면요': '는다면',
        '멋져요': '멋져',
        '다면요': '다면',
        '다니': '다나',
        '져요': '져',
        '만드세요': '만들어',
        '야죠': '야지',
        '죠': '지',
        '해줄게요': '해줄게',
        '대요': '대',
        '돌아갑시다': '돌아가자',
        '해보여요': '해봐',
        '라뇨': '라니',
        '편합니다': '편해',
        '합시다': '하자',
        '드세요': '먹어',
        '아름다워요': '아름답네',
        '드립니다': '줄게',
        '받아들여요': '받아들여',
        '건가요': '간기',
        '쏟아진다': '쏟아지네',
        '슬퍼요': '슬퍼',
        '해서요': '해서',
        '다릅니다': '다르다',
        '니다': '니',
        '내려요': '내려',
        '마셔요': '마셔',
        '아세요': '아냐',
        '변해요': '뱐헤',
        '드려요': '드려',
        '아요': '아',
        '어서요': '어서',
        '뜁니다': '뛴다',
        '속상해요': '속상해',
        '래요': '래',
        '까요': '까',
        '어야죠': '어야지',
        '라니': '라니',
        '해집니다': '해진다',
        '으련만': '으련만',
        '지워져요': '지워져',
        '잘라요': '잘라',
        '고요': '고',
        '셔야죠': '셔야지',
        '다쳐요': '다쳐',
        '는구나': '는구만',
        '은데요': '은데',
        '일까요': '일까',
        '인가요': '인가',
        '아닐까요': '아닐까',
        '텐데요': '텐데',
        '할게요': '할게',
        '보입니다': '보이네',
        '에요': '야',
        '걸요': '걸',
        '한답니다': '한대',
        '을까요': '을까',
        '못해요': '못해',
        '베푸세요': '베풀어',
        '어때요': '어떄',
        '더라구요': '더라구',
        '노라': '노라',
        '반가워요': '반가워',
        '군요': '군',
        '만납시다': '만나자',
        '어떠세요': '어때',
        '달라져요': '달라져',
        '예뻐요': '예뻐',
        '됩니다': '된다',
        '봅시다': '보자',
        '한대요': '한대',
        '싸워요': '싸워',
        '와요': '와',
        '인데요': '인데',
        '야': '야',
        '줄게요': '줄게',
        '기에요': '기',
        '던데요': '던데',
        '걸까요': '걸까',
        '신가요': '신가',
        '어요': '어',
        '따져요': '따져',
        '갈게요': '갈게',
        '봐': '봐',
        '나요': '나',
        '니까요': '니까',
        '마요': '마',
        '씁니다': '쓴다',
        '집니다': '진다',
        '건데요': '건데',
        '지웁시다': '지우자',
        '바랍니다': '바래',
        '는데요': '는데',
        '으니까요': '으니까',
        '셔요': '셔',
        '네여': '네',
        '달라요': '달라',
        '거려요': '거려',
        '보여요': '보여',
        '겁니다': '껄',
        '다': '다',
        '그래요': '그래',
        '한가요': '한가',
        '잖아요': '잖아',
        '한데요': '한데',
        '우세요': '우셈',
        '해야죠': '해야지',
        '세요': '셈',
        '걸려요': '걸려',
        '텐데': '텐데',
        '어딘가': '어딘가',
        '요': '',
        '흘러갑니다': '흘러간다',
        '줘요': '줘',
        '편해요': '편해',
        '거예요': '거야',
        '예요': '야',
        '습니다': '어',
        '아닌가요': '아닌가',
        '합니다': '한다',
        '사라집니다': '사라져',
        '드릴게요': '줄게',
        '다면': '다면',
        '그럴까요': '그럴까',
        '해요': '해',
        '답니다': '다',
        '주무세요': '자라',
        '마세요': '마라',
        '아픈가요': '아프냐',
        '그런가요': '그런가',
        '했잖아요': '했잖아',
        '버려요': '버려',
        '갑니다': '간다',
        '가요': '가',
        '라면요': '라면',
        '아야죠': '아야지',
        '살펴봐요': '살펴봐',
        '남겨요': '남겨',
        '내려놔요': '내려놔',
        '떨려요': '떨려',
        '랍니다': '란다',
        '돼요': '돼',
        '버텨요': '버텨',
        '만나': '만나',
        '일러요': '일러',
        '을게요': '을게',
        '갑시다': '가자',
        '나아요': '나아',
        '어려요': '어려',
        '온대요': '온대',
        '다고요': '다고',
        '할래요': '할래',
        '된대요': '된대',
        '어울려요': '어울려',
        '는군요': '는군',
        '볼까요': '볼까',
        '드릴까요': '줄까',
        '라던데요': '라던데',
        '올게요': '올게',
        '기뻐요': '기뻐',
        '아닙니다': '아냐',
        '둬요': '둬',
        '십니다': '십',
        '아파요': '아파',
        '생겨요': '생겨',
        '해줘요': '해줘',
        '로군요': '로군요',
        '시켜요': '시켜',
        '느껴져요': '느껴져',
        '가재요': '가재',
        '어 ': ' ',
        '느려요': '느려',
        '볼게요': '볼게',
        '쉬워요': '쉬워',
        '나빠요': '나빠',
        '불러줄게요': '불러줄게',
        '살쪄요': '살쪄',
        '봐야겠어요': '봐야겠어',
        '네': '네',
        '어': '어',
        '든지요': '든지',
        '드신다': '드심',
        '가져요': '가져',
        '할까요': '할까',
        '졸려요': '졸려',
        '그럴게요': '그럴게',
        '': '',
        '어린가': '어린가',
        '나와요': '나와',
        '빨라요': '빨라',
        '겠죠': '겠지',
        '졌어요': '졌어',
        '해봐요': '해봐',
        '게요': '게',
        '해드릴까요': '해줄까',
        '인걸요': '인걸',
        '했어요': '했어',
        '원해요': '원해',
        '는걸요': '는걸',
        '좋아합니다': '좋아해',
        '했으면': '했으면',
        '나갑니다': '나간다',
        '왔어요': '왔어',
        '해봅시다': '해보자',
        '물어봐요': '물어봐',
        '생겼어요': '생겼어',
        '해': '해',
        '다녀올게요': '다녀올게',
        '납시다': '나자'
    }
    return my_exword