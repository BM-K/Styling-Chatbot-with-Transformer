import torch
import torch.nn.functional as F
from math import log
from numpy import array
from get_data import tokenizer1
from torch.autograd import Variable
from chatspace import ChatSpace
spacer = ChatSpace()
from konlpy.tag import Mecab
import re

def tokenizer1(text):
    result_text = re.sub('[-=+.,#/\:$@*\"※&%ㆍ!?』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
    a = Mecab().morphs(result_text)
    return ([a[i] for i in range(len(a))])

def _get_length_penalty(text, alpha=1.2, min_length=5):
    p_list = []
    for i in range(len(text)):
        temp_text = tokenizer1(text[i][0])
        length = len(temp_text) 
        p_list.append(((5 + length) ** alpha) / (5 + 1) ** alpha)
    
    lp_list = [ text[j][1]/p_list[j]  for j in range(len(text)) ]
    return lp_list

def compair_beam_and_greedy(beam_pair, greedy_pair):
    lp_list = _get_length_penalty(beam_pair)
    gr_lp_list = _get_length_penalty(greedy_pair)

    low_val = float('inf')
    checked_sen = ""
    for idx in range(len(beam_pair)):
        if lp_list[idx] < low_val:
            low_val = lp_list[idx]
            checked_sen = beam_pair[idx][0]
    
    print("  beam output > ", checked_sen, " |", low_val)
    print("greedy output > ", greedy_pair[0][0]," |", gr_lp_list[0])
    if low_val < gr_lp_list[0]:
        print("use beam")
    else:
        print("use greedy")

def cal_score(pred, score):
    
    pred = F.softmax(pred, dim=-1)
    pred_ids = pred.max(dim=-1)[0]
    pred_ids = pred_ids.to('cpu').tolist()
    score = score * -log(pred_ids)

    return score

def Beam_Search(data, k, first, sequences):
    #sequences = [[list(), 1.0]]
    
    if first:
        data = data.squeeze(0)
    else:
        data = data.unsqueeze(0)
    data = F.softmax(data, dim=-1)

    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                no_tensor_row = row[j].to('cpu').tolist()
                candidate = [seq + [j], score * -log(no_tensor_row)]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]

    return(sequences)

def beam(args, dec_input, enc_input_index, model, first, device, k_, LABEL):
    temp_dec_input = torch.zeros([k_,1], dtype=torch.long)
    temp_dec_input = temp_dec_input + dec_input
    deliver_high_beam_value = torch.zeros([k_,1], dtype=torch.long)
    return_sentence_beamVal_pair = []
    check_k = [float('inf')] * k_
    sequences = [[list(), 1.0]]
    end_sentence = []
    end_idx = []

    if first:
        y_pred = model(enc_input_index.to(device), dec_input.to(device))
        first_beam_sequence = Beam_Search(y_pred, k_, True, sequences)
    
    for i in range(len(deliver_high_beam_value)):
        deliver_high_beam_value[i] = first_beam_sequence[i][0][0]
    
    temp_dec_input = torch.cat(
            [temp_dec_input.to(torch.device('cpu')),
                deliver_high_beam_value.to(torch.device('cpu'))], dim=-1)
    
    check_num = 0
    beam_input_sequence = first_beam_sequence
    
    for i in range(args.max_len):
        which_value = [float('inf')] * k_
        which_node = [0] * k_

        for j in range(len(temp_dec_input)):
            if temp_dec_input[j][-1] == torch.LongTensor([3]):
                continue
            y_pred = model(enc_input_index.to(device), temp_dec_input[j].unsqueeze(0).to(device))
            beam_seq = Beam_Search(y_pred.squeeze(0)[-1], k_, False, [beam_input_sequence[j]])
            
            beam_input_sequence[j] = [[beam_seq[0][0][-1]], beam_seq[0][1]]
            which_node[j] = beam_seq[0][0][-1] # k개의 output중 누적확률 높은 거 get
        
        for l in range(len(deliver_high_beam_value)):
            if temp_dec_input[j][-1] == torch.LongTensor([3]):
                continue
            deliver_high_beam_value[l] = which_node[l]
        
        temp_dec_input = torch.cat(
                            [temp_dec_input.to(torch.device('cpu')),
                                                deliver_high_beam_value.to(torch.device('cpu'))], dim=-1)
        
        for x in range(len(temp_dec_input)):
            for y in range(len(temp_dec_input[x])):
                if temp_dec_input[x][y] == torch.LongTensor([3]) and check_k[x] == float('inf'):
                    check_k[x] = beam_input_sequence[x][1]

        if i+1 == args.max_len:
            for k in range(k_):
                for kk in range(len(temp_dec_input[k])):
                    if temp_dec_input[k][kk] == torch.LongTensor([3]):
                        check_num += 1
                        end_sentence.append(temp_dec_input[k])
                        end_idx.append(k)
                        break
            
            for l in range(len(end_sentence)):
                pred = []
                for idx in range(len(end_sentence[l])):
                
                    if end_sentence[l][idx] == torch.LongTensor([3]):
                        pred_sentence = "".join(pred)
                        pred_str = spacer.space(pred_sentence)
                        #print(pred_str, " |", check_k[end_idx[l]])
                        return_sentence_beamVal_pair.append([pred_str, check_k[end_idx[l]]])
                        break
                    else:
                        if idx == 0:
                            continue
                        pred.append(LABEL.vocab.itos[end_sentence[l][idx]])   
            return return_sentence_beamVal_pair

def inference(device, args, TEXT, LABEL, model, sa_model):
    from KoBERT.Sentiment_Analysis_BERT_main import bert_inference
    sentence = input("문장을 입력하세요 : ")
    se_list = [sentence]

    # https://github.com/SKTBrain/KoBERT
    # SKT 에서 공개한 KoBert Sentiment Analysis 를 통해 입력문장의 긍정 부정 판단.
    sa_label = int(bert_inference(sa_model, se_list))

    sa_token = ''
    # SA Label 에 따른 encoder input 변화.
    if sa_label == 0:
        sa_token = TEXT.vocab.stoi['<nega>']
    else:
        sa_token = TEXT.vocab.stoi['<posi>']

    enc_input = tokenizer1(sentence)
    enc_input_index = []

    for tok in enc_input:
        enc_input_index.append(TEXT.vocab.stoi[tok])

    # encoder input string to index tensor and plus <pad>
    if args.per_soft:
        enc_input_index.append(sa_token)

    for j in range(args.max_len - len(enc_input_index)):
        enc_input_index.append(TEXT.vocab.stoi['<pad>'])

    enc_input_index = Variable(torch.LongTensor([enc_input_index]))

    dec_input = torch.LongTensor([[LABEL.vocab.stoi['<sos>']]])
    #print("긍정" if sa_label == 1 else "부정")

    model.eval()
    pred = []
    
    beam_k = 10
    beam_sen_val_pair = beam(args, dec_input, enc_input_index, model, True, device, beam_k, LABEL)
    greedy_pair = []
    for i in range(args.max_len):
        y_pred = model(enc_input_index.to(device), dec_input.to(device))
        if i == 0:
            score = cal_score(y_pred.squeeze(0)[-1], 1.0)
        else:
            score = cal_score(y_pred.squeeze(0)[-1], score )
        
        y_pred_ids = y_pred.max(dim=-1)[1]
        
        if (y_pred_ids[0, -1] == LABEL.vocab.stoi['<eos>']):
            y_pred_ids = y_pred_ids.squeeze(0)
            #print(">", end=" ")
            for idx in range(len(y_pred_ids)):
                if LABEL.vocab.itos[y_pred_ids[idx]] == '<eos>':
                    pred_sentence = "".join(pred)
                    pred_str = spacer.space(pred_sentence)
                    #print(pred_str, " |", score)
                    greedy_pair.append([pred_str, score])
                    break
                else:
                    pred.append(LABEL.vocab.itos[y_pred_ids[idx]])
            
            compair_beam_and_greedy(beam_sen_val_pair, greedy_pair)
            return 0

        dec_input = torch.cat(
            [dec_input.to(torch.device('cpu')),
             y_pred_ids[0, -1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)
    
    
