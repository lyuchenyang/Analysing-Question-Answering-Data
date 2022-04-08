import pandas as pd
import json
import codecs
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def inspect_qtype():
    dataset_name = 'newsqa'
    train_file = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    def inspect(file):
        df = pd.read_csv(file)
        qtypes = list(df['qtype'])

        total = len(qtypes)
        for qt in set(qtypes):
            print(qt, qtypes.count(qt) / total)

        print('===================')

    inspect(train_file)
    inspect(dev_file)


def inspect_lexical_overlap():
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

    dataset_name = 'newsqa'
    train_file = 'question-answering/' + dataset_name + '/train.json'
    train_csv = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/dev.json'
    dev_csv = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    def cal_bleu(file, csv_file):
        qa_data = json_load(file)['data']
        df = pd.read_csv(csv_file)

        contexts = [[qas['context']] for qas in qa_data]
        questions = [qas['question'] for qas in qa_data]
        # bleu = corpus_bleu(contexts, questions)
        # print(bleu)
        total_bleu = 0
        bleus = []
        # chencherry = SmoothingFunction()
        for qas in tqdm(qa_data):
            ans = qas['answers']['text'][0]

            context = qas['context']
            for sent in nltk.sent_tokenize(context):
                if ans in sent:
                    context = sent
                    break

            reference = context.lower().split()
            hypothesis = qas['question'].lower().split()
            bleu = sentence_bleu([reference], hypothesis, weights=[0.33, 0.33, 0.33])
            total_bleu += bleu
            bleus.append(bleu * 100)
        print(total_bleu / len(qa_data))

        df['bleu'] = bleus

        df.to_csv(csv_file, index=False)

    cal_bleu(train_file, train_csv)
    cal_bleu(dev_file, dev_csv)


def inspect_answer_length():
    train_file = 'question-answering/squad_1.1/train.json'
    dev_file = 'question-answering/squad_1.1/dev.json'

    def inspect_ans(file):
        qa_data = json_load(file)['data']

        length_dict = {}

        for qas in qa_data:
            len_ans = len(qas['answers']['text'][0].split())

            if len_ans not in length_dict:
                length_dict[len_ans] = 0
            length_dict[len_ans] += 1

        for key in length_dict:
            length_dict[key] = length_dict[key] / len(qa_data)
        length_dict = sorted(length_dict.items(), key=lambda x: x[1], reverse=True)
        # print(length_dict)

        print(length_dict)
        return length_dict

    inspect_ans(train_file)


def inspect_qas_length_distribution(tokenize_mode='split', dataset_name=None):
    '''
    10.061096587860591 10 7.296934158278696 12.76489849377865 43316 44283 87599
    119.74879850226601 120 88.68054582262711 163.8053937670798 51372 36227 87599
    3.161908240961655 3 1.4243409791167407 5.770566231689558 52578 35021 87599
    10.216650898770103 10 7.301317122593718 12.7698314108252 4935 5635 10570
    123.87322611163671 124 91.51576112783145 171.8583509513742 6313 4257 10570
    3.016461684011353 3 1.456490727532097 5.326214503637644 6309 4261 10570
    11.28639596342424 11 8.324852569502948 13.957984105615147 41545 46054 87599
    137.8319615520725 138 102.10452351292432 188.50509344891367 51376 36223 87599
    3.3728467219945433 3 1.4370792436630437 6.101782570422535 51247 36352 87599
    11.44314096499527 11 8.322328042328042 13.965953806672369 4725 5845 10570
    141.6682119205298 142 104.61860761510276 195.84020498485907 6277 4293 10570
    3.1884578997161777 3 1.460974417467818 5.579968418678096 6137 4433 10570
    '''
    dataset_name = 'newsqa' if dataset_name is None else dataset_name
    root = ""
    train_file = root + "question-answering/" + dataset_name + "/train.json"
    dev_file = root + "question-answering/" + dataset_name + "/dev.json"

    def average_len_distribution(lens):
        avg_len = sum([i * lens[i] for i in range(1, len(lens))]) / sum(lens)

        round_avg_len = round(avg_len)
        lower_count = sum(lens[:round_avg_len])
        higher_count = sum(lens[round_avg_len:])

        lower_avg = sum([i * lens[i] for i in range(1, round_avg_len)]) / lower_count
        higher_avg = sum([i * lens[i] for i in range(round_avg_len, len(lens))]) / higher_count

        print(avg_len, round_avg_len, lower_avg, higher_avg, lower_count, higher_count, sum(lens))

    def draw_hist(array, name):
        # the histogram of the data
        # plt.hist(array, 1, density=True, facecolor='g')
        sum_array = sum(array)
        array = [float(a / sum_array) for a in array]
        plt.plot(range(len(array)), array)

        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title('Histogram of {} Text Length Distribution'.format(name))
        # plt.xlim(40, 160)
        # plt.grid(True)
        plt.show()

    def word_tokenize(s, mode='tokenize'):
        if mode == 'split':
            return s.split()
        else:
            return nltk.word_tokenize(s)

    def inspect(file_name):
        qas = json_load(file_name)['data']

        qlen, clen, alen = [0] * 40, [0] * 500, [0] * 31
        for qa in qas:
            q, c, a = qa['question'], qa['context'], qa['answers']['text'][0]

            qlen[min(len(word_tokenize(q, tokenize_mode)), 39)] += 1
            clen[min(len(word_tokenize(c, tokenize_mode)), 499)] += 1
            alen[min(len(word_tokenize(a, tokenize_mode)), 29)] += 1

        average_len_distribution(qlen)
        average_len_distribution(clen)
        average_len_distribution(alen)

        # draw_hist(qlen, 'Question')
        # draw_hist(clen, 'Context')
        # draw_hist(alen, 'Answer')

    inspect(train_file)
    inspect(dev_file)
    

def inspect_text_len_and_ans_pos_by_qtype(dataset_name=None):
    dataset_name = 'newsqa' if dataset_name is None else dataset_name
    root = 'question-answering/'
    train = root + dataset_name + '/train_by_qtype.json'
    dev = root + dataset_name + '/dev_by_qtype.json'

    qas = json_load(train)

    all_qas = []
    for key in qas:
        all_qas.extend(qas[key])
    qas['all'] = all_qas
    for key in qas:
        contexts, qs, ans = [e['context'].split() for e in qas[key]], [e['question'].split() for e in qas[key]], \
                            [e['answers']['text'][0].split() for e in qas[key]]
        c_avg_len = round(np.mean([len(c) for c in contexts]),2)
        c_median_len = np.median([len(c) for c in contexts])
        q_avg_len = round(np.mean([len(q) for q in qs]),2)
        q_median_len = np.median([len(q) for q in qs])
        a_avg_len = round(np.mean([len(a) for a in ans]),2)
        a_median_len = np.median([len(a) for a in ans])

        _, _, char_pos = inspect_ans_char_pos(qas[key])
        _, _, word_pos = inspect_ans_word_pos(qas[key])
        _, _, sent_pos = inspect_ans_sent_pos(qas[key])

        char_pos_ = [[e[1] for e in char_pos if e[1] != -1]]
        char_avg_pos = round(np.mean(char_pos_), 2)
        char_median_pos = np.median(char_pos_)

        word_pos_ = [e[1] for e in word_pos if e[1] != -1]
        word_avg_pos = round(np.mean(word_pos_), 2)
        word_median_pos = np.median(word_pos_)

        sent_pos_ = [e[1] for e in sent_pos if e[1] != -1]
        sent_avg_pos = round(np.mean(sent_pos_), 2)
        sent_median_pos = np.median(sent_pos_)


        print(dataset_name, key, c_avg_len, q_avg_len, a_avg_len, c_median_len, q_median_len, a_median_len,
              char_avg_pos, word_avg_pos, sent_avg_pos,
              char_median_pos, word_median_pos, sent_median_pos)


def inspect_text_len_ans_pos_qtype(dataset_name=None):
    dataset_name = 'newsqa' if dataset_name is None else dataset_name
    root = 'question-answering/'
    train = root + dataset_name + '/train.json'
    dev = root + dataset_name + '/dev.json'
    train_qtype = root + dataset_name + '/train_qtype.csv'

    all_qas = json_load(train)['data']

    df = pd.read_csv(train_qtype)
    train_qtypes = list(df['qtype'])

    key = 'all'
    qas = {}
    qas[key] = all_qas

    contexts, qs, ans = [e['context'].split() for e in qas[key]], [e['question'].split() for e in qas[key]], \
                        [e['answers']['text'][0].split() for e in qas[key]]

    c_lens = [len(c) for c in contexts]
    c_avg_len = round(np.mean(c_lens), 2)
    c_median_len = np.median(c_lens)
    q_lens = [len(q) for q in qs]
    q_avg_len = round(np.mean(q_lens), 2)
    q_median_len = np.median(q_lens)
    a_lens = [len(a) for a in ans]
    a_avg_len = round(np.mean(a_lens), 2)
    a_median_len = np.median(a_lens)

    _, _, char_pos = inspect_ans_char_pos(qas[key])
    _, _, word_pos = inspect_ans_word_pos(qas[key])
    _, _, sent_pos = inspect_ans_sent_pos(qas[key])

    char_pos_ = [e[1] for e in char_pos]
    char_avg_pos = round(np.mean(char_pos_), 2)
    char_median_pos = np.median(char_pos_)

    word_pos_ = [e[1] for e in word_pos]
    word_avg_pos = round(np.mean(word_pos_), 2)
    word_median_pos = np.median(word_pos_)

    sent_pos_ = [e[1] for e in sent_pos]
    sent_avg_pos = round(np.mean(sent_pos_), 2)
    sent_median_pos = np.median(sent_pos_)

    def inspect_qtype_by_len(qas, keys, qtypes, median_):
        qtype_by_len = {'Long': {}, 'Short': {}}
        for q, key, qt in zip(qas, keys, qtypes):
            if key == -1:
                continue
            if key <= median_:
                len_key = 'Short'
            else:
                len_key = 'Long'
            if qt not in qtype_by_len[len_key]:
                qtype_by_len[len_key][qt] = 0
            qtype_by_len[len_key][qt] += 1

        for k1 in qtype_by_len:
            total = sum(qtype_by_len[k1][k2] for k2 in qtype_by_len[k1])
            for k2 in qtype_by_len[k1]:
                qtype_by_len[k1][k2] = round((qtype_by_len[k1][k2]/total)*100, 2)
        return qtype_by_len,

    def inspect_qtype_by_ans_pos(qas, keys, qtypes, median_):
        qtype_by_ans_pos = {'Front': {}, 'Back': {}}
        for q, key, qt in zip(qas, keys, qtypes):
            if key == -1:
                continue

            if key <= median_:
                len_key = 'Front'
            else:
                len_key = 'Back'
            if qt not in qtype_by_ans_pos[len_key]:
                qtype_by_ans_pos[len_key][qt] = 0
            qtype_by_ans_pos[len_key][qt] += 1

        for k1 in qtype_by_ans_pos:
            total = sum(qtype_by_ans_pos[k1][k2] for k2 in qtype_by_ans_pos[k1])
            for k2 in qtype_by_ans_pos[k1]:
                qtype_by_ans_pos[k1][k2] = round((qtype_by_ans_pos[k1][k2]/total)*100, 2)
        return qtype_by_ans_pos

    def inspect_ans_pos_by_text_len(text_lens, char_pos, word_pos, sent_pos, median_):
        ans_pos_by_len = {'Long': {'Char': 0, 'Word': 0, 'Sent': 0},
                          'Short': {'Char': 0, 'Word': 0, 'Sent': 0}}
        total_c = {'Long': 0, 'Short': 0}
        for text_len, char_p, word_p, sent_p in zip(text_lens, char_pos, word_pos, sent_pos):
            if text_len <= median_:
                key = 'Short'
            else:
                key = 'Long'
            total_c[key] += 1
            ans_pos_by_len[key]['Char'] += char_p
            ans_pos_by_len[key]['Word'] += word_p
            ans_pos_by_len[key]['Sent'] += sent_p

        for k1 in ans_pos_by_len:
            for k2 in ans_pos_by_len[k1]:
                ans_pos_by_len[k1][k2] = round((ans_pos_by_len[k1][k2]/total_c[k1]), 2)

        return  ans_pos_by_len

    def inspect_text_len_by_ans_pos(ans_pos, context_lens, question_lens, ans_lens, median_):
        text_len_by_ans_pos = {'Front': {'Context': 0, 'Question': 0, 'Answer': 0},
                          'Back': {'Context': 0, 'Question': 0, 'Answer': 0}}
        total_c = {'Front': 0, 'Back': 0}
        for a_pos, c_len, q_len, a_len in zip(ans_pos, context_lens, question_lens, ans_lens):
            if a_pos <= median_:
                key = 'Front'
            else:
                key = 'Back'
            total_c[key] += 1
            text_len_by_ans_pos[key]['Context'] += c_len
            text_len_by_ans_pos[key]['Question'] += q_len
            text_len_by_ans_pos[key]['Answer'] += a_len

        for k1 in text_len_by_ans_pos:
            for k2 in text_len_by_ans_pos[k1]:
                text_len_by_ans_pos[k1][k2] = round((text_len_by_ans_pos[k1][k2] / total_c[k1]), 2)

        return text_len_by_ans_pos

    def inspect_ans_pos_by_ans_pos(ans_pos, char_pos, word_pos, sent_pos, median_):
        ans_pos_by_ans_pos = {'Front': {'Char': 0, 'Word': 0, 'Sent': 0},
                          'Back': {'Char': 0, 'Word': 0, 'Sent': 0}}
        total_c = {'Front': 0, 'Back': 0}
        for ans_p, char_p, word_p, sent_p in zip(ans_pos, char_pos, word_pos, sent_pos):
            if ans_p <= median_:
                key = 'Front'
            else:
                key = 'Back'
            total_c[key] += 1
            ans_pos_by_ans_pos[key]['Char'] += char_p
            ans_pos_by_ans_pos[key]['Word'] += word_p
            ans_pos_by_ans_pos[key]['Sent'] += sent_p

        for k1 in ans_pos_by_ans_pos:
            for k2 in ans_pos_by_ans_pos[k1]:
                ans_pos_by_ans_pos[k1][k2] = round((ans_pos_by_ans_pos[k1][k2]/total_c[k1]), 2)

        return  ans_pos_by_ans_pos

    def inspect_text_len_by_text_len(text_lens, context_lens, question_lens, ans_lens, median_):
        text_len_by_text_len = {'Long': {'Context': 0, 'Question': 0, 'Answer': 0},
                          'Short': {'Context': 0, 'Question': 0, 'Answer': 0}}
        total_c = {'Long': 0, 'Short': 0}
        for t_len, c_len, q_len, a_len in zip(text_lens, context_lens, question_lens, ans_lens):
            if t_len <= median_:
                key = 'Long'
            else:
                key = 'Short'
            total_c[key] += 1
            text_len_by_text_len[key]['Context'] += c_len
            text_len_by_text_len[key]['Question'] += q_len
            text_len_by_text_len[key]['Answer'] += a_len

        for k1 in text_len_by_text_len:
            for k2 in text_len_by_text_len[k1]:
                text_len_by_text_len[k1][k2] = round((text_len_by_text_len[k1][k2] / total_c[k1]), 2)

        return text_len_by_text_len


    qtype_c_len = inspect_qtype_by_len(all_qas, c_lens, train_qtypes, c_median_len)
    qtype_q_len = inspect_qtype_by_len(all_qas, q_lens, train_qtypes, q_median_len)
    qtype_a_len = inspect_qtype_by_len(all_qas, a_lens, train_qtypes, a_median_len)

    qtype_char_pos = inspect_qtype_by_ans_pos(all_qas, char_pos_, train_qtypes, char_median_pos)
    qtype_word_pos = inspect_qtype_by_ans_pos(all_qas, word_pos_, train_qtypes, word_median_pos)
    qtype_sent_pos = inspect_qtype_by_ans_pos(all_qas, sent_pos_, train_qtypes, sent_median_pos)

    ans_pos_by_c_lens = inspect_ans_pos_by_text_len(c_lens, char_pos_, word_pos_, sent_pos_, c_median_len)
    ans_pos_by_q_lens = inspect_ans_pos_by_text_len(q_lens, char_pos_, word_pos_, sent_pos_, q_median_len)
    ans_pos_by_a_lens = inspect_ans_pos_by_text_len(a_lens, char_pos_, word_pos_, sent_pos_, a_median_len)

    text_len_by_char_pos = inspect_text_len_by_ans_pos(char_pos_, c_lens, q_lens, a_lens, char_median_pos)
    text_len_by_word_pos = inspect_text_len_by_ans_pos(word_pos_, c_lens, q_lens, a_lens, word_median_pos)
    text_len_by_sent_pos = inspect_text_len_by_ans_pos(sent_pos_, c_lens, q_lens, a_lens, sent_median_pos)

    ans_pos_by_char_pos = inspect_ans_pos_by_ans_pos(char_pos_, char_pos_, word_pos_, sent_pos_, char_median_pos)
    ans_pos_by_word_pos = inspect_ans_pos_by_ans_pos(word_pos_, char_pos_, word_pos_, sent_pos_, word_median_pos)
    ans_pos_by_sent_pos = inspect_ans_pos_by_ans_pos(sent_pos_, char_pos_, word_pos_, sent_pos_, sent_median_pos)

    text_len_by_c_lens = inspect_text_len_by_text_len(c_lens, c_lens, q_lens, a_lens, c_median_len)
    text_len_by_q_lens = inspect_text_len_by_text_len(q_lens, c_lens, q_lens, a_lens, q_median_len)
    text_len_by_a_lens = inspect_text_len_by_text_len(a_lens, c_lens, q_lens, a_lens, a_median_len)




    return (qtype_c_len, qtype_q_len, qtype_a_len), (qtype_char_pos, qtype_word_pos, qtype_sent_pos), \
           (ans_pos_by_c_lens, ans_pos_by_q_lens, ans_pos_by_a_lens), \
           (text_len_by_char_pos, text_len_by_word_pos, text_len_by_sent_pos), \
           (ans_pos_by_char_pos, ans_pos_by_word_pos, ans_pos_by_sent_pos), \
           (text_len_by_c_lens, text_len_by_q_lens, text_len_by_a_lens)
