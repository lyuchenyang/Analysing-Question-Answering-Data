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

def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def organise_by_qtype():
    dataset_name = 'newsqa'

    train_file = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    train = 'question-answering/' + dataset_name + '/train.json'
    dev = 'question-answering/' + dataset_name + '/dev.json'

    def org_by_qtype(qas_dir, qtypes_dir):
        df = pd.read_csv(qtypes_dir)
        qtypes = list(df['qtype'])
        qas = json_load(qas_dir)['data']

        qas_by_qt = {}
        for qa, qtype in zip(qas, qtypes):
            if qtype not in qas_by_qt:
                qas_by_qt[qtype] = []
            qas_by_qt[qtype].append(qa)

        json_dump(qas_by_qt, qas_dir.split('.json')[0] + '_by_qtype.json')

    org_by_qtype(train, train_file)
    org_by_qtype(dev, dev_file)


def export_qas_by_overlap():
    dataset_name = 'newsqa'
    train_file = 'question-answering/' + dataset_name + '/train.json'
    train_csv = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/dev.json'
    dev_csv = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    train_qas = json_load(train_file)['data']
    train = pd.read_csv(train_csv)

    tb = np.array(list(train['bleu']))

    tbmean = np.median(tb)

    tl_ind, tm_ind = [i for i in range(len(tb)) if tb[i] < tbmean], [i for i in range(len(tb)) if tb[i] >= tbmean]

    tl_qas, tm_qas = [train_qas[i] for i in tl_ind], [train_qas[i] for i in tm_ind]

    def draw_qas_samples(all_qas, output_dir):
        for i in tqdm(range(1, 25)):
            export_qas(draw_samples(all_qas, i * 250), output_dir + str(i * 250) + '.json')

    draw_qas_samples(tm_qas, 'question-answering/squad_1.1/'
                             'sub_datasets/lexical_overlap/newsqa_train_more_overlap_')

    draw_qas_samples(tl_qas, 'question-answering/squad_1.1/'
                             'sub_datasets/lexical_overlap/newsqa_train_less_overlap_')


def export_qas_by_length(dataset_name=None, key=None):
    dataset_name = 'newsqa' if dataset_name is None else dataset_name
    key = 'context' if key is None else key

    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'
    dev_file = root + 'question-answering/' + dataset_name + '/dev.json'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    train_qas = json_load(train_file)['data']

    if key == 'answer':
        tb = np.array([len(e['answers']['text'][0].split()) for e in train_qas])
    else:
        tb = np.array([len(e[key].split()) for e in train_qas])

    tbmean = np.median(tb)
    print(tbmean, dataset_name, key)

    tl_ind, ts_ind = [i for i in range(len(tb)) if tb[i] >= tbmean], [i for i in range(len(tb)) if tb[i] < tbmean]

    tl_qas, ts_qas = [train_qas[i] for i in tl_ind], [train_qas[i] for i in ts_ind]

    base = 500

    def draw_qas_samples(all_qas, output_dir):
        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(all_qas, i * base), output_dir + str(i * base) + '.json')

    draw_qas_samples(tl_qas, root + 'question-answering/' + dataset_name + \
                     '/sub_datasets/length/' + dataset_name + '_train_' + key + '_long_')

    draw_qas_samples(ts_qas, root + 'question-answering/' + dataset_name + \
                     '/sub_datasets/length/' + dataset_name + '_train_' + key + '_short_')


def export_qas_overall(dataset_name=None):
    # dataset_name = 'newsqa'

    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    train_qas = json_load(train_file)['data']

    base = 500

    def draw_qas_samples(all_qas, output_dir):
        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(all_qas, i * base), output_dir + str(i * base) + '.json')

    draw_qas_samples(train_qas, root + 'question-answering/' + dataset_name + \
                     '/sub_datasets/' + dataset_name + '_train_overall_')


def export_qas_overall_proportion(dataset_name=None):
    # dataset_name = 'newsqa'

    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    train_qas = json_load(train_file)['data']

    base = 0.05

    def draw_qas_samples(all_qas, output_dir):
        total = 0
        for i in tqdm(range(1, 21)):
            total += (len(all_qas)*i*base)
            export_qas(draw_samples(all_qas, i * base), output_dir + str(i * base) + '.json')
        print(total/25000)

    draw_qas_samples(train_qas, root + 'question-answering/' + dataset_name + \
                     '/sub_datasets/' + dataset_name + '_train_overall_proportion_')


def inspect_answer_position(dataset_name=None):
    dataset_name = 'squad_1.1' if dataset_name is None else dataset_name
    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'
    dev_file = root + 'question-answering/' + dataset_name + '/dev.json'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    def inspect_ans_char_pos(qas):
        contexts = [e['context'] for e in qas]
        ans_starts = [e['answers']['answer_start'][0] for e in qas]

        avg_pos = round(sum(ans_starts) / len(ans_starts))
        positions = [0] * 6 * avg_pos

        for ans in ans_starts:
            if ans >= (6 * avg_pos):
                ans = 6 * avg_pos - 1
            positions[ans] += 1
        
        median_pos = np.median(ans_starts)
        return positions, median_pos, [[e[0], e[1]] for e in zip(qas, ans_starts)]

    def inspect_ans_word_pos(qas):
        contexts = [e['context'] for e in qas]
        answers = [e['answers']['text'][0] for e in qas]

        def tokenize(lis, mode='word_tokenize'):
            if mode == 'split':
                lis = [e.split() for e in lis]
            else:
                lis = [nltk.word_tokenize(e) for e in lis]
            return lis

        contexts = tokenize(contexts)
        answers = tokenize(answers)

        avg_len = round(sum([len(context) for context in contexts]) / len(contexts))

        positions = [0] * 2 * avg_len

        def inspect_word_pos(context, ans):
            for i in range(len(context) - len(ans) + 1):
                if context[i:i + len(ans)] == ans:
                    return i

            return -1

        not_in_word = 0
        new_qas = []
        for i, (c, a) in enumerate(zip(contexts, answers)):
            pos = inspect_word_pos(c, a)
            if pos == -1:
                not_in_word += 1
                # continue
            if pos >= (2 * avg_len):
                positions[2 * avg_len - 1] += 1
            else:
                positions[pos] += 1
            new_qas.append([qas[i], pos])
        print('not in word: ', not_in_word / len(contexts))
        median_word_pos = np.median([e[1] for e in new_qas])
        return positions, median_word_pos, new_qas

    def inspect_ans_sent_pos(qas):
        contexts = [e['context'] for e in qas]
        answers = [e['answers']['text'][0] for e in qas]

        def tokenize(lis):
            lis = [nltk.sent_tokenize(e) for e in lis]
            return lis

        contexts = tokenize(contexts)
        # answers = tokenize(answers)

        avg_len = round(sum([len(context) for context in contexts]) / len(contexts))

        positions = [0] * 2 * avg_len

        def inspect_word_pos(context, ans):
            for i in range(len(context)):
                if ans in context[i]:
                    return i
            return -1

        not_in = 0
        new_qas = []
        for i, (c, a) in tqdm(enumerate(zip(contexts, answers))):
            pos = inspect_word_pos(c, a)
            if pos == -1:
                not_in += 1
                # continue

            if pos >= (2 * avg_len):
                positions[2 * avg_len - 1] += 1
            else:
                positions[pos] += 1
            new_qas.append([qas[i], pos])
        print('not in sentence', not_in / len(contexts))
        median_word_pos = np.median([e[1] for e in new_qas])
        return positions, median_word_pos, new_qas

    def draw_hist(array, name):
        # the histogram of the data
        # plt.hist(array, 1, density=True, facecolor='g')
        sum_array = sum(array)
        array = [float(a / sum_array) for a in array]
        plt.plot(range(len(array)), array)

        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title('Histogram of {} Answer Position Distribution'.format(name))
        # plt.xlim(40, 160)
        # plt.grid(True)
        plt.show()

    def export_by_avg_pos(qas, avg_pos, base=500, mode='char', output_dir=root):
        front_qas, back_qas = [e[0] for e in qas if e[1] <= avg_pos], [e[0] for e in qas if e[1] > avg_pos]

        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(front_qas, i * base), output_dir + dataset_name +
                       '_train_' + mode + '_front_' + str(i * base) + '.json')

        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(back_qas, i * base), output_dir + dataset_name +
                       '_train_' + mode + '_back_' + str(i * base) + '.json')
        print(avg_pos, len(front_qas), len(back_qas), dataset_name)

    train_qas = json_load(train_file)['data']
    # draw_hist(inspect_ans_char_pos(train_qas)[0], 'Char')
    # draw_hist(inspect_ans_word_pos(train_qas)[0], 'Word')
    # draw_hist(inspect_ans_sent_pos(train_qas)[0], 'Sentence')

    _, char_median_pos, char_qas = inspect_ans_char_pos(train_qas)
    _, word_median_pos, word_qas = inspect_ans_word_pos(train_qas)
    _, sent_median_pos, sent_qas = inspect_ans_sent_pos(train_qas)

    # df = pd.read_csv(root + 'question-answering/' + dataset_name + '/dev_qtype.csv')
    # df['ans_char_pos'] = [e[1] for e in char_qas]
    # df['ans_word_pos'] = [e[1] for e in word_qas]
    # df['ans_sent_pos'] = [e[1] for e in sent_qas]
    #
    # df.to_csv(root + 'question-answering/' + dataset_name + '/dev_qtype.csv', index=False)

    output_dir = root + 'question-answering/' + dataset_name + '/sub_datasets/ans_position/'
    export_by_avg_pos(char_qas, char_median_pos, mode='char', output_dir=output_dir)
    export_by_avg_pos(word_qas, word_median_pos, mode='word', output_dir=output_dir)
    export_by_avg_pos(sent_qas, sent_median_pos, mode='sent', output_dir=output_dir)


def inspect_answer_position_float(dataset_name=None):
    dataset_name = 'squad_1.1' if dataset_name is None else dataset_name
    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'
    dev_file = root + 'question-answering/' + dataset_name + '/dev.json'

    multi_factor = 1 if dataset_name == 'squad_1.1' else 2
    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    def inspect_ans_char_pos(qas):
        contexts = [e['context'] for e in qas]
        ans_starts = [e['answers']['answer_start'][0] for e in qas]
        ans_texts = [e['answers']['text'][0] for e in qas]

        pos_indicators = []

        for context, ans_start, ans in zip(contexts, ans_starts, ans_texts):
            if round((len(context)-len(ans))/2) > (ans_start*multi_factor):
                pos_indicators.append(0)
            else:
                pos_indicators.append(1)

        return [[e[0], e[1]] for e in zip(qas, pos_indicators)]

    def inspect_ans_word_pos(qas):
        contexts = [e['context'] for e in qas]
        answers = [e['answers']['text'][0] for e in qas]

        def tokenize(lis, mode='word_tokenize'):
            if mode == 'split':
                lis = [e.split() for e in lis]
            else:
                lis = [nltk.word_tokenize(e) for e in lis]
            return lis

        contexts = tokenize(contexts)
        answers = tokenize(answers)


        def inspect_word_pos(context, ans):
            for i in range(len(context) - len(ans) + 1):
                if context[i:i + len(ans)] == ans:
                    return i

            return -1

        not_in_word = 0
        new_qas = []
        for i, (c, a) in enumerate(zip(contexts, answers)):
            pos = inspect_word_pos(c, a)
            if pos == -1:
                not_in_word += 1
                # continue
            if round((len(c)-len(a))/2) > (pos*multi_factor):
                new_qas.append([qas[i], 0])
            else:
                new_qas.append([qas[i], 1])
        print('not in word: ', not_in_word / len(contexts))
        return new_qas

    def inspect_ans_sent_pos(qas):
        contexts = [e['context'] for e in qas]
        answers = [e['answers']['text'][0] for e in qas]

        def tokenize(lis):
            lis = [nltk.sent_tokenize(e) for e in lis]
            return lis

        contexts = tokenize(contexts)

        def inspect_word_pos(context, ans):
            for i in range(len(context)):
                if ans in context[i]:
                    return i
            return -1

        not_in = 0
        new_qas = []
        for i, (c, a) in tqdm(enumerate(zip(contexts, answers))):
            pos = inspect_word_pos(c, a)
            if pos == -1:
                not_in += 1

            if round(len(c)/2) > (pos*multi_factor):
                new_qas.append([qas[i], 0])
            else:
                new_qas.append([qas[i], 1])
        print('not in sentence', not_in / len(contexts))
        return new_qas

    def export_by_avg_pos(qas, base=500, mode='char', output_dir=root):
        front_qas, back_qas = [e[0] for e in qas if e[1] == 0], [e[0] for e in qas if e[1] == 1]

        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(front_qas, i * base), output_dir + dataset_name +
                       '_train_' + mode + '_front_' + str(i * base) + '_float_new.json')

        for i in tqdm(range(1, 51)):
            export_qas(draw_samples(back_qas, i * base), output_dir + dataset_name +
                       '_train_' + mode + '_back_' + str(i * base) + '_float_new.json')
        print(len(front_qas), len(back_qas), mode, dataset_name)

    train_qas = json_load(train_file)['data']

    char_qas = inspect_ans_char_pos(train_qas)
    word_qas = inspect_ans_word_pos(train_qas)
    sent_qas = inspect_ans_sent_pos(train_qas)

    # df = pd.read_csv(root + 'question-answering/' + dataset_name + '/dev_qtype.csv')
    # df['ans_char_pos_float_new'] = [e[1] for e in char_qas]
    # df['ans_word_pos_float_new'] = [e[1] for e in word_qas]
    # df['ans_sent_pos_float_new'] = [e[1] for e in sent_qas]
    #
    # df.to_csv(root + 'question-answering/' + dataset_name + '/dev_qtype.csv', index=False)
    #
    # print([e[1] for e in char_qas].count(0), [e[1] for e in char_qas].count(1), len(char_qas))
    # print([e[1] for e in word_qas].count(0), [e[1] for e in word_qas].count(1), len(word_qas))
    # print([e[1] for e in sent_qas].count(0), [e[1] for e in sent_qas].count(1), len(sent_qas))

    output_dir = root + 'question-answering/' + dataset_name + '/sub_datasets/ans_position/'
    export_by_avg_pos(char_qas, mode='char', output_dir=output_dir)
    export_by_avg_pos(word_qas, mode='word', output_dir=output_dir)
    export_by_avg_pos(sent_qas, mode='sent', output_dir=output_dir)
    
if __name__ == '__main__':
    data_sets = ['newsqa', 'squad_1.1']
    keys = ['context', 'question', 'answer']
    for ds in data_sets:
        for key in keys:
            export_qas_by_length(ds, key)
