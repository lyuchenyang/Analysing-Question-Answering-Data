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


def process_squad_json():
    train_file = 'question-answering/squad_1.1/train.json'
    dev_file = 'question-answering/squad_1.1/dev.json'
    with open(train_file, 'r') as f:
        train_js = []
        for line in f.readlines():
            js = json.loads(line)
            train_js.append(js)
        train_js = {'data': train_js, 'version:': 1.1}
        json_dump(train_js,
                  'question-answering/squad_1.1/train.json')

    with open(dev_file, 'r') as f:
        train_js = []
        for line in f.readlines():
            js = json.loads(line)
            train_js.append(js)
        train_js = {'data': train_js, 'version:': 1.1}
        json_dump(train_js,
                  'question-answering/squad_1.1/dev.json')
    # js = json_load('question-answering/squad_1.1/train.json')
    # js_1 = json_load('question-answering/squad_1.1/dev.json')


def process_question_classification():
    root_dir = 'question_classification/'
    train_file = 'train.txt'

    dev_file = 'dev.txt'

    def process_file(file, postfix):
        with open(file + postfix, 'r') as f:
            labels = []
            texts = []

            for line in f.readlines():
                line = line.split(' ')
                label = line[0].split(':')[0].strip()
                text = ' '.join(line[1:]).strip()
                labels.append(label)
                texts.append(text)
        dic = {'text': texts, 'label': labels}
        df = pd.DataFrame(data=dic)
        df.to_csv(file + postfix.split('.')[0] + '.csv', index=False)

    process_file(root_dir, train_file)
    process_file(root_dir, dev_file)
    return


def combine_qa_for_question_type_pred():
    dataset_name = 'newsqa'

    train_file = 'question-answering/' \
                 + dataset_name + '/train.json'
    dev_file = 'question-answering/' \
               + dataset_name + '/dev.json'

    def process_qa(file):
        js = json_load(file)

        ids = []
        labels = []
        texts = []
        for d in js['data']:
            id = d['id']
            question = d['question']

            ids.append(id)
            labels.append('DESC')
            texts.append(question)
        dic = {'text': texts, 'label': labels}
        df = pd.DataFrame(data=dic)
        df.to_csv(file.split('.json')[0] + '_qtype.csv', index=False)

    process_qa(train_file)
    process_qa(dev_file)


def inject_qtype_cls():
    dataset_name = 'newsqa'
    train_file = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    preds_dir = '/tmp/newsqa_1/'

    def process_qtype(file, file_type):
        df = pd.read_csv(file)

        qtype_preds = pd.read_csv(preds_dir + 'preds_newsqa_' + file_type + '.txt', sep='\t')
        # with open(preds_dir + 'preds_squad_' + file_type + '.txt', 'r') as f:
        #     lines = f.readlines()
        #     lines = [line.strip() for line in lines]

        preds = list(qtype_preds['prediction'])
        df['qtype'] = preds
        df.to_csv(file, index=False)

    process_qtype(train_file, 'train')
    process_qtype(dev_file, 'dev')


def return_longest(lis):
    lens = [len(e) for e in lis]

    return lens.index(max(lens))


def transform_qa_to_huggingface_format():
    qa_file = 'cmrc/cmrc2018_trial.json'

    qa_data = json_load(qa_file)['data']

    new_qas = []

    for qas in qa_data:
        for paragraph in qas['paragraphs']:
            for qa in paragraph['qas']:
                n_qas = {'id': qa['id'], 'title': qas['title'], 'context': paragraph['context'],
                         'question': qa['question'], 'answers': {}}

                n_qas['answers']['text'] = [e['text'] for e in qa['answers']]
                n_qas['answers']['answer_start'] = [e['answer_start'] for e in qa['answers']]

                new_qas.append(n_qas)

    new_qas = {'data': new_qas, 'version': 1.1}
    json_dump(new_qas, qa_file)

    return


def transform_newsqa():
    root_dir = 'question-answering/newsqa/'

    train = root_dir + 'NewsQA_train.jsonl'
    dev = root_dir + 'NewsQA_dev.jsonl'

    def transform(file, mode):
        with open(file, 'r') as f:
            qas = []
            for line in tqdm(f.readlines()[1:]):
                js = json.loads(line)
                for e in js['qas']:
                    qa = {'id': e['qid'], 'title': "article", 'context': js['context'], 'question': e['question']}
                    ans_text, ans_starts = [], []
                    for ans in e['answers']:
                        if ans in js['context']:
                            ans_text.append(ans)
                            ans_starts.append(js['context'].index(ans))
                    if len(ans_text) == 0:
                        continue
                    qa['answers'] = {'text': ans_text, 'answer_start': ans_starts}
                    qas.append(qa)
            print(mode + ' sample: ', len(qas))
            qas = {'data': qas, 'version:': 1.1}
            json_dump(qas,
                      'question-answering/newsqa/' + mode + '.json')

    transform(train, 'train')
    transform(dev, 'dev')


def transform_to_squad_like():
    dataset_name = 'squad_1.1'
    train_file = 'question-answering/' + dataset_name + '/train.json'
    train_csv = 'question-answering/' + dataset_name + '/train_qtype.csv'
    dev_file = 'question-answering/' + dataset_name + '/sub_datasets/dev_single_answer.json'
    dev_csv = 'question-answering/' + dataset_name + '/dev_qtype.csv'

    def transform(file_dir):
        data = json_load(file_dir)['data']

        new_qas = []
        for qas in data:
            n_qas = {'paragraphs': [], 'title': qas['title']}
            qa = {'context': qas['context'], 'qas': [
                {'answers': [], 'id': qas['id'], 'question': qas['question']}
            ]}

            n_ans = []
            for ans, ans_start in zip(qas['answers']['text'], qas['answers']['answer_start']):
                n_an = {'text': ans, 'answer_start': ans_start}
                n_ans.append(n_an)
            qa['qas'][0]['answers'] = n_ans
            n_qas['paragraphs'].append(qa)

            new_qas.append(n_qas)

        qa_data = {'data': new_qas, 'version': 1.1}
        json_dump(qa_data, file_dir.split('.json')[0] + '_official.json')

    # transform(train_file)
    transform(dev_file)
 

def sample_qa_data():
    dataset_name = 'squad_1.1'
    train_file = 'question-answering/' + dataset_name + '/train.json'
    dev_file = 'question-answering/' + dataset_name + '/dev.json'

    def export_qas(qas_data, output_dir):
        data = {'data': qas_data, 'version': 1.1}
        json_dump(data, output_dir)

    def draw_qa_asmples(qas, out_dir):
        total = 0
        for i in tqdm(range(1, 20)):
            total += len(qas) * i * 0.05
            # n_qas = draw_samples(qas, i*0.05)
            # export_qas(n_qas, out_dir + str(round(i*0.05, 2)) + '.json')
        print(total)

    train_qas = json_load(train_file)['data']
    # dev_qas = json_load(dev_file)['data']

    draw_qa_asmples(train_qas, 'question-answering/' + dataset_name + '/sub_datasets/train_')
