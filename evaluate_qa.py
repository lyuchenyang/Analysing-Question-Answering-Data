"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import pandas as pd
import codecs
# import nltk
from tqdm import tqdm
from collections import OrderedDict

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('--data_file', required=False, metavar='data.json', help='Input data JSON file.')
    parser.add_argument('--pred_file', required=False, metavar='pred.json', help='Model predictions.')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                        help='Model estimates of probability of no answer.')
    parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')
    parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                        help='Save precision-recall curves to directory.')
    parser.add_argument('--verbose', '-v', action='store_true')
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = OrderedDict()
    f1_scores = OrderedDict()
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    # print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not isinstance(exact_scores, OrderedDict):
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores) / total),
            ('f1', 100.0 * sum(f1_scores) / total),
            ('total', total),
        ])
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def plot_pr_curve(precisions, recalls, out_image, title):
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i + 1)
        cur_r = true_pos / float(num_true_pos)
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i + 1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {'ap': 100.0 * avg_prec}


def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs,
                                  qid_to_has_ans, out_image_dir):
    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_exact.png'),
        title='Precision-Recall curve for Exact Match score')
    pr_f1 = make_precision_recall_eval(
        f1_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_f1.png'),
        title='Precision-Recall curve for F1 score')
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
        title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')
    merge_eval(main_eval, pr_exact, 'pr_exact')
    merge_eval(main_eval, pr_f1, 'pr_f1')
    merge_eval(main_eval, pr_oracle, 'pr_oracle')


def histogram_na_prob(na_probs, qid_list, image_dir, name):
    if not qid_list:
        return
    x = [na_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel('Model probability of no-answer')
    plt.ylabel('Proportion of dataset')
    plt.title('Histogram of no-answer probability: %s' % name)
    plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
    plt.clf()


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores: continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def eval_all_proportion():
    if OPTS.data_file and OPTS.pred_file:
        with open(OPTS.data_file) as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']
        with open(OPTS.pred_file) as f:
            preds = json.load(f)
    else:
        dataset_name = 'squad_1.1'
        with open('question-answering/' + dataset_name + '/dev_official.json') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        keys = ['HUM', 'LOC', 'ENTY', 'DESC', 'NUM', 'ABBR']

        eval_results = {}
        for i in tqdm(range(5), desc='Evaluating keys...'):
            eval_results[keys[i]] = {}
            for r in range(1, 17):
                eval_results[keys[i]][str(r * 500)] = {}
                pred_file = 'question-answering/eval_results/' + 'preds_' + keys[i] + '_' + str(r * 500) + '.json'

                all_preds = json_load(pred_file)

                dev_by_qtype_file = 'question-answering/' + dataset_name + '/dev_by_qtype.json'
                dev_qtype = json_load(dev_by_qtype_file)

                all_preds_by_qtype = {}

                for key in dev_qtype:
                    all_preds_by_qtype[key] = {}

                    for e in dev_qtype[key]:
                        # assign predicted answer to corresponding key and id
                        all_preds_by_qtype[key][e['id']] = all_preds[e['id']]

                all_preds_by_qtype['Overall'] = all_preds

                # print(all_preds_by_qtype.keys())
                for key in all_preds_by_qtype:
                    eval_results[keys[i]][str(r * 500)][key] = {}
                    # print('===================')
                    # print('scores of key: ', key)
                    preds = all_preds_by_qtype[key]
                    if len(preds) == 0:
                        continue
                    if OPTS.na_prob_file:
                        with open(OPTS.na_prob_file) as f:
                            na_probs = json.load(f)
                    else:
                        na_probs = {k: 0.0 for k in preds}

                    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
                    exact_raw, f1_raw = get_raw_scores(dataset, preds)
                    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                                          OPTS.na_prob_thresh)
                    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                                       OPTS.na_prob_thresh)
                    out_eval = make_eval_dict(exact_thresh, f1_thresh)

                    eval_results[keys[i]][str(r * 500)][key]['em'] = out_eval['exact']
                    eval_results[keys[i]][str(r * 500)][key]['f1'] = out_eval['f1']
                    # print(out_eval[0])
                    # if OPTS.out_file:
                    #     with open(OPTS.out_file, 'w') as f:
                    #         json.dump(out_eval, f)
                    # else:
                    #     print(json.dumps(out_eval, indent=2))

        json_dump(eval_results,
                  'question-answering/eval_results/qtype_squad_proportion_eval_results.json')


def eval_all_lexical_overlap():
    if OPTS.data_file and OPTS.pred_file:
        with open(OPTS.data_file) as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']
        with open(OPTS.pred_file) as f:
            preds = json.load(f)
    else:
        dataset_name = 'newsqa'
        with open('question-answering/' + dataset_name + '/dev_official.json') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        train_csv = 'question-answering/' + dataset_name + '/train_qtype.csv'
        dev_csv = 'question-answering/' + dataset_name + '/dev_qtype.csv'

        dev = pd.read_csv(dev_csv)
        db = np.array(list(dev['bleu']))

        dbmeann = np.median(db)

        dl_ind, dm_ind = [i for i in range(len(db)) if db[i] < dbmeann], [i for i in range(len(db)) if db[i] >= dbmeann]

        keys = ['less_overlap', 'more_overlap']
        interval = 250
        eval_results = {}
        for i in tqdm(range(2), desc='Evaluating keys...'):
            eval_results[keys[i]] = {}
            for r in tqdm(range(1, 25)):
                eval_results[keys[i]][str(r * interval)] = {}
                pred_file = 'question-answering/eval_results/' + \
                            'preds_newsqa_train_' + keys[i] + '_' + str(r * interval) + '.json'

                all_preds = json_load(pred_file)

                eval_results[keys[i]][str(r * interval)] = {}
                eval_results[keys[i]][str(r * interval)]['Overall'] = {}
                eval_results[keys[i]][str(r * interval)]['Less Overlap'] = {}
                eval_results[keys[i]][str(r * interval)]['More Overlap'] = {}

                qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
                exact_raw, f1_raw = get_raw_scores(dataset, all_preds)
                em_keys, f1_keys = list(exact_raw.keys()), list(f1_raw.keys())
                all_eval = make_eval_dict(exact_raw, f1_raw)
                less_eval = make_eval_dict([exact_raw[em_keys[i]] for i in dl_ind],
                                           [f1_raw[em_keys[i]] for i in dl_ind])
                more_eval = make_eval_dict([exact_raw[em_keys[i]] for i in dm_ind],
                                           [f1_raw[em_keys[i]] for i in dm_ind])

                eval_results[keys[i]][str(r * interval)]['Overall']['em'] = all_eval['exact']
                eval_results[keys[i]][str(r * interval)]['Overall']['f1'] = all_eval['f1']

                eval_results[keys[i]][str(r * interval)]['Less Overlap']['em'] = less_eval['exact']
                eval_results[keys[i]][str(r * interval)]['Less Overlap']['f1'] = less_eval['f1']

                eval_results[keys[i]][str(r * interval)]['More Overlap']['em'] = more_eval['exact']
                eval_results[keys[i]][str(r * interval)]['More Overlap']['f1'] = more_eval['f1']

        json_dump(eval_results,
                  'question-answering/eval_results/qtype_newsqa_lexical_overlap_eval_results.json')


def eval_all_proportion_plus_overall():
    if OPTS.data_file and OPTS.pred_file:
        with open(OPTS.data_file) as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']
        with open(OPTS.pred_file) as f:
            preds = json.load(f)
    else:
        dataset_name = 'newsqa'
        with open('question-answering/' + dataset_name + '/dev_official.json') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        keys = ['HUM', 'LOC', 'ENTY', 'DESC', 'NUM', 'ABBR']

        eval_results = json_load('question-answering/eval_results/qtype_newsqa_proportion_eval_results.json')
        dev_by_qtype_file = 'question-answering/' + dataset_name + '/dev_by_qtype.json'
        dev_qtype = json_load(dev_by_qtype_file)

        eval_results['Overall'] = {}
        for r in range(1, 20):
            eval_results['Overall'][str(round(r * 0.05, 2))] = {}
            pred_file = 'question-answering/eval_results/' + 'preds_newsqa_train_' + str(round(r * 0.05, 2)) + '.json'

            all_preds = json_load(pred_file)

            all_preds_by_qtype = {}

            for key in dev_qtype:
                all_preds_by_qtype[key] = {}

                for e in dev_qtype[key]:
                    # assign predicted answer to corresponding key and id
                    all_preds_by_qtype[key][e['id']] = all_preds[e['id']]

            all_preds_by_qtype['Overall'] = all_preds

            # print(all_preds_by_qtype.keys())
            for key in all_preds_by_qtype:
                eval_results['Overall'][str(round(r * 0.05, 2))][key] = {}
                # print('===================')
                # print('scores of key: ', key)
                preds = all_preds_by_qtype[key]
                if len(preds) == 0:
                    continue
                if OPTS.na_prob_file:
                    with open(OPTS.na_prob_file) as f:
                        na_probs = json.load(f)
                else:
                    na_probs = {k: 0.0 for k in preds}

                qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
                exact_raw, f1_raw = get_raw_scores(dataset, preds)
                exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                                      OPTS.na_prob_thresh)
                f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                                   OPTS.na_prob_thresh)
                out_eval = make_eval_dict(exact_thresh, f1_thresh)

                eval_results['Overall'][str(round(r * 0.05, 2))][key]['em'] = out_eval['exact']
                eval_results['Overall'][str(round(r * 0.05, 2))][key]['f1'] = out_eval['f1']
                # print(out_eval[0])
                # if OPTS.out_file:
                #     with open(OPTS.out_file, 'w') as f:
                #         json.dump(out_eval, f)
                # else:
                #     print(json.dumps(out_eval, indent=2))

        json_dump(eval_results,
                  'question-answering/eval_results/qtype_newsqa_proportion_eval_results_plus_overall.json')


def eval_all_length(dataset_name=None, main_key=None):
    # dataset_name = 'newsqa'
    # main_key = 'context'

    train_median_lens = {'squad_1.1':{'context': 110, 'question': 10, 'answer': 2},
                      'newsqa':{'context': 534, 'question': 6, 'answer': 2}
    }

    with open('question-answering/' + dataset_name + '/dev_official.json') as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    
    root = ''
    train_file = 'question-answering/' + dataset_name + '/train.json'
    dev_file = 'question-answering/' + dataset_name + '/dev.json'

    dev_qas = json_load(dev_file)['data']

    if main_key is 'answer':
        tb = np.array([len(e['answers']['text'][0].split()) for e in dev_qas])
    else:
        tb = np.array([len(e[main_key].split()) for e in dev_qas])

    tbmean = train_median_lens[dataset_name][main_key]

    tl_ind, ts_ind = [i for i in range(len(tb)) if tb[i] >= tbmean], [i for i in range(len(tb)) if tb[i] < tbmean]


    keys = ['Long', 'Short']
    interval = 500
    eval_results = {}
    for i in tqdm(range(2), desc='Evaluating keys...'):
        eval_results[keys[i]] = {}
        for r in tqdm(range(1, 51)):
            eval_results[keys[i]][str(r * interval)] = {}
            pred_file = root + 'question-answering/eval_results/' + \
                        'preds_' + dataset_name + '_train_' + main_key + '_' + \
                        keys[i].lower() + '_' + str(r * interval) + '.json'
            # preds_newsqa_train_answer_short_22500

            all_preds = json_load(pred_file)

            eval_results[keys[i]][str(r * interval)] = {}
            eval_results[keys[i]][str(r * interval)]['Overall'] = {}
            eval_results[keys[i]][str(r * interval)]['Long'] = {}
            eval_results[keys[i]][str(r * interval)]['Short'] = {}

            qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
            exact_raw, f1_raw = get_raw_scores(dataset, all_preds)
            em_keys, f1_keys = list(exact_raw.keys()), list(f1_raw.keys())
            all_eval = make_eval_dict(exact_raw, f1_raw)
            long_eval = make_eval_dict([exact_raw[em_keys[i]] for i in tl_ind],
                                       [f1_raw[em_keys[i]] for i in tl_ind])
            short_eval = make_eval_dict([exact_raw[em_keys[i]] for i in ts_ind],
                                       [f1_raw[em_keys[i]] for i in ts_ind])

            eval_results[keys[i]][str(r * interval)]['Overall']['em'] = all_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Overall']['f1'] = all_eval['f1']

            eval_results[keys[i]][str(r * interval)]['Long']['em'] = long_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Long']['f1'] = long_eval['f1']

            eval_results[keys[i]][str(r * interval)]['Short']['em'] = short_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Short']['f1'] = short_eval['f1']

    json_dump(eval_results,
              root + 'question-answering/eval_results/' + dataset_name + '_' + main_key +
              '_length_eval_results.json')


def eval_all_answer_position(dataset_name=None, main_key=None):
    # dataset_name = 'newsqa'
    # main_key = 'char', 'word', 'sent'

    train_median_pos = {'squad_1.1':{'char': 262, 'word': 46, 'sent': 1},
                      'newsqa':{'char': 358, 'word': 67, 'sent': 2}
    }

    with open('question-answering/' + dataset_name + '/dev_official.json') as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']

    root = ""
    train_file = root + 'question-answering/' + dataset_name + '/train.json'
    dev_file = root + 'question-answering/' + dataset_name + '/dev.json'


    df = pd.read_csv(root + 'question-answering/' + dataset_name + '/dev_qtype.csv')
    dev_pos = list(df['ans_' + main_key + '_pos'])

    dev_pos_mean = train_median_pos[dataset_name][main_key]

    dp_f_ind, dp_b_ind = [i for i in range(len(dev_pos)) if dev_pos[i] <= dev_pos_mean and dev_pos[i]!=-1], \
                         [i for i in range(len(dev_pos)) if dev_pos[i] > dev_pos_mean and dev_pos[i]!=-1]

    keys = ['Front', 'Back']
    interval = 500 if dataset_name == 'newsqa' else 250
    eval_results = {}
    for i in tqdm(range(2), desc='Evaluating keys...'):
        eval_results[keys[i]] = {}
        for r in tqdm(range(1, 51)):
            eval_results[keys[i]][str(r * interval)] = {}
            pred_file = root + 'question-answering/eval_results/' + \
                        'preds_' + dataset_name + '_train_' + main_key + '_' + \
                        keys[i].lower() + '_' + str(r * interval) + '.json'
            # preds_newsqa_train_answer_short_22500

            all_preds = json_load(pred_file)

            eval_results[keys[i]][str(r * interval)] = {}
            eval_results[keys[i]][str(r * interval)]['Overall'] = {}
            eval_results[keys[i]][str(r * interval)]['Front'] = {}
            eval_results[keys[i]][str(r * interval)]['Back'] = {}

            qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
            exact_raw, f1_raw = get_raw_scores(dataset, all_preds)
            em_keys, f1_keys = list(exact_raw.keys()), list(f1_raw.keys())
            all_eval = make_eval_dict(exact_raw, f1_raw)
            front_eval = make_eval_dict([exact_raw[em_keys[i]] for i in dp_f_ind],
                                       [f1_raw[em_keys[i]] for i in dp_f_ind])
            back_eval = make_eval_dict([exact_raw[em_keys[i]] for i in dp_b_ind],
                                       [f1_raw[em_keys[i]] for i in dp_b_ind])

            eval_results[keys[i]][str(r * interval)]['Overall']['em'] = all_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Overall']['f1'] = all_eval['f1']

            eval_results[keys[i]][str(r * interval)]['Front']['em'] = front_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Front']['f1'] = front_eval['f1']

            eval_results[keys[i]][str(r * interval)]['Back']['em'] = back_eval['exact']
            eval_results[keys[i]][str(r * interval)]['Back']['f1'] = back_eval['f1']

    json_dump(eval_results,
              root + 'question-answering/eval_results/' + dataset_name + '_' + main_key +
              '_ans_pos_eval_results.json')


if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.out_image_dir:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    # eval_all_proportion()
    # eval_all_lexical_overlap()
    # eval_all_proportion_plus_overall()

    data_sets = ['newsqa', 'squad_1.1']
    keys = ['context', 'question', 'answer']
    for ds in data_sets:
        for key in keys:
            eval_all_length(ds, key)
    
#     data_sets = ['squad_1.1', 'newsqa']
#     keys = ['char', 'word', 'sent']
#     for ds in data_sets:
#         for key in keys:
#             eval_all_answer_position(ds, key)

