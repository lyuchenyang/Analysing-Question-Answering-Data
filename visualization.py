from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn import manifold
import numpy as np

def visualize_perofrmance_by_qtype_merge_view():
    eval_results_dir = 'question-answering/eval_results/qtype_newsqa_proportion_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("EM (upper) and F-1 Score (lower) Change Over Different Question Type with Increased Data Size",
                 fontsize=14)

    all_keys = list(eval_results.keys()) + ['Overall']
    # all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 5, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['em'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 5, i + 6)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    plt.show()
    # plt.savefig(output_dir)


def visualize_perofrmance_by_qtype_merge_dataset():
    eval_results_dir = 'question-answering/eval_results/qtype_squad_proportion_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    # fig.suptitle("F-1 Score Change Over Different Question Type with Increased Data Size on SQuAD1.1 (top) and NewsQA (bottom)",
    #              fontsize=14)

    all_keys = list(eval_results.keys()) + ['Overall']
    # all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 5, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='x-small')

    eval_results_dir = 'question-answering/eval_results/qtype_newsqa_proportion_eval_results.json'

    eval_results = json_load(eval_results_dir)

    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 5, i + 6)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='x-small')

    plt.show()
    # plt.savefig(output_dir)


def visualize_performance_by_lexical_overlap_merge_view():
    eval_results_dir = 'question-answering/eval_results/qtype_squad_lexical_overlap_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        "EM (upper) and F-1 Score (lower) Change Over Different Lexical Overlap Level with Increased Data Size",
        fontsize=14)

    all_keys = list(eval_results.keys())
    all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 2, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['em'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 2, i + 3)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    plt.show()
    # plt.savefig(output_dir)


def visualize_performance_by_lexical_overlap_merge_dataset():
    eval_results_dir = 'question-answering/eval_results/qtype_squad_lexical_overlap_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(8, 8))
    # fig.suptitle(
    #     "F-1 Score Change Over Different Lexical Overlap Level \n with Increased Data Size on SQuAD1.1 (top) and NewsQA (bottom)",
    #     fontsize=14)

    all_keys = list(eval_results.keys())
    all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 2, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        sub_title = 'Less Overlap' if key == 'less_overlap' else 'More Overlap'
        ax.set_title(sub_title)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    eval_results_dir = 'question-answering/eval_results/qtype_newsqa_lexical_overlap_eval_results.json'

    eval_results = json_load(eval_results_dir)

    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(2, 2, i + 3)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        sub_title = 'Less Overlap' if key == 'less_overlap' else 'More Overlap'
        ax.set_title(sub_title)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right', fontsize='small')

    plt.show()
    # plt.savefig(output_dir)


def visualize_performance_by_lexical_overlap():
    eval_results_dir = 'question-answering/eval_results/qtype_newsqa_lexical_overlap_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("EM Change Over Different Lexical Overlap with Increased Data Size", fontsize=14)

    all_keys = list(eval_results.keys())
    all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(1, 2, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['em'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.legend(loc='lower right')

    plt.show()
    # plt.savefig(output_dir)


def visualize_performance_by_qtype():
    eval_results_dir = 'question-answering/eval_results/qtype_newsqa_proportion_eval_results_plus_overall.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("F-1 Score Change Over Different Question Type with Increased Data Size", fontsize=14)

    all_keys = list(eval_results.keys())
    # all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    for i, (key) in enumerate(all_keys):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(1, 6, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:
            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_ylim([0, 100])
        # ax.axis('tight')
        ax.legend(loc='lower right')

    plt.ylim(0, 100)
    plt.show()
    # plt.savefig(output_dir)


def visualize_performance_by_qas_length(dataset_name=None, main_key=None):
    dataset_name = 'newsqa' if dataset_name is None else dataset_name
    main_key = 'context' if main_key is None else main_key

    root = ""
    eval_results_dir = root + 'question-answering/eval_results/' + \
                       dataset_name + '_' + main_key + '_length_eval_results.json'

    eval_results = json_load(eval_results_dir)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("F-1 Change Over Different {} Length with Increased Data Size on {}".
                 format(main_key.capitalize(), dataset_name),
                 fontsize=14)

    all_keys = list(eval_results.keys())
    all_keys = ['Overall', 'Long', 'Short']
    for i, (key) in enumerate(eval_results):
        # main_key
        # data size
        # sub_key
        # EM, F-1
        ax = fig.add_subplot(1, 2, i + 1)
        x = list(eval_results[key].keys())

        perf_by_key = {}
        for k in all_keys:
            perf_by_key[k] = []
        for data_size in eval_results[key]:

            for sub_key in all_keys:
                perf_by_key[sub_key].append(eval_results[key][data_size][sub_key]['f1'])

        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'g', 'r']
        for sub_key, color in zip(perf_by_key.keys(), colors):
            ax.plot(x, perf_by_key[sub_key], c=color, label=sub_key)

        ax.set_title(key)
        ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')

        if dataset_name == 'newsqa':
            ax.set_ylim([0, 80])
        else:
            ax.set_ylim([0, 90])
        ax.legend(loc='lower right')

    plt.show()
    # output_dir = root + ''
    # plt.savefig(output_dir)


def visualize_performance_by_contrastive(mode=None):
    mode = 'text_len' if mode is None else mode

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig = plt.figure(figsize=(15, 8))
    # fig.suptitle("Performance Change Over Different Answer Position with Increased Data Size", fontsize=14)

    root = ""

    dataset_name = ['squad_1.1', 'newsqa']
    if mode == 'text_len':
        main_keys = ['context', 'question', 'answer']
    if mode == 'ans_pos':
        main_keys = ['char', 'word', 'sent']

    metrics = ['em', 'f1']

    metric_names = {'em': 'EM', 'f1': 'F-1'}
    dataset_names = {'squad_1.1': 'SQuAD1.1', 'newsqa': 'NewsQA'}

    if mode == 'text_len':
        all_keys = ['Overall', 'Long', 'Short']
    if mode == 'ans_pos':
        all_keys = ['Overall', 'Front', 'Back']
    all_main_keys = all_keys[1:]

    i = 0
    for ds in dataset_name:
        for main_key in main_keys:
            if mode == 'text_len':
                eval_results_dir = root + 'question-answering/eval_results/' + \
                                   ds + '_' + main_key + '_length_eval_results.json'
            if mode == 'ans_pos':
                eval_results_dir = root + 'question-answering/eval_results/' + \
                                   ds + '_' + main_key + '_ans_pos_eval_results.json'

            eval_results = json_load(eval_results_dir)

            for metric in metrics:
                ax = fig.add_subplot(2, 6, i + 1)
                i += 1
                perf_by_key_all = {}
                for query_key in all_main_keys:
                    x = list(eval_results[query_key].keys())
                    perf_by_key = {}

                    for k in all_keys:
                        perf_by_key[k] = []

                    for data_size in eval_results[query_key]:
                        for sub_key in all_keys:
                            perf_by_key[sub_key].append(eval_results[query_key][data_size][sub_key][metric])
                    perf_by_key_all[query_key] = perf_by_key

                perf_by_key_constrast = {}

                for sub_key in all_keys:
                    perf_by_key_constrast[sub_key] = [e1/e2 for e1,e2 in
                                                      zip(perf_by_key_all[all_main_keys[0]][sub_key],
                                                          perf_by_key_all[all_main_keys[1]][sub_key])]

                    # if main_key != 'answer':
                    #     kernel_size = 5
                    #     kernel = np.ones(kernel_size) / kernel_size
                    #     perf_by_key_constrast[sub_key] = np.convolve(perf_by_key_constrast[sub_key],
                    #                                                  kernel, mode='same')

                # colors = ['b', 'g', 'r', 'c', 'm', 'y']
                colors = ['b', 'g', 'r']
                for sub_key, color in zip(perf_by_key_constrast.keys(), colors):
                    print(np.mean(perf_by_key_constrast[sub_key]), metric, sub_key, main_key, ds)
                    ax.plot(x, perf_by_key_constrast[sub_key], c=color, label=sub_key)

                ax.plot(x, [1]*len(x), c='k', linestyle='dotted')

                ax.set_title(metric_names[metric] + ' ' + main_key.capitalize() + ' ' + dataset_names[ds])
                ax.xaxis.set_major_formatter(NullFormatter())
                # ax.yaxis.set_major_formatter(NullFormatter())
                ax.axis('tight')
                ax.set_ylim([0, 8])
                # ax.set_ylim([-60, 60])
                # if dataset_name == 'newsqa':
                #     ax.set_ylim([0, 80])
                # else:
                #     ax.set_ylim([0, 90])
                ax.legend(loc='upper right')

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.show()
    output_dir = root + 'question-answering/plots/performance_change_over_' + mode +'_variation_ratio.pdf'
    print(output_dir)

    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_dir, bbox_inches='tight',
                pad_inches=0)

    # plt.savefig(output_dir)


def visualize_performance_by_contrastive_lexical_overlap():
    fig = plt.figure(figsize=(7.5, 8))
    fig.suptitle("Performance Change Over Different Degree of Lexical Overlap with Increased Data Size", fontsize=14)

    root = ""

    dataset_name = ['squad', 'newsqa']
    metrics = ['em', 'f1']


    all_keys = ['Overall', 'Less Overlap', 'More Overlap']
    all_main_keys = ['less_overlap', 'more_overlap']

    i = 0
    for ds in dataset_name:
        eval_results_dir = root + 'question-answering/eval_results/' \
                                  'qtype_'+ ds + '_lexical_overlap_eval_results.json'

        eval_results = json_load(eval_results_dir)

        for metric in metrics:
            ax = fig.add_subplot(2, 2, i + 1)
            i += 1
            perf_by_key_all = {}
            for query_key in all_main_keys:
                x = list(eval_results[query_key].keys())
                perf_by_key = {}

                for k in all_keys:
                    perf_by_key[k] = []

                for data_size in eval_results[query_key]:
                    for sub_key in all_keys:
                        perf_by_key[sub_key].append(eval_results[query_key][data_size][sub_key][metric])
                perf_by_key_all[query_key] = perf_by_key

            perf_by_key_constrast = {}

            for sub_key in all_keys:
                perf_by_key_constrast[sub_key] = [e1-e2 for e1,e2 in
                                                  zip(perf_by_key_all[all_main_keys[0]][sub_key],
                                                      perf_by_key_all[all_main_keys[1]][sub_key])]

            # colors = ['b', 'g', 'r', 'c', 'm', 'y']
            colors = ['b', 'g', 'r']
            for sub_key, color in zip(perf_by_key_constrast.keys(), colors):
                ax.plot(x, perf_by_key_constrast[sub_key], c=color, label=sub_key)

            ax.set_title(metric + ' ' + ds)
            ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
            # ax.set_ylim([0, 8])
            # if dataset_name == 'newsqa':
            #     ax.set_ylim([0, 80])
            # else:
            #     ax.set_ylim([0, 90])
            ax.legend(loc='lower right')

    plt.show()
    # output_dir = root + ''
    # plt.savefig(output_dir)
    

if __name__ == '__main__':
    # visualize_performance_by_contrastive_lexical_overlap()
    # visualize_performance_by_contrastive('ans_pos')
    visualize_performance_by_contrastive('text_len')
    # data_sets = ['newsqa', 'squad_1.1']
    # keys = ['context', 'question', 'answer']
    # for ds in data_sets:
    #     for key in keys:
    #         visualize_performance_by_qas_length(ds, key)

    # visualize_performance_by_qas_length()
        # visualize_performance_by_lexical_overlap_merge_dataset()
    # visualize_perofrmance_by_qtype_merge_dataset()
    # visualize_performance_by_lexical_overlap_merge_view()
    # visualize_perofrmance_by_qtype_merge_view()
    # visualize_performance_by_lexical_overlap()
    # visualize_performance_by_qtype()
