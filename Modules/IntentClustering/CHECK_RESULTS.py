import json
from os import walk
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns


def ingest():
    PATH = "results/"
    f = []
    for (dirpath, dirnames, filenames) in walk(PATH):
        f.extend(filenames)
    json_dicts = {}
    for file_name in f:
        with open(PATH + file_name, "r") as fp:
            json_dicts[file_name] = json.load(fp)
    # pprint(json_dicts)
    final = {}
    list_dicts = []
    for HYPER_CONFIG, RESULTS_ONLY in json_dicts.items():
        for metric, data in RESULTS_ONLY.items():
            if metric == 'eval_codebleu':
                for sub_metric, sub_results in data.items():
                    final_metric_name = metric + '--' + sub_metric
                    final[final_metric_name] = sub_results
            else:
                final[metric] = data
        final['hyperparams'] = HYPER_CONFIG
        final = {}
        list_dicts.append(final)
    # pprint(list_dicts)
    df = pd.DataFrame(list_dicts).infer_objects().reset_index(drop=False)
    new_hyps_df = df['hyperparams'].str.split(pat='_', expand=True).infer_objects()
    STRUCTURE = ["NFUNCS",
                "C2D_LLMS",
                "_SC_THRESHOLD_VALUE",
                "IC_METHODS",
                "IC_EMBEDDERS",
                "SC_SCORING",
                "SC_SPLIT_METHOD",
                "IC_KVALS",
                "C2C_LLMS",
                "C2C_TEST_SIZE",
                "C2C_BATCH_SIZES",
                "C2C_WEIGHT_DECAYS",
                "C2C_EPOCH_NS",
                "C2C_LR"]
    new_hyps_df.columns = STRUCTURE
    new_hyps_df['C2C_LR'] = new_hyps_df['C2C_LR'].str.split('.json').str.join('').astype(float)
    for col in list(new_hyps_df):
        try:
            new_hyps_df[col] = new_hyps_df[col].astype(float)
        except Exception as e:
            pass
    new_hyps_df.to_csv('HYPS_COMBOS.csv')
    
    # data = {}
    # for col in list(new_hyps_df):
    #     data[col] = dict(new_hyps_df[col].value_counts())
    # pprint(data)

    FINAL_DF = pd.concat([df, new_hyps_df], axis=1).dropna()
    FINAL_DF.drop('hyperparams', axis=1, inplace=True)
    return FINAL_DF

def make_graph(grouping='C2D_LLMS', filters=['CODETRANS', 'GPT'], score_col='eval_bleu', x_col='IC_KVALS',
               x_label='', y_label='', group_label=''):
    plt.figure(figsize=(4, 6))
    TITLE = f'{y_label} VS. {x_label}\nacross {group_label} types'
    plt.title(TITLE)
    _temp = FINAL_DF[FINAL_DF[grouping] == filters[0]].reset_index(drop=True).copy()
    plt.scatter(_temp[x_col], _temp[score_col], color='red', label=filters[0])
    _temp = FINAL_DF[FINAL_DF[grouping] == filters[1]].reset_index(drop=True).copy()
    plt.scatter(_temp[x_col], _temp[score_col], color='blue', label=filters[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.savefig(TITLE.replace('\n', '__').replace(' ', '_') + '.png')
    plt.show()

"""
################################################
################################################
################################################
################################################
"""

FINAL_DF = ingest()
FINAL_DF.dtypes

pprint({col: dict(FINAL_DF[col].value_counts()) for col in list(FINAL_DF) if FINAL_DF[col].dtype == object })

##############################
##############################
# ACROSS ALL C2D LLM MODELS

make_graph(grouping='C2D_LLMS',
           filters=['CODETRANS', 'GPT'],
           score_col='eval_bleu',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='BLEU Score',
           group_label='Code-to-Doc LLM')

make_graph(grouping='C2D_LLMS',
           filters=['CODETRANS', 'GPT'],
           score_col='eval_codebleu--CodeBLEU',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CodeBLEU',
           group_label='Code-to-Doc LLM')

make_graph(grouping='C2D_LLMS',
           filters=['CODETRANS', 'GPT'],
           score_col='eval_chrf',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CHRF Score',
           group_label='Code-to-Doc LLM')

##############################
##############################
# ACROSS ALL SPLIT METHODS

# FINAL_DF['SC_SPLIT_METHOD'].unique()
make_graph(grouping='SC_SPLIT_METHOD',
           filters=['PERCENTILE', 'SHARED'],
           score_col='eval_chrf',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CHRF Score',
           group_label='Scoring Method')

make_graph(grouping='SC_SPLIT_METHOD',
           filters=['PERCENTILE', 'SHARED'],
           score_col='eval_codebleu--CodeBLEU',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CodeBLEU Score',
           group_label='Scoring Method')

make_graph(grouping='SC_SPLIT_METHOD',
           filters=['PERCENTILE', 'SHARED'],
           score_col='eval_bleu',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='BLEU Score',
           group_label='Scoring Method')

##############################
##############################
# ACROSS ALL CLUSTERING METHODS

make_graph(grouping='IC_METHODS',
           filters=['dbscan', 'kmeans'],
           score_col='eval_chrf',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CHRF Score',
           group_label='Clustering Method')

make_graph(grouping='IC_METHODS',
           filters=['dbscan', 'kmeans'],
           score_col='eval_codebleu--CodeBLEU',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='CodeBLEU Score',
           group_label='Clustering Method')

make_graph(grouping='IC_METHODS',
           filters=['dbscan', 'kmeans'],
           score_col='eval_bleu',
           x_col='IC_KVALS',
           x_label='K-Value',
           y_label='BLEU Score',
           group_label='Clustering Method')

######

make_graph(grouping='IC_METHODS',
           filters=['dbscan', 'kmeans'],
           score_col='eval_bleu',
           x_col='C2C_LR',
           x_label='Learning Rate',
           y_label='BLEU Score',
           group_label='Clustering Method')

# EVAL_LIST = [e for e in list(FINAL_DF) if e not in STRUCTURE]

# # # Pairplot
# # fig = plt.figure(figsize=(30, 30))
# # sns.pairplot(FINAL_DF)
# # plt.tight_layout()
# # plt.savefig('testing_this.png', bbox_inches='tight')

# # plt.plot(FINAL_DF[''])
# # plt.close()

# fig = plt.figure(figsize=(30, 50)) 
# gs = gridspec.GridSpec(len(STRUCTURE), len(EVAL_LIST))

# # FINAL_DF.dtypes
# # pprint(list(FINAL_DF))
# # FINAL_DF.apply(pd.to_numeric)
# # pd.is_num(FINAL_DF)

# # This is the row number
# for hyperparam in STRUCTURE:
#     # This is the column number
#     for metric in EVAL_LIST:
#         x, y = STRUCTURE.index(hyperparam), EVAL_LIST.index(metric)
#         ax = plt.subplot(gs[x, y])

#         # pprint(FINAL_DF[hyperparam])
#         # pprint(FINAL_DF[metric])

#         _ = ax.scatter(FINAL_DF[hyperparam], FINAL_DF[metric])
#         _ = ax.set_title(f"{metric} VS. {hyperparam}")
#         _ = ax.set_xlabel(hyperparam)
#         _ = ax.set_ylabel(metric)
#         _ = fig.tight_layout()
#         _ = fig.add_subplot(ax)
# fig.tight_layout()
# plt.show()
# plt.savefig('MAJOR_RESULTS.png', bbox_inches=True)
# # plt.close()

# # fig, axis = plt.subplots(2,3, figsize=(8, 8))
# # FINAL_DF.hist(ax=axis)
# # plt.show()

# # EOF