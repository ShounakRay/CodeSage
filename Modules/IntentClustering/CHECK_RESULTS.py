import json
from os import walk
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns


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

pprint(list_dicts)

df = pd.DataFrame(list_dicts).infer_objects().reset_index(drop=False)
# pprint(list(df))
# pprint(final)
# final.keys()
# df['hyps'] = df['index'].apply(lambda hyp: hyp.split('_'))
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
new_hyps_df['C2C_LR'] = float(''.join(new_hyps_df['C2C_LR'].str.split('.json')[0]))

for col in list(new_hyps_df):
    try:
        new_hyps_df[col] = new_hyps_df[col].astype(float)
    except Exception as e:
        pass

# pprint(new_hyps_df.dtypes)

"""
################################################
################################################
################################################
################################################
"""

# STRUCTURE is all hyperparams
# EVAL_LIST is all scores

# new_hyps_df['IC_KVALS']

FINAL_DF = pd.concat([df, new_hyps_df], axis=1).dropna()
# FINAL_DF.drop('eval_codebleu', axis=1, inplace=True)

pprint(list(FINAL_DF))

FINAL_DF.dtypes

_TEMPP = FINAL_DF.corr()
plt.figure(figsize=(20, 20))
_ = sns.heatmap(_TEMPP)
plt.tight_layout()
plt.show()

""" SEPSEPSEPSEPSEPSEPSEPSEP """

# Label with boundaries and percentiles
plt.figure(figsize=(4, 6))
plt.title('BLEU Score VS. k-value')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'PERCENTILE'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_bleu'], color='red', label='PERCENTILE')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'SHARED'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_bleu'], color='blue', label='SHARED')
plt.xlabel('k-value for KMeans')
plt.xlabel('BLEU Score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()

""" SEPSEPSEPSEPSEPSEPSEPSEP """



""" SEPSEPSEPSEPSEPSEPSEPSEP """

plt.figure(figsize=(4, 6))
plt.title('CodeBLEU Score VS. k-value')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'PERCENTILE'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_codebleu--CodeBLEU'], color='red', label='PERCENTILE')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'SHARED'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_codebleu--CodeBLEU'], color='blue', label='SHARED')
plt.xlabel('k-value for KMeans')
plt.xlabel('CodeBLEU Score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()

""" SEPSEPSEPSEPSEPSEPSEPSEP """

plt.figure(figsize=(4, 6))
plt.title('CHRF Score VS. k-value')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'PERCENTILE'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_chrf'], color='red', label='PERCENTILE')
_temp = FINAL_DF[FINAL_DF['SC_SPLIT_METHOD'] == 'SHARED'].reset_index(drop=True).copy()
plt.scatter(_temp['IC_KVALS'], _temp['eval_chrf'], color='blue', label='SHARED')
plt.xlabel('k-value for KMeans')
plt.xlabel('CodeBLEU Score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()


EVAL_LIST = [e for e in list(FINAL_DF) if e not in STRUCTURE]

# # Pairplot
# fig = plt.figure(figsize=(30, 30))
# sns.pairplot(FINAL_DF)
# plt.tight_layout()
# plt.savefig('testing_this.png', bbox_inches='tight')

# plt.plot(FINAL_DF[''])
# plt.close()

fig = plt.figure(figsize=(30, 50)) 
gs = gridspec.GridSpec(len(STRUCTURE), len(EVAL_LIST))

# FINAL_DF.dtypes
# pprint(list(FINAL_DF))
# FINAL_DF.apply(pd.to_numeric)
# pd.is_num(FINAL_DF)

# This is the row number
for hyperparam in STRUCTURE:
    # This is the column number
    for metric in EVAL_LIST:
        x, y = STRUCTURE.index(hyperparam), EVAL_LIST.index(metric)
        ax = plt.subplot(gs[x, y])

        # pprint(FINAL_DF[hyperparam])
        # pprint(FINAL_DF[metric])

        _ = ax.scatter(FINAL_DF[hyperparam], FINAL_DF[metric])
        _ = ax.set_title(f"{metric} VS. {hyperparam}")
        _ = ax.set_xlabel(hyperparam)
        _ = ax.set_ylabel(metric)
        _ = fig.tight_layout()
        _ = fig.add_subplot(ax)
fig.tight_layout()
plt.show()
plt.savefig('MAJOR_RESULTS.png', bbox_inches=True)
# plt.close()

# fig, axis = plt.subplots(2,3, figsize=(8, 8))
# FINAL_DF.hist(ax=axis)
# plt.show()

# EOF