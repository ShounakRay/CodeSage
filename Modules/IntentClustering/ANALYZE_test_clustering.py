import pandas as pd
import json
import matplotlib.pyplot as plt

def make_graph(FINAL_DF, grouping='C2D_LLMS', filters=['CODETRANS', 'GPT'], score_col='eval_bleu', x_col='IC_KVALS',
               x_label='', y_label='', group_label=''):
    plt.figure(figsize=(5, 4))
    TITLE = f'{y_label} VS. {x_label}\nacross {group_label} types'
    plt.title(TITLE)
    colors = ['red', 'blue', 'green', 'purple']
    mapping = {'code_trans': 'CodeTrans',
               'detailed_description': 'GPT'}
    for fil in filters:
        _temp = FINAL_DF[FINAL_DF[grouping] == fil].reset_index(drop=True).copy()
        plt.scatter(_temp[x_col], _temp[score_col], color=colors[filters.index(fil)], label=mapping[fil])
    # _temp = FINAL_DF[FINAL_DF[grouping] == filters[1]].reset_index(drop=True).copy()
    # plt.scatter(_temp[x_col], _temp[score_col], color='blue', label=filters[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.savefig('SIL_' + TITLE.replace('\n', '__').replace(' ', '_') + '.png', bbox_inches='tight')
    plt.show()

df_ct = pd.read_pickle("Modules/IntentClustering/storeData/EPS_reference_GPT_performance__code_trans.pickle")
# df_pu = pd.read_pickle("Modules/IntentClustering/storeData/reference_GPT_performance__purpose.pickle")
df_dt = pd.read_pickle("Modules/IntentClustering/storeData/EPS_reference_GPT_performance__detailed_description.pickle")

# with open("Modules/IntentClustering/storeData/REF_id_to_cats.json", 'r') as fp:
#     REF_id_to_cats = json.load(fp)
# with open("Modules/IntentClustering/storeData/REF_id_to_docs__detailed_description.json", 'rb') as fp:
#     REF_id_to_docs = json.load(fp)

df = df_ct.copy().append(df_dt, ignore_index=True).reset_index(drop=True)

# df['silhouette'].hist()
# plt.show()

make_graph(FINAL_DF=df,
           grouping='C2D_Method',
           filters=list(df['C2D_Method'].unique()),
           score_col='silhouette',
           x_col='k_value',
           x_label='K-Value',
           y_label='Silhouette Score',
           group_label='Code-to-Doc Method')

make_graph(FINAL_DF=df,
           grouping='C2D_Method',
           filters=list(df['C2D_Method'].unique()),
           score_col='silhouette',
           x_col='eps_value',
           x_label='Epsilon Value',
           y_label='Silhouette Score',
           group_label='Code-to-Doc Method')

df.set_index('k_value', inplace=False).groupby('C2D_Method')['silhouette'].plot(legend=True)
plt.show()

# EOF