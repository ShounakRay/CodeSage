import pandas as pd
import json
import matplotlib.pyplot as plt

df_ct = pd.read_pickle("Modules/IntentClustering/storeData/reference_GPT_performance__code_trans.pickle")
df_pu = pd.read_pickle("Modules/IntentClustering/storeData/reference_GPT_performance__purpose.pickle")
df_dt = pd.read_pickle("Modules/IntentClustering/storeData/reference_GPT_performance__detailed_description.pickle")

with open("Modules/IntentClustering/storeData/REF_id_to_cats.json", 'r') as fp:
    REF_id_to_cats = json.load(fp)

with open("Modules/IntentClustering/storeData/REF_id_to_docs__detailed_description.json", 'rb') as fp:
    REF_id_to_docs = json.load(fp)

df = df_ct.copy().append(df_pu, ignore_index=True).append(df_dt, ignore_index=True).reset_index(drop=True)

df[df['C2D_Method'] == 'purpose']['silhouette'].describe()
df[df['C2D_Method'] == 'code_trans']['silhouette'].describe()

df['silhouette'].hist()
plt.show()

plt.scatter(df['k_value'], df['silhouette'])
plt.show()

df.set_index('k_value', inplace=False).groupby('C2D_Method')['silhouette'].plot(legend=True)
plt.show()

# EOF