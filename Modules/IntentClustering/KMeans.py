from sklearn.cluster import KMeans
import numpy as np
from vectorizer import _get_data, vectorize
import json

with open("_tempData/functions.json", "r") as f:
    doc_to_id = {}
    for metaData in list(json.load(f)["code_reference"].items()):
        code_id, info = metaData
        doc_to_id[info["documentation"]] = code_id

data = _get_data(source = "json")
v_data = vectorize(data=data, mode="pretrained")
kmeans = KMeans(n_clusters=5, random_state=41, n_init="auto").fit(v_data)

new_dict = {}
for i in range(len(kmeans.labels_)):
    cluster_id = int(kmeans.labels_[i])
    data_app = doc_to_id[data[i]]
    if cluster_id not in new_dict.keys():
        new_dict[cluster_id] = [data_app]
    else:
        new_dict[cluster_id].append(data_app)

with open('_tempData/kmeans_result.json', 'w') as fp:
    json.dump(new_dict, fp)