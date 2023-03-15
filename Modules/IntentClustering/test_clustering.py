from git import Repo
import os
import sys
from data2clusters import IntentClustering
from datasets import load_dataset
from collections import Counter
from pprint import pprint
from sklearn.metrics import silhouette_score
import json
import numpy as np
import pandas as pd
import _pickle as cPickle

# with open('Modules/IntentClustering/storeData/v_data.pickle', 'rb') as fp:
#     output = cPickle.load(fp)

# if not os.path.isdir('Modules/IntentClustering/Reference'):
#     os.makedirs('Modules/IntentClustering/Reference')
#     Repo.clone_from("https://github.com/TheAlgorithms/Python.git", 'Modules/IntentClustering/Reference')

"""
1. [Optionally] Remove docstrings from each python file (so the documentation isn't that easy to generate)
2. [Normal]     Assign a function id to each function (same format as `code_snippets`)
3. [Normal]     Get documentation for each function id (through processing in the glue) via `Code2DocModule`
4. [Special]    Make a dictionary mapping {true_folder/cluster : List[function_ids]} and assign each "true_folder/cluster" a color
5. [Normal]     Retrieve a dictionary mapping {{detected_cluster : List[function_ids]}}
6. [Special]    Visualize/Create {{detected_cluster : List[function_ids]}} clusters but breakdown content on true-clusters/colors in
                    each detected cluster
7. [Interpret]  The more fragmentation in each "detected_cluster" aka distribution of each "true_folder/cluster", the worse
                    the algorithm each. Fragmentation can be measured as the weighted # of "true_folder/clusters" in
                    each "detected_cluster"
8. [Interpret]  Just qualitatively look at each of the clusters and see if it makes sense.
9. [Compare]    Qualitatively see if the KMeans clustering on the "Reference" dataset splits into similar categories as the
                    "True Dataset"
"""
if __name__ == "__main__":
    # mapping = {intent_category : detailed_description}
    RAW_OUTPUT = load_dataset("michaelnath/annotated_github_dataset_2")
    RAW_OUTPUT = RAW_OUTPUT['train']
    RAW_OUTPUT = RAW_OUTPUT.to_pandas().reset_index(drop=False)
    # NUM_TRUE_CLUSTERS = RAW_OUTPUT['repo_name'].nunique()
    NUM_TRUE_CLUSTERS = 50

    DOC_MODES = [['purpose', 'code_trans', 'detailed_description'][2]]
    REF_id_to_cats = dict(zip(RAW_OUTPUT['index'], RAW_OUTPUT['repo_name']))

    with open(f"Modules/IntentClustering/storeData/REF_id_to_cats.json", 'w') as fp:
            json.dump(REF_id_to_cats, fp, indent=4)

    # silhouette_scores = {}
    SILHOUETTE_SCORES = []
    for DOC_M in DOC_MODES:
        REF_id_to_docs = dict(zip(RAW_OUTPUT['index'], RAW_OUTPUT[DOC_M]))

        with open(f"Modules/IntentClustering/storeData/REF_id_to_docs__{DOC_M}.json", 'w') as fp:
            json.dump(REF_id_to_docs, fp, indent=4)

        for code_id, documentation in REF_id_to_docs.items():
            REF_id_to_docs[code_id] = {"documentation": documentation}

        # silhouette_scores[DOC_M] = {}
        for K_VALUE in np.arange(30, 500, 30):
            K_VALUE = int(K_VALUE)
            print(f"DOC_MODE: {DOC_M}, Clustering with k-value {K_VALUE}")
            doc2clusters = IntentClustering(function_ids=REF_id_to_docs.keys(),
                                            code_reference=REF_id_to_docs)
            clusters = doc2clusters.core_get_clusters(embedder='strans', method='kmeans',
                                                    n_clusters=K_VALUE, eps=0.5,
                                                    min_samples=5, n_jobs=-1,
                                                    doc_source='Detailed')
            sil_score = silhouette_score(doc2clusters.v_data, doc2clusters._labels, metric = 'euclidean')

            # LIMIATION: only working on good data
            # TODO: Test on different types of documentation
            # reference = {}
            for cluster_id, list_doc_ids in clusters.items():
                true_folder_freqs = dict(Counter([REF_id_to_cats[DET_id] for DET_id in list_doc_ids]))
                # reference[cluster_id] = true_folder_freqs
                SILHOUETTE_SCORES.append((DOC_M, K_VALUE, sil_score, cluster_id, list_doc_ids, true_folder_freqs))

            # silhouette_scores[K_VALUE] = {'sil_score': sil_score, 'c_dists': reference}

    # with open("reference_GPT_performance.json", 'w') as fp:
    #     json.dumps(silhouette_scores, fp)

    df = pd.DataFrame(SILHOUETTE_SCORES)
    df.columns = ['C2D_Method', 'k_value', "silhouette", "algo_cluster_id",
                "list_ids_cluster_id", "true_folder_freqs"]
    df.to_pickle(f"Modules/IntentClustering/storeData/reference_GPT_performance__{DOC_MODES[0]}.pickle")

# EOF