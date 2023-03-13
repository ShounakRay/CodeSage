from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters
from Modules.Code2Code.models.t5_code_2_code_model import T5Code2CodeModel
import numpy as np

########################################################################
######################### HYPERPARAMETERS ##############################
########################################################################

N_SNIPPETS = 100;

C2D_VERBOSITY = 1
assert C2D_VERBOSITY in (1, 2, 3, 4, 5)
C2D_LLM = 'CODETRANS'
assert C2D_VERBOSITY in  ('CODETRANS')

IC_ALGO = 'KMEANS'
assert IC_ALGO in ("KMEANS/SOM, KMEANS, DBSCSAN")
IC_KVAL = 10
assert type(IC_ALGO) == np.number

SC_LOWPERC = 0.1
assert 0.01 <= SC_LOWPERC <= 0.99
SC_HIGHPERC = 0.9
assert 0.01 <= SC_HIGHPERC <= 0.99
SC_BOUNDARY = 50
print(f"CUSTOM NOTE: Ensure that `SC_BOUNDARY` is set to - say - \
      the overall median of scores across all clusters.\n\n")
SC_METHOD = 'PERCENTILE'
assert SC_METHOD in ('PERCENTILE', 'SHARED')

C2C_LLM = 'CODE-T5'
assert C2C_LLM in ('CODE-T5', 'SOMETHING_ELSE')
C2C_LR = 0.3
assert 0.05 <= C2C_LR <= 0.95
C2C_EPOCH_N = 2
assert 1 <= C2C_EPOCH_N <= 5
C2C_BATCH_SIZE = 16
assert 1 <= C2C_BATCH_SIZE <= 64
C2C_WEIGHT_DECAY = 0.01
assert 0.001 <= C2C_WEIGHT_DECAY <= 0.1

########################################################################
############################ PIPELINE ##################################
########################################################################

# Get Dataset
# TODO: Why is `CodeSnippetDataset` pulled from Extracontent? Make sure this isn't the case.
code_snippets = CodeSnippetDataset(languages=["Python"],
                                all_features_above=0,
                                n_samples=N_SNIPPETS)
print("Got snippets!\n")

# Get documentation from dataset
code2doc = Code2DocModule(code_snippets, C2D_LLM = C2D_LLM)
data_with_docs = code2doc.get_docs()
    
print("Got documentations!\n")

# Turn dataset into clusters
doc2clusters = IntentClustering(function_ids=data_with_docs['function_ids'], code_reference=data_with_docs['code_reference'])
clusters = doc2clusters.core_get_clusters(embedder="STrans", method='kmeans', n_clusters=IC_KVAL, eps=0.5, min_samples=5, n_jobs=-1)
print("Got clusters!\n")
"""
Cluster output is:
{ cluster_id (int) : [function_id, function_id, function_id (Any)] }
"""

# Score clusters
clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'],
                                       SC_METHOD=SC_METHOD,
                                       SC_LOWPERC=SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_HIGHPERC=SC_HIGHPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_BOUNDARY=SC_BOUNDARY if SC_METHOD == 'SHARED' else None)
scored_dataset = clusters2scoredDataset.get_scored_dataset()
print("Scored clusters!\n")

"""
TODO: Add these hyperparameters to a modularized equivalent.
C2C_LLM = 'CODE-T5'
C2C_LR = 0.3
C2C_EPOCH_N = 2
C2C_BATCH_SIZE = 16
C2C_WEIGHT_DECAY = 0.01
"""
# Train with Seq2Seq model
model = T5Code2CodeModel("base")
model.train(scored_dataset, "glued_code_to_code_model")
print("Trained model!\n")

# Perform inference
example_bad_function = "def hello_world(): pfds('hello_world')"
resulting_good_function = model(example_bad_function)["translation_text"]
print(resulting_good_function)

# TODO: Find a way to return the BLEU score for analyzing pipeline performance