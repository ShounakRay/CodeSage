import numpy as np
from sklearn.model_selection import GridSearchCV
from datasets import load_dataset
from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters
from Modules.Code2Code.models.t5_code_2_code_model import T5Code2CodeModel

# ########################################################################
# ######################### HYPERPARAMETERS ##############################
# ########################################################################

# MAX_FUNCTION_STRING_LENGTH = 512
# N_SNIPPETS = 10000

# C2D_LLM = 'CODETRANS'
# assert C2D_LLM in  ('CODETRANS', 'CODEX', 'GPT')

# # IC_ALGO = 'KMEANS'
# # assert IC_ALGO in ("KMEANS/SOM, KMEANS, DBSCSAN")
# # IC_KVAL = 10
# # assert type(IC_KVAL) == np.number

# SC_LOWPERC = 0.1
# assert 0.01 <= SC_LOWPERC <= 0.99
# SC_HIGHPERC = 0.9
# assert 0.01 <= SC_HIGHPERC <= 0.99
# SC_BOUNDARY = 50
# print(f"CUSTOM NOTE: Ensure that `SC_BOUNDARY` is set to - say - \
#       the overall median of scores across all clusters.\n\n")
# SC_METHOD = 'PERCENTILE'
# assert SC_METHOD in ('PERCENTILE', 'SHARED')

# C2C_LLM = 'CODE-T5'
# assert C2C_LLM in ('CODE-T5', 'SOMETHING_ELSE')

# C2C_EVAL_METRIC = "bleu"
# assert C2C_EVAL_METRIC in ('bleu', 'chrf')

# C2C_TEST_SIZE = 0.2
# assert 0.01 <= C2C_TEST_SIZE <= 0.99

# C2C_LR = 0.3
# assert 0.05 <= C2C_LR <= 0.95
# C2C_EPOCH_N = 2
# assert 1 <= C2C_EPOCH_N <= 5
# C2C_BATCH_SIZE = 16
# assert 1 <= C2C_BATCH_SIZE <= 64
# C2C_WEIGHT_DECAY = 0.01
# assert 0.001 <= C2C_WEIGHT_DECAY <= 0.1

# ########################################################################
# ############################ PIPELINE ##################################
# ########################################################################

# # Get Dataset
# dataset = CodeSnippetDataset(github=False, languages=["Python"])
# code_snippets = dataset.get_n_snippets(N_SNIPPETS, max_length=MAX_FUNCTION_STRING_LENGTH)
# """
# dict_keys(['function', 'repo_name', 'path', 'features', 'purpose', 'detailed_description', 'code_trans', 'id'])
# """
# print("Got snippets!\n")


# # Transforms dataset into code format
# code2doc = Code2DocModule()
# data_with_docs = code2doc.get_docs(code_snippets, C2D_LLM = C2D_LLM)
# """
# data_with_docs = {
#  	function_ids: [“id1”],
#  	code_reference: {
#  		“id1”: {
#  			“code”: …,
#  			“reputation”: […,…,,..,..],
#  			“documentation”: depends on C2D_LLM(string)

#  		}
#  	}
#  }
# """

# print("Got documentations!\n")

# # Turn dataset into clusters
# doc2clusters = IntentClustering(function_ids=data_with_docs['function_ids'], code_reference=data_with_docs['code_reference'])
# clusters = doc2clusters.core_get_clusters(embedder="strans", method='dbscan', n_clusters=IC_KVAL, eps=0.5, min_samples=5, n_jobs=-1)

# print("Got clusters!\n")
# """
# Cluster output is:
# { cluster_id (int) : [function_id, function_id, function_id (Any)] }
# """

# # Score clusters
# clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'],
#                                        SC_METHOD=SC_METHOD,
#                                        SC_LOWPERC=SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
#                                        SC_HIGHPERC=SC_HIGHPERC if SC_METHOD == 'PERCENTILE' else None,
#                                        SC_BOUNDARY=SC_BOUNDARY if SC_METHOD == 'SHARED' else None)
# scored_dataset = clusters2scoredDataset.get_scored_dataset()
# print("Scored clusters!\n")

# MODEL_OUTPUT_DIR = "c2c_model_with_chrf_and_nonzero_reps"
# # Train with Seq2Seq model
# model = T5Code2CodeModel("base", C2C_EVAL_METRIC=C2C_EVAL_METRIC)
# model.train(scored_dataset, 
#             MODEL_OUTPUT_DIR, 
#             C2C_TEST_SIZE=C2C_TEST_SIZE, 
#             C2C_LR=C2C_LR, 
#             C2C_BATCH_SIZE=C2C_BATCH_SIZE, 
#             C2C_WEIGHT_DECAY=C2C_WEIGHT_DECAY, 
#             C2C_EPOCH_N=C2C_EPOCH_N
#             )
# print("Trained model!")

# # Perform inference
# example_bad_function = "def hello_world(): pfds('hello_world')"
# resulting_good_function = model(example_bad_function)["translation_text"]
# print(resulting_good_function)

def run_end_to_end_with_parameters(
      FUNCTIONS_DATASET_URI,
      MAX_FUNCTION_STRING_LENGTH,
      C2D_LLM,
      IC_METHOD,
      IC_EMBEDDER,
      IC_KVAL,
      SC_LOWPERC,
      SC_HIGHPERC,
      SC_BOUNDARY,
      SC_METHOD,
      C2C_LLM,
      C2C_EVAL_METRIC,
      C2C_TEST_SIZE,
      C2C_LR,
      C2C_EPOCH_N,
      C2C_BATCH_SIZE,
      C2C_WEIGHT_DECAY,
):
      assert C2D_LLM in  ('C`ODETRANS', 'CODEX', 'GPT')
      assert 1 <= SC_LOWPERC <= 99
      assert 1 <= SC_HIGHPERC <= 99
      assert 0 <= SC_BOUNDARY <= 100
      assert IC_METHOD in ("kmeans", "dbscan")
      assert IC_EMBEDDER in ("tfidf", "STrans", "Elmo")
      assert 1 <= IC_KVAL <= 20
      assert SC_METHOD in ('PERCENTILE', 'SHARED')
      assert C2C_LLM in ('CODE-T5', 'SOMETHING_ELSE')
      assert C2C_EVAL_METRIC in ('bleu', 'chrf')
      assert 0.01 <= C2C_TEST_SIZE <= 0.99
      assert 2e-5 <= C2C_LR <= 0.01
      assert 1 <= C2C_EPOCH_N <= 5
      assert 1 <= C2C_BATCH_SIZE <= 64
      assert 0.01 <= C2C_WEIGHT_DECAY <= 0.1 
      dataset = load_dataset(FUNCTIONS_DATASET_URI, split="train")
      code_snippets = dataset.filter(lambda example: len(example["function"].split()) <= MAX_FUNCTION_STRING_LENGTH)[:]
      code2doc = Code2DocModule()
      data_with_docs = code2doc.get_docs(code_snippets, C2D_LLM = C2D_LLM) 
      doc2clusters = IntentClustering(function_ids=data_with_docs['function_ids'], code_reference=data_with_docs['code_reference'])
      clusters = doc2clusters.core_get_clusters(embedder=IC_EMBEDDER, method=IC_METHOD, n_clusters=IC_KVAL, eps=0.5, min_samples=5, n_jobs=-1)
      clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'],
                                       SC_METHOD=SC_METHOD,
                                       SC_LOWPERC=SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_HIGHPERC=SC_HIGHPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_BOUNDARY=SC_BOUNDARY if SC_METHOD == 'SHARED' else None)
      
      scored_dataset = clusters2scoredDataset.get_scored_dataset()
      model = T5Code2CodeModel("base", C2C_EVAL_METRIC=C2C_EVAL_METRIC)
      MODEL_OUTPUT_DIR = "C2C_Model_03_14_2023"
      model.train(scored_dataset, 
            MODEL_OUTPUT_DIR, 
            C2C_TEST_SIZE=C2C_TEST_SIZE, 
            C2C_LR=C2C_LR, 
            C2C_BATCH_SIZE=C2C_BATCH_SIZE, 
            C2C_WEIGHT_DECAY=C2C_WEIGHT_DECAY, 
            C2C_EPOCH_N=C2C_EPOCH_N
      ) 
       
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV

# # Load the iris dataset
# iris = load_iris()

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# # Define the hyperparameters to tune
# param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

# # Create the SVM classifier object
# svm_clf = SVC()

# # Create the GridSearchCV object with 5-fold cross-validation
# grid_search = GridSearchCV(svm_clf, param_grid, cv=5)

# # Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters and the corresponding accuracy score
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best accuracy score: ", grid_search.best_score_)


def simulate():
      C2D_LLMS = ['CODETRANS', 'CODEX', 'GPT']
      SC_LOWPERCS = np.linspace(10, 49, 6)
      SC_HIGHPERCS = 1 - SC_LOWPERCS
      SC_BOUNDARIES = np.linspace(1, 100, 5)
      IC_METHODS = ["kmeans", "dbscan"]
      IC_EMBEDDERS = ["tfidf", "STrans", "Elmo"]
      SC_METHODS = ["PERCENTILE", "SHARED"]
      IC_KVALS = np.linspace(2, 20, 5)
      C2C_LLMS = ['CODE-T5']
      C2C_EVAL_METRICS = ["bleu", "chrf"]
      C2C_TEST_SIZE = np.linspace(0.1, 0.3, 5)
      C2C_BATCH_SIZES = [8] 
      C2C_WEIGHT_DECAYS = np.linspace(0.01, 0.05, 5)
      C2C_EPOCH_NS = np.arange(2, 6, 1)

      
      run_end_to_end_with_parameters(
            FUNCTIONS_DATASET_URI="michaelnath/annotated_github_dataset_2",
            MAX_FUNCTION_STRING_LENGTH=512,
            C2D_LLM = "GPT",
            IC_METHOD="kmeans",
            IC_EMBEDDER="STrans",
            IC_KVAL = 10,
            SC_LOWPERC = 20,
            SC_HIGHPERC = 80,
            SC_BOUNDARY = 60,
            SC_METHOD = "SHARED",
            C2C_LLM = "CODE-T5",
            C2C_EVAL_METRIC="chrf",
            C2C_TEST_SIZE=0.2,
            C2C_LR=2e-5,
            C2C_EPOCH_N=3,
            C2C_BATCH_SIZE=4,
            C2C_WEIGHT_DECAY=0.01, 
      )
      

if __name__ == "__main__":
      simulate()