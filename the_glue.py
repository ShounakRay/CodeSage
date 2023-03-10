from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters
from Modules.Code2Code.models.t5_code_2_code_model import T5Code2CodeModel

MAX_FUNCTION_STRING_LENGTH = 512
N_SNIPPETS = 100
DOCUMENTATION_INFERENCE_BATCH_SIZE = 4

# Get Dataset
dataset = CodeSnippetDataset(github=False, languages=["Python"])

code_snippets = dataset.get_n_snippets(N_SNIPPETS, max_length=MAX_FUNCTION_STRING_LENGTH)

print("Got snippets!")


# Get documentation from dataset
code2doc = Code2DocModule()
data_with_docs = code2doc.get_docs(code_snippets, DOCUMENTATION_INFERENCE_BATCH_SIZE)
    
print("Got documentations!")

# Turn dataset into clusters
doc2clusters = IntentClustering(data_with_docs['function_ids'], data_with_docs['code_reference'])
clusters = doc2clusters.get_clusters(n_clusters=5)
print("Got clusters!")


# Score clusters
clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'])
scored_dataset = clusters2scoredDataset.get_scored_dataset()
print(scored_dataset[0])

print("Scored clusters!")

print(scored_dataset[0])

# Train with Seq2Seq model
model = T5Code2CodeModel("base")
model.train(scored_dataset, "c2c_model_with_chrf_and_nonzero_reps")
# print("Trained model!")

# # Perform inference
# example_bad_function = "def hello_world(): pfds('hello_world')"
# resulting_good_function = model(example_bad_function)["translation_text"]
# print(resulting_good_function)