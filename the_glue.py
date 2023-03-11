from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters
from Modules.Code2Code.models.t5_code_2_code_model import T5Code2CodeModel
import json 

N_SNIPPETS = 10000;

# Get Dataset
dataset = CodeSnippetDataset(languages=["Python"])
code_snippets = dataset.get_n_snippets(N_SNIPPETS)
print("Got snippets!")

# # Get documentation from dataset
code2doc = Code2DocModule(code_snippets)
data_with_docs = code2doc.get_docs()
print("Got documentations!")

# with open('output.json', 'r') as json_file:
#     data_with_docs = json.load(json_file)

# Turn dataset into clusters
doc2clusters = IntentClustering(data_with_docs['function_ids'], data_with_docs['code_reference'])
clusters = doc2clusters.get_clusters(n_clusters=5)
print("Got clusters!")

# Score clusters
clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'])
scored_dataset = clusters2scoredDataset.get_scored_dataset()
print("Scored clusters!")

# Train with Seq2Seq model
model = T5Code2CodeModel("base")
model.train(scored_dataset, "glued_code_to_code_model")
print("Trained model!")

# Perform inference
example_bad_function = "def hello_world(): pfds('hello_world')"
resulting_good_function = model(example_bad_function)["translation_text"]
print(resulting_good_function)