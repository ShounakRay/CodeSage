from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters

N_SNIPPETS = 10;

# Get Dataset
dataset = CodeSnippetDataset(languages=["Python"])
code_snippets = dataset.get_n_snippets(N_SNIPPETS)

# Get documentation from dataset
code2doc = Code2DocModule(code_snippets)
data_with_docs = code2doc.get_docs()

# Turn dataset into clusters
doc2clusters = IntentClustering(data_with_docs['function_ids'], data_with_docs['code_reference'])
clusters = doc2clusters.get_clusters()

# Score clusters
clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'])
scored_dataset = clusters2scoredDataset.get_scored_dataset()

# Train with Seq2Seq model

print(clusters)
