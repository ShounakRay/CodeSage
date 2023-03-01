from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset

dataset = CodeSnippetDataset(languages=["Python"])
code_snippets = dataset.get_n_snippets(10)

code2doc = Code2DocModule(code_snippets)
data_with_docs = code2doc.get_docs()

print(data_with_docs)
