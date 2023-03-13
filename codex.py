from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.Code2Explanation.code2doc import Code2DocModule

MAX_FUNCTION_STRING_LENGTH = 512
N_SNIPPETS = 100
DOCUMENTATION_INFERENCE_BATCH_SIZE = 4

# Get Dataset
dataset = CodeSnippetDataset(github=False, languages=["Python"])
code_snippets = dataset.get_n_snippets(N_SNIPPETS, max_length=MAX_FUNCTION_STRING_LENGTH)
print("Got snippets!")

code2doc = Code2DocModule()

for func in code_snippets['function']:
  print("Purpose: " + code2doc.get_codex_doc(func, "purpose"))
  # print("Purpose: " + code2doc.get_gpt_doc(func, "purpose"))
  # print("Detailed Description: " + code2doc.get_gpt_doc(func, "detailed_description"))