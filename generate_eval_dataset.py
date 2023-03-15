from datasets import load_dataset
from Modules.Code2Explanation.code2doc import Code2DocModule

code2doc = Code2DocModule()
ANNOTATED_DATASET_URI = 'michaelnath/functions_annotated_with_intents'
ds = load_dataset(ANNOTATED_DATASET_URI, split='train')

# Codex Purpose
ds = ds.add_column("purpose", [code2doc.get_codex_doc(func, "purpose").strip() for func in ds['function']])

# Code Trans
ds = ds.add_column("code_trans", code2doc.get_code_trans_docs(ds['function']))

# GPT Detailed Description
ds = ds.add_column("detailed_description", [code2doc.get_gpt_doc(func, "detailed_description") for func in ds['function']])

DATASET_NAME = "functions_annotated_with_intents"
ds.push_to_hub(DATASET_NAME)  
