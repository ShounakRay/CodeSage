# IMPORTS
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
import requests
import ciso8601
from typing import Dict
import time
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader
import numpy as np
from Modules.Code2Explanation.code2doc import Code2DocModule
from pprint import pprint
from datasets import Dataset

GITHUB_DATASET_URI = "codeparrot/github-code"
ANNOTATED_DATASET_URI = 'michaelnath/annotated-code-functions-base'
repo_to_features_mapping = dict()
def construct_feature_set(code_entry):
    features = dict();
    user_repo_name = code_entry.split('/')
    owner = user_repo_name[0]
    repo = "".join(user_repo_name[1:])
    header = {"Authorization": 'token '}
    repo_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=header)
    print(repo_response.headers)
    content = json.loads(repo_response.text)
    features["num_stars"] = content.get("stargazers_count", 0)
    features["num_forks"] = content.get("forks_count", 0)
    features["num_watchers"] = content.get("watchers_count", 0)
    features["num_open_issues"] = content.get("open_issues_count", 0)
    parsed_datetime = ciso8601.parse_datetime(content.get("created_at", datetime.now().isoformat()))
    timestamp = time.mktime(parsed_datetime.timetuple())
    features["created_at"] = timestamp
    return [features["num_stars"], features["num_forks"], features["num_watchers"], features["num_open_issues"], features["created_at"]]
# THE BELOW IS FOR PYTHON FILES (LEVERAGES INDENTATION RULES)
def construct_list_of_functions(raw_code):
    lines = raw_code.split('\n')
    start = -1
    functions = []
    begin_considering_function_termination = False
    amnt_tabs = 0
    for i in range(len(lines)):
        # disregard empty lines (prune trailing whitespace later)
        if (start != -1 and len(lines[i]) > 0):
            amnt_tabs_new = len(lines[i].rstrip()) - len(lines[i].strip())
            if amnt_tabs_new <= amnt_tabs and begin_considering_function_termination:
                functions.append(("\n".join(lines[start:i])).strip())
                start = -1
                begin_considering_function_termination = False
        if lines[i].lstrip().startswith(("def ", "async def ")):
            start = i
            amnt_tabs = len(lines[i].rstrip()) - len(lines[i].strip())
        if start != -1 and not begin_considering_function_termination and ":" in lines[i] and ")" in lines[i]:
            begin_considering_function_termination = True 
    return functions

def augment_code_entry(entries):
        entries["functions"] = []
        entries["reputation_features"] = []
        PAD_WORD = "BLEH"
        PAD_LENGTH = 512
        for i in range(len(entries["code"])):
            functions = construct_list_of_functions(entries["code"][i])
            entries["functions"].append(functions + [PAD_WORD] * max(0, PAD_LENGTH - len(functions)))
            if repo_to_features_mapping.get(entries["repo_name"][i], None) == None:
                repo_to_features_mapping[entries["repo_name"][i]] = construct_feature_set(entries["repo_name"][i])
            entries["reputation_features"] += [repo_to_features_mapping[entries["repo_name"][i]]]
        entries["reputation_features"] = torch.Tensor(entries["reputation_features"]).view(len(entries["functions"]), 5)
        return entries

ds = load_dataset("codeparrot/github-code", split="train", streaming=True, languages=["Python"])
BATCH_SIZE = 2
ds=ds.map(augment_code_entry, batched=True, batch_size=BATCH_SIZE, remove_columns=["code", "license", "size", "language"])

dataloader = DataLoader(ds, batch_size=2, num_workers=2)

dicty= dict()
dicty["function"] = []
dicty["repo_name"] = []
dicty["path"] = []
dicty["features"] = []

DESIRED_NUM_FUNCTIONS = 10000

for _, batch in enumerate(dataloader):
    features = batch["reputation_features"]
    functions = np.array(batch["functions"]).reshape(BATCH_SIZE, -1)
    indices = torch.where(torch.all(features > 0, axis=1))[0]
    for index in indices:
        actual_functions = list(functions[index][functions[index] != "BLEH"])
        dicty["function"] += actual_functions
        dicty["repo_name"] += [batch["repo_name"][index]] * len(actual_functions)
        dicty["path"] += [batch["path"][index]] * len(actual_functions) 
        dicty["features"] += [features[index]] * len(actual_functions)
    if len(dicty["function"]) > DESIRED_NUM_FUNCTIONS:
        break

ds = Dataset.from_dict(dicty)

code2doc = Code2DocModule()

# Codex Purpose
ds = ds.add_column("purpose", [code2doc.get_codex_doc(func, "purpose").strip() for func in ds['function']])

# GPT Detailed Description
ds = ds.add_column("detailed_description", [code2doc.get_gpt_doc(func, "detailed_description") for func in ds['function']])

# Code Trans
ds = ds.add_column("code_trans", code2doc.get_code_trans_docs(ds['function']))

# Push to hugging face!
DATASET_NAME = "annotated_github_dataset"
ds.push_to_hub(DATASET_NAME)  