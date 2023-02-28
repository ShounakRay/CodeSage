# IMPORTS
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
import requests
import ciso8601
import time
import json
from datetime import datetime

DATASET_URI = "codeparrot/github-code"


class CodeSnippetDataset:
    def construct_feature_set(self, code_entry) -> dict[str:int]:
        features = dict();
        user_repo_name = code_entry['repo_name'].split('/')
        owner = user_repo_name[0]
        repo = "".join(user_repo_name[1:])
        repo_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}")
        content = json.loads(repo_response.text)
        features["num_stars"] = content.get("stargazers_count", 0)
        features["num_forks"] = content.get("forks_count", 0)
        features["num_watchers"] = content.get("watchers_count", 0)
        features["num_open_issues"] = content.get("open_issues_count", 0)
        parsed_datetime = ciso8601.parse_datetime(content.get("created_at", datetime.now().isoformat()))
        timestamp = time.mktime(parsed_datetime.timetuple())
        features["created_at"] = timestamp
        return features
    # THE BELOW IS FOR PYTHON FILES (LEVERAGES INDENTATION RULES)
    def construct_list_of_functions(self, code_entry) -> list[str]:
        raw_code = code_entry["code"]
        lines = raw_code.split('\n')
        start = -1
        functions = []
        amnt_tabs = 0
        for i in range(len(lines)):
            # disregard empty lines (prune trailing whitespace later)
            if (start != -1 and len(lines[i]) > 0):
                amnt_tabs_new = len(lines[i].rstrip()) - len(lines[i].strip())
                if amnt_tabs_new <= amnt_tabs:
                    functions.append(("\n".join(lines[start:i])).strip())
                    start = -1
            elif lines[i].lstrip().startswith("def "):
                start = i
                amnt_tabs = len(lines[i].rstrip()) - len(lines[i].strip())
        return functions
     
    def augment_code_entry(self, entry):
        entry["reputation_features"] = []
        entry["functions"] = self.construct_list_of_functions(entry)
        return entry
        
    def get_n_snippets(self, n: int) -> list[dict]:
        """Provides the next n code snippets from the GitHub dataset. 

        Args:
            n (int): number of snippets desired

        Returns:
            list[dict]: an array of size n, where each element is a dictionary containing the functions / 
            reputation features of a code snippet.
        """
        snippets = self.dataset.take(n).remove_columns("code")
        self.dataset = self.dataset.skip(n)
        return list(snippets)        

    def __init__(self, languages: list[str]) -> None:
        """__init__
        Args:
            languages (list[str]): programming languages to be included. Must be some of "Python", "Javascript", "Java", etc...
                                   for now only supports Python
        """
        
        self.dataset : IterableDataset = load_dataset(DATASET_URI, streaming=True, split='train', languages=languages)
        self.dataset : IterableDataset = self.dataset.map(self.augment_code_entry)
        
# Example Usage
if __name__ == "__main__":
    # ds = CodeSnippetDataset(languages=["Python"])
    # snippets = ds.get_n_snippets(4)
    # print(snippets[0])
    # snippets = ds.get_n_snippets(4)
    # print(snippets[0]) # will NOT be the same as in previous invocation
    # print("\n".join(snippets[0]["functions"]))
    owner = "Azure"
    repo = "azure-sdk-for-python"
    repo_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}")
    content = json.loads(repo_response.text)
    print(content)
