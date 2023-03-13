from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import json
import torch
import math
from random import choice
from string import ascii_lowercase

class Code2DocModule():
    def __init__(self, snippets, inference_batch_size = 4):
        self.snippets = snippets;
        self.inference_batch_size = inference_batch_size
        self.model = self.train_model()
    
    def train_model(self, batch_size=4):
      model = SummarizationPipeline(
      model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune"),
      tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune", skip_special_tokens=True), device=0)
      return model
    
    def write_to_file(self, data):
      with open("output.json", "w") as f:
        json.dump(data, f)
        
    def get_docs(self):
      code_reference = {}
      function_ids = []

      count = 0
      print("Begin processing {} functions!".format(len(self.snippets["function"])))
      documentations = []
      for i in range(1, math.ceil(len(self.snippets["function"]) / self.inference_batch_size) + 1):
        with torch.no_grad():
          responses = self.model(self.snippets["function"][(i - 1) * self.inference_batch_size: i * self.inference_batch_size])
        documentations += [response["summary_text"] for response in responses]
        print(f"Batch {i} done!")
      print(len(documentations))
      print(len(self.snippets["function"]))
      for i, snippet in enumerate(self.snippets["function"]):
        id = self.snippets["id"][i]
        function_ids.append(id)
        code_reference[id] = {
          "code": snippet,
          "documentation": documentations[i],
          "reputation": {
            "num_stars": self.snippets["features"][i][0],
            "num_forks": self.snippets["features"][i][1],
            "num_watchers": self.snippets["features"][i][2],
            "num_open_issues": self.snippets["features"][i][3]
          }
        }
        count += 1
        print(str(count) + " functions processed.")
      
      print("No. of processed functions: ", len(code_reference.keys()))

      data = {
         "function_ids": function_ids,
         "code_reference": code_reference
      }

      self.write_to_file(data)
      
      return data
