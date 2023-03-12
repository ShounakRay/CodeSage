from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import json
import math

class Code2DocModule():
    def __init__(self, snippets):
        self.snippets = snippets;
        self.model = self.train_model()
    
    def train_model(self):
        model = SummarizationPipeline(
      model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune"),
      tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune", skip_special_tokens=True),
      device=0)
        
        return model
    
    def write_to_file(self, data):
      with open("output.json", "w") as f:
        json.dump(data, f)
    def get_docs(self):
      code_reference = {}
      function_ids = []

      count = 0
      batch_size = 128
      print("Begin processing {} functions!".format(len(self.snippets["function"])))
      for i in range(1, math.ceil(len(self.snippets["function"]) / batch_size)):
        self.model(self.snippets["function"][(i - 1) * batch_size: i * batch_size])
        print(f"Batch {i} done!")
      # responses = self.model(self.snippets["function"])
      
      batch_size
      documentations = [response["summary_text"] for response in responses]
      print(documentations)
      
      for i, snippet in enumerate(self.snippets["function"]):
        id = self.snippets["id"][i]
        function_ids.append(id)
        code_reference[id] = {
          "code": snippet,
          "documentation": documentations[i],
          "reputation": self.snippets["features"]
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
