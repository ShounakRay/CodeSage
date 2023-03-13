from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import json
import torch
import math
from random import choice
from string import ascii_lowercase
import openai
openai.api_key = "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"

class Code2DocModule():

    def train_model(self, batch_size=4):
      model = SummarizationPipeline(
      model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune"),
      tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune", skip_special_tokens=True), device=0)
      return model
    
    def write_to_file(self, data):
      with open("output.json", "w") as f:
        json.dump(data, f)
    
    # purpose, runtime
    def get_codex_doc(self, func, type="purpose"):
      if (type == "purpose"):
        prompt = "# Python 3\n" + func + '"""\nThe purpose of the above function is'
        response = openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=50,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["."]
                  )
        return response.choices[0].text
    
    # detailed_description, purpose, runtime 
    def get_gpt_doc(self, func, type="detailed_description"):

      if (type == "detailed_description"):
        prompt = func + "\nWhat does this function do?"
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=50,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                  )
        return response.choices[0].text.strip()
      elif (type == "purpose"):
        prompt = func + "\nThe purpose of this function is"
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=50,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["."]
                  )
        return response.choices[0].text.strip()

    def get_docs(self, snippets, inference_batch_size = 4):
      # train the model!
      model = self.train_model()

      code_reference = {}
      function_ids = []

      count = 0
      print("Begin processing {} functions!".format(len(snippets["function"])))
      documentations = []
      for i in range(1, math.ceil(len(snippets["function"]) / inference_batch_size) + 1):
        with torch.no_grad():
          responses = self.model(snippets["function"][(i - 1) * inference_batch_size: i * inference_batch_size])
        documentations += [response["summary_text"] for response in responses]
        print(f"Batch {i} done!")
      print(len(documentations))
      print(len(snippets["function"]))
      for i, snippet in enumerate(snippets["function"]):
        id = snippets["id"][i]
        function_ids.append(id)
        code_reference[id] = {
          "code": snippet,
          "documentation": documentations[i],
          "reputation": {
            "num_stars": snippets["features"][i][0],
            "num_forks": snippets["features"][i][1],
            "num_watchers": snippets["features"][i][2],
            "num_open_issues": snippets["features"][i][3]
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
