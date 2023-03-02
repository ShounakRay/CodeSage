from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import json

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
      for snippet in self.snippets:
        for i, func in enumerate(snippet['functions']):
          id = snippet['repo_name']+"_"+snippet['path']+"_"+str(i)

          function_ids.append(id)
          code_reference[id] = {
              "code": func,
              "documentation": self.model([func])[0]['summary_text'],
              "reputation": snippet['reputation_features']
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
