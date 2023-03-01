import pandas as pd
from datasets import Dataset
from math import ceil, floor

def compute_reputation_score(code_snippet):
    features = code_snippet["reputation"]
    return features["num_stars"] + features["num_forks"] + features["num_watchers"] - features["num_open_issues"]

class ScoreClusters():
  def __init__(self, clusters, code_reference):
    self.clusters = clusters;
    self.code_reference = code_reference;
    self.scored_clusters = []
  
  def get_scored_dataset(self):
    inputs = []
    outputs = []

    for intent_category in self.clusters:
      intents = self.clusters[intent_category]
      sorted_intents = sorted(intents, reverse=True, key = lambda intent: compute_reputation_score(self.code_reference[intent]))
      self.scored_clusters[intent_category] = (sorted_intents[:5], sorted_intents[5:])
      
    for intent in self.scored_clusters.keys():
        code_ids = self.scored_clusters[intent]

        for bad_code_id in code_ids[1]:
            inputs.append(self.code_reference[bad_code_id]['code'])
        
        for good_code_id in code_ids[0]:
            outputs.append(self.code_reference[good_code_id]['code'])
    
    dicty = dict();
    dicty["input"] = []
    dicty["target"] = []

    for bad_code in inputs:
      for good_code in outputs:
        dicty["input"].append(bad_code)
        dicty["target"].append(good_code)

    df = pd.DataFrame.from_dict(dicty)
    hf_ds = Dataset.from_pandas(df)

    return hf_ds



