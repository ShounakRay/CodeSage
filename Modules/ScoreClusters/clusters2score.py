import pandas as pd
from datasets import Dataset
from math import ceil, floor

def compute_reputation_score(code_snippet):
    features = code_snippet["reputation"]
    return features["num_stars"] + features["num_forks"] + features["num_watchers"] - features["num_open_issues"]

def fifty_fifty_threshold(sorted_intents):
  return (sorted_intents[:floor(len(sorted_intents) / 2)], sorted_intents[floor(len(sorted_intents) / 2):]) 


class ScoreClusters():
  def __init__(self, clusters, code_reference, scoring_function=compute_reputation_score, thresholding_function=fifty_fifty_threshold):
    self.clusters = clusters
    self.code_reference = code_reference
    self.scoring_function = scoring_function
    self.thresholding_function = thresholding_function
    self.scored_clusters = {}
  
  def write_to_file(self, df):
     df.to_csv("scored_dataset.csv")

  def get_scored_dataset(self):
    inputs = []
    outputs = []

    for intent_category in self.clusters:
      intents = self.clusters[intent_category]
      if len(intents) == 1: continue
      sorted_intents = sorted(intents, reverse=True, key = lambda intent: self.scoring_function(self.code_reference[intent]))
      self.scored_clusters[intent_category] = self.thresholding_function(sorted_intents)
      
    for intent in self.scored_clusters.keys():
        code_ids = self.scored_clusters[intent]
        for bad_code_id in code_ids[1]:
            inputs.append(self.code_reference[bad_code_id]['code'])
        for good_code_id in code_ids[0]:
            outputs.append(self.code_reference[good_code_id]['code'])
        # outputs.append([self.code_reference[good_code_id]["code"] for good_code_id in code_ids[0]])
    
    dicty = dict();
    dicty["input"] = []
    dicty["target"] = []

    for bad_code in inputs:
      for good_code in outputs:
        dicty["input"].append(bad_code)
        dicty["target"].append(good_code)

    df = pd.DataFrame.from_dict(dicty)
    self.write_to_file(df)

    hf_ds = Dataset.from_pandas(df)
    return hf_ds



