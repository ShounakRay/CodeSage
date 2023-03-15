import pandas as pd
from datasets import Dataset
from math import ceil, floor
import numpy as np

# calculate score! 
# [num_stars, num_forks, num_watchers, num_open_issues]
def compute_linear_reputation_score(code_snippet):
    features = code_snippet["reputation"]
    return features[0] + features[1] + features[2]

def compute_quadratic_reputation_score(code_snippet):
    features = code_snippet["reputation"]
    return (features[0]**2) + (features[1]**1.5) + features[2]

def shared(_, __, sorted_intents, thresholding_params):
  boundary = thresholding_params[0]
  return (sorted_intents[:floor(len(sorted_intents) / (100/boundary))], sorted_intents[floor(len(sorted_intents) / (100/(boundary))):])

def percentile(scoring_function, code_reference, sorted_intents, thresholding_params):
   lower_perc = thresholding_params[0]
   higher_perc = thresholding_params[1]
   scores = [scoring_function(code_reference[intent]) for intent in sorted_intents]

   if (len(set(scores)) == 1):
      return ([], [])
   exemplars = []
   bad_functions = []
   for i, intent in enumerate(sorted_intents):
     if scores[i] >= np.percentile(scores, higher_perc):
       exemplars.append(intent)
     elif scores[i] < np.percentile(scores, lower_perc):
       bad_functions.append(intent)
  
   return (exemplars, bad_functions)


class ScoreClusters():
  def __init__(self, clusters, code_reference, SC_SCORING, SC_METHOD, SC_LOWPERC, SC_HIGHPERC, SC_BOUNDARY):
    self.clusters = clusters
    self.code_reference = code_reference
    self.scored_clusters = {}
    
    if (SC_SCORING == 'QUADRATIC'): 
      self.scoring_function = compute_quadratic_reputation_score
    elif (SC_SCORING == 'LINEAR'):
       self.scoring_function = compute_linear_reputation_score

    if (SC_METHOD == 'SHARED'): 
      self.thresholding_function = shared
      self.thresholding_params = [SC_BOUNDARY]
    elif (SC_METHOD == 'PERCENTILE'):
      self.thresholding_function = percentile
      self.thresholding_params = [SC_LOWPERC, SC_HIGHPERC]
  
  def write_to_file(self, df):
     df.to_csv("scored_dataset.csv")

  def get_scored_dataset(self):
    inputs = []
    outputs = []

    for intent_category in self.clusters:
      intents = self.clusters[intent_category]
      if len(intents) == 1: continue
      sorted_intents = sorted(intents, reverse=True, key = lambda intent: self.scoring_function(self.code_reference[intent]))
      response = self.thresholding_function(self.scoring_function, self.code_reference, sorted_intents, self.thresholding_params)
      if response[0] == [] or response[1] == []:
         continue
      self.scored_clusters[intent_category] = response
      
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
    # self.write_to_file(df)

    hf_ds = Dataset.from_pandas(df)
    return hf_ds