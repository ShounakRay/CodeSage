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

# code_reference = {
#   "1": {
#     "code": "def has_size(self): return self.has_size_",
#     "documentation": "True if has size False otherwise",
#     "reputation": [1000,10,10,10]
#   },
#   "2": {
#     "code": "def set_max(self, x):\n    self.has_max_ = 1\n    self.max_ = x",
#     "documentation": "Set maximum value",
#     "reputation": [10,10,10,10]
#   },
#   "3": {
#     "code": "def has_max(self): return self.has_max_",
#     "documentation": "in_range",
#     "reputation": [10,10,10,10]
#   },
#   "4": {
#     "code": "def reserve_list(self): return self.reserve_",
#     "documentation": "list( )",
#     "reputation": [10,10,10,10]
#   },
#   "5": {
#     "code": "def mutable_reserve(self, i):\n    return self.reserve_[i]",
#     "documentation": "mutable reserve function",
#     "reputation": [10,10,10,10]
#   },
#   "6": {
#     "code": "def clear_reserve(self):\n    self.reserve_ = []",
#     "documentation": "Clear reserve array .",
#     "reputation": [10,10,10,10]
#   },
#   "7": {
#     "code": "def set_trusted(self, x):\n    self.has_trusted_ = 1\n    self.trusted_ = x",
#     "documentation": "Set the trusted value for this message .",
#     "reputation": [10,10,10,10]
#   },
#   "8": {
#     "code": "def has_trusted(self): return self.has_trusted_",
#     "documentation": "data(self",
#     "reputation": [10,10,10,10]
#   },
#   "9": {
#     "code": "def Equals(self, x):\n    if x is self: return 1\n    if self.has_header_ != x.has_header_: return 0\n    if self.has_header_ and self.header_ != x.header_: return 0\n    if self.has_model_key_ != x.has_model_key_: return 0\n    if self.has_model_key_ and self.model_key_ != x.model_key_: return 0\n    if self.has_size_ != x.has_size_: return 0\n    if self.has_size_ and self.size_ != x.size_: return 0\n    if self.has_max_ != x.has_max_: return 0\n    if self.has_max_ and self.max_ != x.max_: return 0\n    if len(self.reserve_) != len(x.reserve_): return 0\n    for e1, e2 in zip(self.reserve_, x.reserve_):\n      if e1 != e2: return 0\n    if self.has_trusted_ != x.has_trusted_: return 0\n    if self.has_trusted_ and self.trusted_ != x.trusted_: return 0\n    return 1",
#     "documentation": "Compare two instance objects for equality",
#     "reputation": [10,10,10,10]
#   },
# }
  
# clusters = {
#   "1": [],
#   "2": ["6", "7", "8", "9", "4", "1"],
# }
# if __name__ == '__main__':

#   scored_clusters = {}
#   for intent_category in clusters:
#     intents = clusters[intent_category]
#     if len(intents) == 1: continue
#     sorted_intents = sorted(intents, reverse=True, key = lambda intent: compute_quadratic_reputation_score(code_reference[intent]))
#     response = percentile(compute_quadratic_reputation_score, code_reference, sorted_intents, [90, 90])
#     if response[0] == [] or response[1] == []:
#         continue
#     scored_clusters[intent_category] = response

#   print(scored_clusters)
    
