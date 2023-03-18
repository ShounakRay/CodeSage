import itertools 
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, T5ForConditionalGeneration 
from Modules.Code2Explanation.code2doc import Code2DocModule
from Modules.Code2Code.Extracontent.code_snippet_dataset import CodeSnippetDataset
from Modules.IntentClustering.data2clusters import IntentClustering
from Modules.ScoreClusters.clusters2score import ScoreClusters
from Modules.Code2Code.models.t5_code_2_code_model import T5Code2CodeModel
from pprint import pprint

def run_end_to_end_with_parameters(
      FUNCTIONS_DATASET_URI,
      MAX_FUNCTION_STRING_LENGTH,
      C2D_LLM,
      IC_METHOD,
      IC_EMBEDDER,
      IC_KVAL,
      SC_LOWPERC,
      SC_BOUNDARY,
      SC_METHOD,
      SC_SCORING,
      C2C_MODEL_OUTPUT_DIR,
      C2C_LLM,
      C2C_TEST_SIZE,
      C2C_LR,
      C2C_EPOCH_N,
      C2C_BATCH_SIZE,
      C2C_WEIGHT_DECAY,
):
      assert C2D_LLM in  ('CODETRANS', 'CODEX', 'GPT')
      assert IC_METHOD in ("kmeans", "dbscan")
      assert IC_EMBEDDER in ("tfidf", "strans", "elmo")
      assert SC_METHOD in ('PERCENTILE', 'SHARED')
      assert SC_SCORING in ('QUADRATIC', 'LINEAR')
      assert C2C_LLM in ('CODE-T5')
      assert 1 <= C2C_EPOCH_N <= 5
      assert 1 <= C2C_BATCH_SIZE <= 64
      assert 0.01 <= C2C_WEIGHT_DECAY <= 0.1 
      dataset = load_dataset(FUNCTIONS_DATASET_URI, split="train")
      detailed_docs = np.array(dataset["detailed_description"])
      NUM_FUNCTIONS_TO_FINETUNE_UNDER = len(detailed_docs[detailed_docs != ""]) - 500
      C2C_MODEL_OUTPUT_DIR = str(NUM_FUNCTIONS_TO_FINETUNE_UNDER) + "_" + C2C_MODEL_OUTPUT_DIR
      code_snippets = dataset.filter(lambda example: len(example["function"].split()) <= MAX_FUNCTION_STRING_LENGTH)[:NUM_FUNCTIONS_TO_FINETUNE_UNDER]
      code2doc = Code2DocModule()
      data_with_docs = code2doc.get_docs(code_snippets, C2D_LLM = C2D_LLM) 
      doc2clusters = IntentClustering(function_ids=data_with_docs['function_ids'], code_reference=data_with_docs['code_reference'])
      clusters = doc2clusters.core_get_clusters(embedder=IC_EMBEDDER, method=IC_METHOD, n_clusters=IC_KVAL,
                                                eps=0.5, min_samples=5, n_jobs=-1,
                                                doc_source='Detailed' if C2D_LLM == 'GPT' else 'CodeTrans')
      clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'],
                                       SC_SCORING=SC_SCORING,
                                       SC_METHOD=SC_METHOD,
                                       SC_LOWPERC=SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_HIGHPERC=100-SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_BOUNDARY=SC_BOUNDARY if SC_METHOD == 'SHARED' else None)
      scored_dataset = Dataset.from_dict(clusters2scoredDataset.get_scored_dataset().shuffle(seed=420)[:100_000])
      # try:
      #       scored_dataset.push_to_hub(C2C_MODEL_OUTPUT_DIR);
      # except Exception as e:
      #       print(f"Pushing dataset failure due to {e}")
      model = T5Code2CodeModel("small")
      model.train(scored_dataset, 
            C2C_MODEL_OUTPUT_DIR, 
            C2C_TEST_SIZE=C2C_TEST_SIZE, 
            C2C_LR=C2C_LR, 
            C2C_BATCH_SIZE=C2C_BATCH_SIZE, 
            C2C_WEIGHT_DECAY=C2C_WEIGHT_DECAY, 
            C2C_EPOCH_N=C2C_EPOCH_N
      ) 


def compute_quadratic_reputation_score(features):
    return (features[0]**2) + (features[1]**1.5) + features[2]

def create_score_column(entries):
      # print(entries.keys())
      entries["score"] = [compute_quadratic_reputation_score(feature_set) for feature_set in entries["features"]]
      return entries

def simulate_baseline():
      FUNCTIONS_DATASET_URI = "michaelnath/annotated_github_dataset_2"
      ds = load_dataset(FUNCTIONS_DATASET_URI, split='train') 
      augmented_ds = ds.shuffle(seed=420)
      # augmented_ds = ds.map(create_score_column, batched=True, batch_size=8).sort("score")
      first_half = list(augmented_ds[:round(len(augmented_ds)/2)]["function"])
      second_half = list(augmented_ds[round(len(augmented_ds) / 2):]["function"])
      mappings = list(itertools.product(first_half, second_half))[:100_000]
      bad_code = np.array(mappings)[:, 0]
      good_code = np.array(mappings)[:, 1]
      model = T5Code2CodeModel("small")
      print(bad_code[0])
      print(good_code[0])
      scored_dataset = Dataset.from_dict({"input": bad_code, "target": good_code})
      print(scored_dataset[0])
      C2C_MODEL_OUTPUT_DIR = "baseline_codet5_50_50_split_with_reps"
      C2C_TEST_SIZE = 0.02
      C2C_LR = 1e-4
      C2C_BATCH_SIZE = 8
      C2C_WEIGHT_DECAY = 0.01
      C2C_EPOCH_N = 1
      print(len(scored_dataset))
      model.train(scored_dataset, 
            C2C_MODEL_OUTPUT_DIR, 
            C2C_TEST_SIZE=C2C_TEST_SIZE, 
            C2C_LR=C2C_LR, 
            C2C_BATCH_SIZE=C2C_BATCH_SIZE, 
            C2C_WEIGHT_DECAY=C2C_WEIGHT_DECAY, 
            C2C_EPOCH_N=C2C_EPOCH_N
      ) 
      
      


def simulate():
      C2D_LLMS=['GPT', 'CODETRANS']
      # Two
      SC_LOWPERCS=[10, 40]
      SC_BOUNDARIES=[50]
      IC_METHODS=["kmeans", "dbscan"]
      IC_EMBEDDERS=["strans"]
      # Two
      SC_SCORING=["QUADRATIC"]
      # Three
      IC_KVALS=[30, 60, 50, 120, 180, 250]
      C2C_LLMS=['CODE-T5']
      C2C_TEST_SIZE=[0.02]
      C2C_BATCH_SIZES=[16] 
      C2C_WEIGHT_DECAYS=[0.01]
      C2C_EPOCH_NS=[1]
      # Two
      C2C_LR=[0.01, 3e-4, 1e-4]

      shared_combination = itertools.product(
            C2D_LLMS,
            SC_BOUNDARIES,
            IC_METHODS,
            IC_EMBEDDERS,
            SC_SCORING,
            ["SHARED"],
            IC_KVALS,
            C2C_LLMS,
            C2C_TEST_SIZE,
            C2C_BATCH_SIZES,
            C2C_WEIGHT_DECAYS,
            C2C_EPOCH_NS,
            C2C_LR
      )

      percentile_combinations = itertools.product(
            C2D_LLMS,
            SC_LOWPERCS,
            IC_METHODS,
            IC_EMBEDDERS,
            SC_SCORING,
            ["PERCENTILE"],
            IC_KVALS,
            C2C_LLMS,
            C2C_TEST_SIZE,
            C2C_BATCH_SIZES,
            C2C_WEIGHT_DECAYS,
            C2C_EPOCH_NS,
            C2C_LR
      )

      combinations = itertools.chain(shared_combination, percentile_combinations)

      for combination in list(combinations):
            output_dir_name = "_".join([str(x) for x in combination])
      
            try:
                  run_end_to_end_with_parameters(
                        FUNCTIONS_DATASET_URI="michaelnath/annotated_github_dataset_2",
                        MAX_FUNCTION_STRING_LENGTH=512,
                        C2D_LLM = combination[0],
                        SC_LOWPERC = combination[1] if combination[5] == "PERCENTILE" else None,
                        SC_BOUNDARY = combination[1] if combination[5] == "SHARED" else None,
                        IC_METHOD = combination[2],
                        IC_EMBEDDER = combination[3],
                        SC_SCORING = combination[4],
                        SC_METHOD = combination[5],
                        IC_KVAL = combination[6],
                        C2C_LLM = combination[7],
                        C2C_TEST_SIZE=combination[8],
                        C2C_BATCH_SIZE=combination[9],
                        C2C_WEIGHT_DECAY=combination[10],
                        C2C_EPOCH_N=combination[11],
                        C2C_LR=combination[12],
                        C2C_MODEL_OUTPUT_DIR=output_dir_name
                  )
            except Exception as e:
                  print(f"Main failure: {e}")
                  
def evaluate_models_on_test_dataset():
      C2D_LLMS=['GPT', 'CODETRANS']
      # Two
      SC_LOWPERCS=[10, 40]
      SC_BOUNDARIES=[50]
      IC_METHODS=["kmeans", "dbscan"]
      IC_EMBEDDERS=["strans"]
      # Two
      SC_SCORING=["QUADRATIC"]
      # Three
      IC_KVALS=[30, 60, 50, 120, 180, 250]
      C2C_LLMS=['CODE-T5']
      C2C_TEST_SIZE=[0.02]
      C2C_BATCH_SIZES=[16] 
      C2C_WEIGHT_DECAYS=[0.01]
      C2C_EPOCH_NS=[1]
      # Two
      C2C_LR=[0.01, 3e-4, 1e-4]
      shared_combination = itertools.product(
            C2D_LLMS,
            SC_BOUNDARIES,
            IC_METHODS,
            IC_EMBEDDERS,
            SC_SCORING,
            ["SHARED"],
            IC_KVALS,
            C2C_LLMS,
            C2C_TEST_SIZE,
            C2C_BATCH_SIZES,
            C2C_WEIGHT_DECAYS,
            C2C_EPOCH_NS,
            C2C_LR
      )
      
      percentile_combinations = itertools.product(
            C2D_LLMS,
            SC_LOWPERCS,
            IC_METHODS,
            IC_EMBEDDERS,
            SC_SCORING,
            ["PERCENTILE"],
            IC_KVALS,
            C2C_LLMS,
            C2C_TEST_SIZE,
            C2C_BATCH_SIZES,
            C2C_WEIGHT_DECAYS,
            C2C_EPOCH_NS,
            C2C_LR
      ) 
      combinations = itertools.chain(shared_combination, percentile_combinations)

      for combination in list(combinations):
            output_dir_name = "_".join([str(x) for x in combination])
            try:
                  model = T5ForConditionalGeneration.from_pretrained("8681_" + output_dir_name)
            except Exception as e:
                  print("Error Fetching Model as it doesn't exit")
                  continue
            FUNCTIONS_DATASET_URI="michaelnath/annotated_github_dataset_2"
            dataset = load_dataset(FUNCTIONS_DATASET_URI, split="train")
            detailed_docs = np.array(dataset["detailed_description"])
            NUM_FUNCTIONS_TO_FINETUNE_UNDER = len(detailed_docs[detailed_docs != ""]) - 500
            MAX_FUNCTION_STRING_LENGTH=512,
            code_snippets = dataset.filter(lambda example: len(example["function"].split()) <= MAX_FUNCTION_STRING_LENGTH)[NUM_FUNCTIONS_TO_FINETUNE_UNDER:]
            code2doc = Code2DocModule()
            C2D_LLM = combination[0],
            SC_LOWPERC = combination[1] if combination[5] == "PERCENTILE" else None,
            SC_BOUNDARY = combination[1] if combination[5] == "SHARED" else None,
            IC_METHOD = combination[2]
            IC_EMBEDDER = combination[3]
            SC_SCORING = combination[4]
            SC_METHOD = combination[5]
            IC_KVAL = combination[6]
            C2C_TEST_SIZE=combination[8]
            data_with_docs = code2doc.get_docs(code_snippets, C2D_LLM = C2D_LLM) 
            doc2clusters = IntentClustering(function_ids=data_with_docs['function_ids'], code_reference=data_with_docs['code_reference'])
            clusters = doc2clusters.core_get_clusters(embedder=IC_EMBEDDER, method=IC_METHOD, n_clusters=IC_KVAL,
                                                eps=0.5, min_samples=5, n_jobs=-1,
                                                doc_source='Detailed' if C2D_LLM == 'GPT' else 'CodeTrans')
            clusters2scoredDataset = ScoreClusters(clusters, data_with_docs['code_reference'],
                                       SC_SCORING=SC_SCORING,
                                       SC_METHOD=SC_METHOD,
                                       SC_LOWPERC=SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_HIGHPERC=100-SC_LOWPERC if SC_METHOD == 'PERCENTILE' else None,
                                       SC_BOUNDARY=SC_BOUNDARY if SC_METHOD == 'SHARED' else None)
            scored_dataset = Dataset.from_dict(clusters2scoredDataset.get_scored_dataset())
            loader = DataLoader(scored_dataset, batch_size=8, num_workers=2)
            for _, batch in enumerate(loader):
                  print(batch["input"])
                  break
            break
                                               
            
            
       
if __name__ == "__main__":
      # simulate()
      simulate_baseline()
      # evaluate_models_on_test_dataset()
      
