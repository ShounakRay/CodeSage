from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from Modules.IntentClustering.vectorizer import vectorize
import json
import _pickle as cPickle
import numpy as np

from datasets import load_dataset
from multiprocessing import Pool
# from Modules.IntentClustering.multiprocessed import MULT_preprocess
import os

"""
FOR TESTING PURPOSES:
git clone https://github.com/TheAlgorithms/Python/tree/master
Each folder is it's own intent category (roughly)
Just feed these through the clustering algorithm(s)
See how many clusters it comes up with
"""

class IntentClustering():
   def __init__(self, function_ids, code_reference):
      self.function_ids = function_ids
      self.code_reference = code_reference

   def write_to_file(self, data):
      with open('clusters.json', 'w') as fp:
        json.dump(data, fp)

   def _preprocess(self, embedder="STrans", load_numpys=True, save_numpys=False, model='Detailed'):
      # We're skipping this part for local testing
      # with Pool(os.cpu_count() - 1) as pool:
      #    mapping = pool.starmap(MULT_preprocess, list(self.code_reference.items()))
      # doc_to_id = dict(mapping)

      if load_numpys:
         with open(f'Modules/IntentClustering/storeData/{model}/data-{model}.pickle', 'rb') as fp:
            data = cPickle.load(fp)
         with open(f'Modules/IntentClustering/storeData/{model}/v_data-{model}.pickle', 'rb') as fp:
            v_data = cPickle.load(fp)
         with open(f'Modules/IntentClustering/storeData/{model}/doc_to_id-{model}.pickle', 'rb') as fp:
            doc_to_id = cPickle.load(fp)
      else:
         doc_to_id = {}
         for code_id, info in self.code_reference.items():
            documentation_i = info["documentation"]
            doc_to_id[documentation_i] = code_id

         data = list(doc_to_id.keys())
         v_data = vectorize(data=data, embedder=embedder)
         print("Did vectorization on size: ", len(data))

      if save_numpys:
         with open(f'Modules/IntentClustering/storeData/{model}/data-{model}.pickle', 'wb') as fp:
            cPickle.dump(data, fp) # , protocol=pickle.HIGHEST_PROTOCOL)
         with open(f'Modules/IntentClustering/storeData/{model}/v_data-{model}.pickle', 'wb') as fp:
            cPickle.dump(v_data, fp) # , protocol=pickle.HIGHEST_PROTOCOL)
         with open(f'Modules/IntentClustering/storeData/{model}/doc_to_id-{model}.pickle', 'wb') as fp:
            cPickle.dump(doc_to_id, fp) # , protocol=cPickle.HIGHEST_PROTOCOL)

      return data, v_data, doc_to_id

   def _postprocess(self):
      clusters = {}
      for index in range(len(self._labels)):
         cluster_id = self._labels[index]
         documentation_i = self.data[index]
         data_app = str(self.doc_to_id[documentation_i])
         """
         Exmaple format of `doc_to_id.`
         {'Sends Geolocation data to falcon REST API': 'fd798e8f-157a-4214-9ddd-5cbfe607fa51'}
         """
         # raise ValueError
         # Next line is for testing only
         # data_app = documentation_i
         if cluster_id not in clusters.keys():
            clusters[cluster_id] = [data_app]
         else:
            clusters[cluster_id].append(data_app)
   
      return clusters

   def kmeans(self, n_clusters=10, n_jobs=-1):
      kmeans = KMeans(n_clusters=n_clusters, random_state=41, n_init=5).fit(self.v_data)
      self._labels = kmeans.labels_ = np.array(kmeans.labels_, dtype=int)

   def dbscan(self, eps=0.5, min_samples=5, n_jobs=-1):
      dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(self.v_data)
      self._labels = dbscan.labels_ = np.array(dbscan.labels_, dtype=int)
   
   def core_get_clusters(self, embedder="strans", method='kmeans',
                         n_clusters=10, eps=0.5, min_samples=2, n_jobs=-1,
                         doc_source='Detailed'):
      method, embedder = method.lower(), embedder.lower()
      
      assert embedder in ("tfidf", "strans", "elmo")
      assert method in ("kmeans", "dbscan")
      assert type(n_clusters) == int
      assert 0.001 <= eps <= 1.0
      assert n_jobs in (None, -1)
      assert type(min_samples) == int
      
      print("Clustering\tPre-processing...")
      self.data, self.v_data, self.doc_to_id = self._preprocess(embedder=embedder,
                                                                save_numpys=False,
                                                                load_numpys=False,
                                                                model=doc_source)
      
      if method == 'kmeans':
         print("Clustering\tRunning KMeans...")
         self.kmeans(n_clusters=n_clusters, n_jobs=n_jobs)
      elif method == 'dbscan':
         print("Clustering\tRunning DBSCAN...")
         self.dbscan(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
      else:
         return ValueError("No other methods implemented yet.")
      print("Clustering\tPost-processing...")
      clusters = self._postprocess()

      # This is the format mapping cluster_ids to list of function_ids in a dictionary
      return clusters

   # def saveSOM(self):
   #    # Class/categorical balance is important!
   #    neurons = 5 * np.sqrt(len(data)) / 2
   #    learning_rate = 0.4
   #    epochs = 50000       # Can be determined by likelihood that every sample is seen in the data. (or change algo accordingly)
   #    sigma_0 = 100       # Should be some function of num_features. Pull harder if there's a lot of complexity.
   #    convergence_threshold = 1e-4
   #    # You want to observe the progression of a pattern slowly
   #    # TODO: Add Threading
   #    # Use PCA initialization for reproducibility and accuracy

   #    S = SOM(neurons=neurons, learning_rate=learning_rate, epochs=epochs, sigma_0=sigma_0, convergence_threshold=convergence_threshold, neuron_dim=2)
   #    S.create_feature_map(len(data[0]))
   #    # grid_neurons vs mds_neurons
   #    S.fit(data, animate=True, animate_method='mds_neurons', anim_every_n_epochs=10, only_draw_nodes=False)
   #    # S._plot_neuronal_mds(fname='json_neuronal_mds.jpeg', save=True)
   #    S._plot_neuronal_grid(fname='json_neuronal_grid.jpeg', save=True)

   #    def get_clusters(self):
   #    if self.IC_ALGO == "KMEANS":
   #       return self.kmeans(n_clusters=self.IC_KVAL)
   #    elif self.IC_ALGO == "KMEANS/SOM":
   #       # Save SOM
   #       self.saveSOM()
   #       SOM_NUMCLUSTERS = self.IC_KVAL
   #       return self.kmeans(n_clusters=SOM_NUMCLUSTERS)
   #    elif self.IC_ALGO == "DBSCSAN":
   #       return self.dbscan()
   #    else:
   #          raise ValueError(f"Clustering algorithm {self.IC_ALGO} not applicable.")


if __name__ == "__main__":
   # We're going to test our intent clustering code based on the
   #     HuggingFace dataset of 30,000 functions
   dataset = load_dataset("michaelnath/annotated-code-functions-base", split = "train")
   dataset[0].values()


# EOF