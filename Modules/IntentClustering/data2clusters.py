from sklearn.cluster import KMeans
import numpy as np
from Modules.IntentClustering.vectorizer import _get_data, vectorize
import json

class IntentClustering():
    def __init__(self, function_ids, code_reference, IC_ALGO, IC_KVAL):
      self.function_ids = function_ids
      self.code_reference = code_reference
      self.IC_ALGO = IC_ALGO
      self.IC_KVAL = IC_KVAL

    def write_to_file(self, data):
      with open('clusters.json', 'w') as fp:
        json.dump(data, fp)

    def kmeans(self, n_clusters):
      doc_to_id = {}
      for metaData in list(self.code_reference.items()):
          code_id, info = metaData
          doc_to_id[info["documentation"]] = code_id

      # data = _get_data(source = "json")
      data = [metaData["documentation"] for metaData in list(self.code_reference.values())]
      v_data = vectorize(data=data, mode="pretrained")
      kmeans = KMeans(n_clusters=n_clusters, random_state=41, n_init="auto").fit(v_data)

      clusters = {}
      for i in range(len(kmeans.labels_)):
        cluster_id = int(kmeans.labels_[i])
        data_app = doc_to_id[data[i]]
        if cluster_id not in clusters.keys():
            clusters[cluster_id] = [data_app]
        else:
            clusters[cluster_id].append(data_app)

      self.write_to_file(clusters)
      
      return clusters

    def dbscan(self):
       return NotImplementedError
    
    def saveSOM(self):
       return NotImplementedError

    def get_clusters(self):
      if self.IC_ALGO == "KMEANS":
         return self.kmeans(n_clusters=self.IC_KVAL)
      elif self.IC_ALGO == "KMEANS/SOM":
         # Save SOM
         self.saveSOM()
         SOM_NUMCLUSTERS = self.IC_KVAL
         return self.kmeans(n_clusters=SOM_NUMCLUSTERS)
      elif self.IC_ALGO == "DBSCSAN":
         return self.dbscan()
      else:
         raise ValueError(f"Clustering algorithm {self.IC_ALGO} not applicable.")

# EOF