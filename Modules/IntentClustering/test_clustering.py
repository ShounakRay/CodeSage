from git import Repo
import os

if not os.path.isdir('Modules/IntentClustering/Reference'):
    os.makedirs('Modules/IntentClustering/Reference')
    Repo.clone_from("https://github.com/TheAlgorithms/Python.git", 'Modules/IntentClustering/Reference')

"""
1. [Optionally] Remove docstrings from each python file (so the documentation isn't that easy to generate)
2. [Normal]     Assign a function id to each function (same format as `code_snippets`)
3. [Normal]     Get documentation for each function id (through processing in the glue) via `Code2DocModule`
4. [Special]    Make a dictionary mapping {true_folder/cluster : List[function_ids]} and assign each "true_folder/cluster" a color
5. [Normal]     Retrieve a dictionary mapping {{detected_cluster : List[function_ids]}}
6. [Special]    Visualize/Create {{detected_cluster : List[function_ids]}} clusters but breakdown content on true-clusters/colors in
                    each detected cluster
7. [Interpret]  The more fragmentation in each "detected_cluster" aka distribution of each "true_folder/cluster", the worse
                    the algorithm each. Fragmentation can be measured as the weighted # of "true_folder/clusters" in
                    each "detected_cluster"
8. [Interpret]  Just qualitatively look at each of the clusters and see if it makes sense.
9. [Compare]    Qualitatively see if the KMeans clustering on the "Reference" dataset splits into similar categories as the
                    "True Dataset"
"""

# EOF