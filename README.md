# AutoFeedback-on-code

## TODO:
[Link Google Doc here]

Immediate Reference:
```
"""
Format that Shounak needs (Joe sends this to him):
[code_id1, code_id2, ...]

Reference dictionary: {code_id: (documentation, actual_code, reputation_score)}

Shounak's output:
{code_id: intent_category}


Michael (seq2seq intermediate step):
{intent_category: (good_code -> List[code_id],
                   bad_code -> List[int])}

Dataset/Input to the seq2seq: completeGraph (every combo) between all the `good_code` and `bad_code`
                                for a given intent_category

"""

# Joe:
# TODO: Update documentation that we use
# (via CodeBert OR another API that provides better documentation) in _get_data
# TODO: Create an alternative way to get reputation scores through Code Reviewer

# Michael:
# TODO: Convert Jupyter notebook into something Shounak can directly pull an output from
# TODO: Make an arbitrary dataset and build a basic seq2seq (includes thresholding based on reputation score)
# TODO: (Eventually) Multiprocess the dataset loading

# Shounak:
# TODO: Find a better embedding for the documentation instead of TF-IDF vectorizer
# TODO: Building out KMeans supervised clustering and comparing it to SOM unsupervised,
#       sanity-check which one is better

## Structure of Respository
```

### _accessories
This are any helper functions that will be shared across any of the Modules.

### Modules
Note: every module has `Extracontent` which is code you're no longer using, but might be important for reference and `_accessories` which are helper functions for your stage process.

#### Stage 1 – Data Ingestion
Hits the `CodeBERT` or `alternative` API.
This ingests the remote dataset and spits out format:
`[code_id1, code_id2, ...]` with reference dictionary `{code_id: (documentation, actual_code, reputation_score)}`.

#### Stage 2 – Intent Clustering
Done via supervised KMeans or unsupervised self-organizing map.
This takes all the code, and spits out format `{code_id: intent_category}` for all `code_id`s.

#### Stage 3 – Code2Code
This is our seq2seq model (RNN –> ... -> Transformer) that:
1. Generates the dataset for the seq2seq
2. Trains the seq2seq model

#### Stage 4 – Code2Explanation
This is able to ingest a given code snippet and output exemplar code snippets and explanations.
Hits the `Codex` API.