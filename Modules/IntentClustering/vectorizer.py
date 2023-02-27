# @Author: shounak.ray
# @Date:   2022-06-29T23:24:32-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T03:11:01-07:00

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

def tfidf_vectorizer(raw_documents, input='content', **kwargs):
    """
    USAGE of TF-IDF – Bag of Words Vectorizer + Transformer:
    tfidf_vectorizer(raw_documents, input='content', max_features = None, use_idf = True, smooth_idf = True, sublinear_tf = True)
        > returns numpy array.
    """
    # Verify structure of raw_documents
    if type(raw_documents) not in [np.ndarray, np.array, list]:
        try:
            raw_documents = np.array(raw_documents)
        except Exception as e:
            print('Failed to convert text to np.array. Proceeding with caution...')

    if input == 'content':
        try:
            if type(raw_documents[0]) is not str:
                raise ValueError('Input was specified as content, but didn\'t get content.')
        except:
            raise ValueError('Failed to verify content-like structure of input. Aborted.')

    # max_features: int, default=None
    #   If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
    vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict',
                                 strip_accents=None, lowercase=True, norm='l2', **kwargs)
    return vectorizer.fit_transform(raw_documents=raw_documents).toarray()


def _get_data(_CATS=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']):
    from sklearn.datasets import fetch_20newsgroups
    return fetch_20newsgroups(subset='train', categories=_CATS, shuffle=True, random_state=42).data


# def _test():
#     _CATS = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
#     return tfidf_vectorizer(_get_data(), input='content', max_features=None, use_idf=True, smooth_idf=True, sublinear_tf=True)

# EOF
