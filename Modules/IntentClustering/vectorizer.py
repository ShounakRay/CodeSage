# @Author: shounak.ray
# @Date:   2022-06-29T23:24:32-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T03:11:01-07:00

import pkg_resources
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

# NOTE: For ELMO, can't run this locally because of ARM issues - but need tensorflow version <= 1.9.x
# import tensorflow_hub as hub
# pkg_resources.require("tensorflow==1.9.0")
# import tensorflow as tf

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

def vectorize(data, embedder='pretrained', input='content', **kwargs):
    if embedder == "tfidf":
        embeddings = _tfidf_vectorizer(raw_documents=data,
                                    input='content', max_features=None,
                                    use_idf=True, smooth_idf=True, sublinear_tf=True)
    
    elif embedder == 'STrans':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(data)
    
    elif embedder == "Elmo":
        # TODO: Testing pending on non-ARM VMachine
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            embeddings = sess.run(tf.reduce_mean(embeddings,1))        
    else:
        raise ValueError("We don't support this vectorizer mode yet.")
    
    assert len(embeddings.shape) == 2
    return embeddings

def _tfidf_vectorizer(raw_documents, input='content', **kwargs):
    """
    USAGE of TF-IDF – Bag of Words Vectorizer + Transformer:
    _tfidf_vectorizer(raw_documents, input='content', max_features = None, use_idf = True, smooth_idf = True, sublinear_tf = True)
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

def _get_data(source='json', _CATS=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']):
    assert source in ("json", "newsgroups")
    if source == "newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='train', categories=_CATS, shuffle=True, random_state=42).data
    elif source == "json":
        with open("Modules/IntentClustering/_tempData/functions.json", "r") as f:
            data = [metaData["documentation"] for metaData in list(json.load(f)["code_reference"].values())]
    else:
        raise ValueError("Source not supported")
    return data

if __name__ == "__main__":
    print(_get_data("json"))

# def _test():
#     _CATS = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
#     return tfidf_vectorizer(_get_data(), input='content', max_features=None, use_idf=True, smooth_idf=True, sublinear_tf=True)

# EOF
