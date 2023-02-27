# @Author: shounak.ray
# @Date:   2022-06-29T23:59:42-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T00:00:03-07:00

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
#
# # Data import
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#
# # Vectorizer – bag of words
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# count_vect.vocabulary_.get(u'from', 'Not found!')
#
# # TF-IDF – Transformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# TF_IDF_Vectorized = X_train_tfidf.toarray()
#
# df = pd.DataFrame(TF_IDF_Vectorized, index=count_vect.get_feature_names(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print(df.head(25))
# df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print(df.head(25))
