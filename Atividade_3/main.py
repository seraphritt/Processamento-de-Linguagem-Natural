from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas


# lendo csv
df = pandas.read_csv("data/Dmoz-Computers.csv")
train_size = int(df.shape[0] * 0.8)
x_train = df['text'][:train_size]
x_test = df['text'][train_size:]
categoriess_on_df = set(df['class'])
count_vector = CountVectorizer()
x_train_tf = count_vector.fit_transform(x_train)
x_test_tf = count_vector.transform(x_test)
# print(x_train_tf)

tfidf_t = TfidfTransformer()
xtrain_tfidf = tfidf_t.fit_transform(x_train_tf)
xtest_tfidf = tfidf_t.transform(x_test_tf)
# print(xtrain_tfidf)

mnb = MultinomialNB().fit(xtrain_tfidf, news_train.target)
predicted = mnb.predict(xtest_tfidf)

# print(f"Accuracy: {metrics.accuracy_score(news_test.target, predicted)}")
# print(confusion_matrix(news_test.target, predicted))
# print(f1_score(news_test.target, predicted, average='macro'))
# print(f1_score(news_test.target, predicted, average='micro'))
