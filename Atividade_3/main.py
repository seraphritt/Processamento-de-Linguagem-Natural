from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas


# lendo csv
df = pandas.read_csv("data/Dmoz-Computers.csv")
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(df.shape[0] * 0.8)
x_train = df['text'][:train_size]
x_test = df['text'][train_size:]
categories_on_df = list(set(df['class']))
dict_cats = {y: x for x, y in enumerate(categories_on_df)}
print(dict_cats)
print(categories_on_df)
count_vector = CountVectorizer()
x_train_tf = count_vector.fit_transform(x_train)
x_test_tf = count_vector.transform(x_test)
# print(x_train_tf)

tfidf_t = TfidfTransformer()
xtrain_tfidf = tfidf_t.fit_transform(x_train_tf)
xtest_tfidf = tfidf_t.transform(x_test_tf)
# print(xtrain_tfidf)
v = []
for each in df['class'][:train_size]:
    v.append(dict_cats[each])
mnb = MultinomialNB().fit(xtrain_tfidf, v)
predicted = mnb.predict(xtest_tfidf)
print(df['class'][train_size:])
p = []
for each in df['class'][train_size:]:
    p.append(dict_cats[each])
print(p)
print(predicted)
print(f"Accuracy: {metrics.accuracy_score(p, predicted)}")
print(confusion_matrix(p, predicted))
print(f1_score(p, predicted, average='macro'))
print(f1_score(p, predicted, average='micro'))