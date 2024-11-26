from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas
from sklearn import linear_model
# lendo csv
df = pandas.read_csv("data/classic4.csv")
df_copy = df.copy()
x_train, x_test = train_test_split(df_copy, test_size=0.2, stratify=df_copy['class'])
categories_on_df = list(set(df['class']))
dict_cats = {y: x for x, y in enumerate(categories_on_df)}
lista_cats = list(dict_cats.keys())
print(lista_cats)
print(f"Numero de classes: {len(lista_cats)}")

count_vector = CountVectorizer()
x_train_tf = count_vector.fit_transform(x_train['text'])
x_test_tf = count_vector.transform(x_test['text'])

tfidf_t = TfidfTransformer()
xtrain_tfidf = tfidf_t.fit_transform(x_train_tf)
xtest_tfidf = tfidf_t.transform(x_test_tf)
y_train = []
for each in x_train['class']:
    y_train.append(dict_cats[each])
y_test = []
for each in x_test['class']:
    y_test.append(dict_cats[each])
lreg = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lreg.fit(xtrain_tfidf, y_train)
predicted = lreg.predict(xtest_tfidf)
# print(xtrain_tfidf)
print(f"Accuracy: {metrics.accuracy_score(y_test, predicted)}")
print(confusion_matrix(y_test, predicted))
print(f1_score(y_test, predicted, average='macro'))
print(f1_score(y_test, predicted, average='micro'))
linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')