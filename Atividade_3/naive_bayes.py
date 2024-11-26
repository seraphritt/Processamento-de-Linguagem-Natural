from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import utils


# lendo csv
df = utils.read_csv("data/classic4.csv")
# divide a base de dados em 80% train e 20% test
x_train, x_test = train_test_split(df, test_size=0.2, stratify=df['class'])

categories_on_df = list(set(df['class']))   # pega todas as categorias da base de dados
dict_cats = {y: x for x, y in enumerate(categories_on_df)}
lista_cats = list(dict_cats.keys())
print(lista_cats)
print(f"Numero de classes: {len(lista_cats)}")

x_train_tf, x_test_tf = utils.occ_vectorizer(x_train['text'], x_test['text'])

tfidf_t = TfidfTransformer()
xtrain_tfidf, xtest_tfidf = utils.tfidif(x_train_tf, x_test_tf)

y_train = utils.get_ytrain(x_train['class'], dict_cats)
y_test = utils.get_ytest(x_test['class'], dict_cats)

predicted = utils.model_selector("naive bayes", xtrain_tfidf, xtest_tfidf, y_train)

print(f"Accuracy: {metrics.accuracy_score(y_test, predicted)}")
print(confusion_matrix(y_test, predicted))
print(f1_score(y_test, predicted, average='macro'))
print(f1_score(y_test, predicted, average='micro'))