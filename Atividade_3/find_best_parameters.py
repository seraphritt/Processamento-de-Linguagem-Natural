import utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
import pandas as pd
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

predicted_NB = utils.model_selector("naive bayes", xtrain_tfidf, xtest_tfidf, y_train)
# gridsearch para naive bayes
# alpha: parâmetro de suavização (smoothing) aditiva (Laplace/Lidstone)
# fit_prior: se for True, o modelo aprende a priorizar classes. False significa que o modelo usa uma distribuição uniforme e não prioriza as classes
# force_alpha: se for False e alpha for menor que 1e-10, alpha é fixado em 1e-10. Caso True, o alpha permanece inalterado em todos os casos.
param_grid = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'fit_prior': [True, False],
    'force_alpha': [True, False],
}
# usa-se cv = 5 por padrão, ou seja, cross-validation com 5 folds
# usa-se f1-macro por padrão da função
grid_search_NB = utils.grid_search(MultinomialNB(), param_grid)

grid_search_NB.fit(xtrain_tfidf, y_train)
utils.show_results(grid_search_NB, "Naive Bayes Grid Search")

best_model_NB = grid_search_NB.best_estimator_
f1_macroNB = best_model_NB.score(xtest_tfidf, y_test)
print(f"Test set f1_macro: {f1_macroNB:.3f}")
print("--------------------------------------------------------------------------------------------------------------")

predicted_LRG = utils.model_selector("logistic regression", xtrain_tfidf, xtest_tfidf, y_train)
# gridsearch para logistic regression
# C: parâmetro responsável pelo inverso da força da regulatização. Quanto mais próximo de 0, mais forte é a regularização.
# solver: algoritmo de escolha que será utilizado para a otimização. Foram escolhidos apenas 2 porque são os únicos que aceitam ambas regularizações l1 e l2.
# penalty: especifica a norma da penalidade. l1, l2 ou ambas (elasticnet)
param_grid = {
    'C': [0.1, 0.5, 1.0],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2'],
}

grid_search_LogReg = utils.GridSearchCV(linear_model.LogisticRegression(), param_grid)

grid_search_LogReg.fit(xtrain_tfidf, y_train)

utils.show_results(grid_search_LogReg, "Logistic Regression Grid Search")

best_model_LogReg = grid_search_LogReg.best_estimator_
f1_macroLR = best_model_LogReg.score(xtest_tfidf, y_test)
print(f"Test set f1_macro: {f1_macroLR:.3f}")
print("--------------------------------------------------------------------------------------------------------------")

# gridsearch para svm
# n_estimators: número de árvores na forest
# max_depth: A profundidade máxima da árvore
# min_samples_split: o número mínimo de amostras necessárias para dividir um nó interno da árvore
# min_samples_leaf: o número mínimo de amostras que pode estar em um nó folha

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_RF = utils.GridSearchCV(RandomForestClassifier(), param_grid, verbose=1, n_jobs=-1)
grid_search_RF.fit(xtrain_tfidf, y_train)

utils.show_results(grid_search_LogReg, "Random Forest Grid Search")

best_modelRF = grid_search_RF.best_estimator_
f1_macroRF = best_modelRF.score(xtest_tfidf, y_test)
print(f"Test set f1_macro: {f1_macroRF:.3f}")
print("--------------------------------------------------------------------------------------------------------------")
# salvando todas as informações em um csv por meio de um dataframe
data = utils.gridsearchs_data(grid_search_NB, f1_macroNB, grid_search_LogReg, f1_macroLR, grid_search_RF, f1_macroRF)
df = pd.DataFrame(data)

csv_file_path = 'Grid_Search_Results.csv'
df.to_csv(csv_file_path, index=False)
