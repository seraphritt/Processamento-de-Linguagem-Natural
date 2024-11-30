from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def read_csv(file_name):
    df = pandas.read_csv(file_name)
    return df


def occ_vectorizer(df_train_text, df_test_text):
    count_vector = CountVectorizer()
    x_train_tf = count_vector.fit_transform(df_train_text)
    x_test_tf = count_vector.transform(df_test_text)  # vetor de ocorrências das palavras
    return x_train_tf, x_test_tf


def tfidif(x_train_tf, x_test_tf):
    tfidf_t = TfidfTransformer()
    xtrain_tfidf = tfidf_t.fit_transform(x_train_tf)  # TFIDF da base de treino
    xtest_tfidf = tfidf_t.transform(x_test_tf)  # TFIDF da base de test
    return xtrain_tfidf, xtest_tfidf


def get_ytrain(x_train_class, dict_cats):
    y_train = []
    for each in x_train_class:
        # dicionário no formato {indice: classe}
        y_train.append(
            dict_cats[each])  # a partir do dataframe é feito um vetor com as classes (labels) da base de treino
    return y_train


def get_ytest(x_test_class, dict_cats):
    y_test = []
    for each in x_test_class:
        # dicionário no formato {indice: classe}
        y_test.append(
            dict_cats[each])  # a partir do dataframe é feito um vetor com as classes (labels) da base de teste
    return y_test


def model_selector(model: str, xtrain_tfidf, xtest_tfidf, y_train):
    # função que seleciona qual modelo será usado
    if model.lower() == "naive bayes":
        mnb = MultinomialNB().fit(xtrain_tfidf, y_train)
        return mnb.predict(xtest_tfidf)
    elif model.lower() == "logistic regression":
        lreg = linear_model.LogisticRegression()
        lreg.fit(xtrain_tfidf, y_train)
        return lreg.predict(xtest_tfidf)
    elif model.lower() == "random forest":
        random_f = RandomForestClassifier(n_estimators=100, random_state=42).fit(xtrain_tfidf, y_train)
        return random_f.predict(xtest_tfidf)
    else:
        print("Select one of these 3 models: naive bayes, logistic regression, random forest")


def grid_search(estimator, parameters, cv=5, scoring='f1_macro', verbose=0, n_jobs=1):
    return GridSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs)


def gridsearchs_data(NB, NBScore, LogReg, LogRegScore, RandomForest, RandomForestScore):
    data = [
        {
            "Model": "Random Forest",
            "Best Score": RandomForest.best_score_,
            "Best Parameters": RandomForest.best_params_,
            "Test Set f1_macro": RandomForestScore
        },
        {
            "Model": "Logistic Regression",
            "Best Score": LogReg.best_score_,
            "Best Parameters": LogReg.best_params_,
            "Test Set f1_macro": LogRegScore
        },
        {
            "Model": "Naive Bayes",
            "Best Score": NB.best_score_,
            "Best Parameters": NB.best_params_,
            "Test Set f1_macro": NBScore
        }
    ]
    return data


def show_results(grid_search, model_name):
    print(model_name)
    print(f"Best score: {grid_search.best_score_:.3f}")
    print(f"Best parameters: {grid_search.best_params_}")