import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from src.preprocess import preprocess

def load_data(path):
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    dataset = pd.read_csv(path, encoding="ISO-8859-1", names=DATASET_COLUMNS)

    dataset = dataset[['sentiment', 'text']]
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

    return dataset

def train_models(dataset):
    text = list(dataset['text'])
    sentiment = list(dataset['sentiment'])

    processed = preprocess(text)

    X_train, X_test, y_train, y_test = train_test_split(
        processed, sentiment, test_size=0.05, random_state=0
    )

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)

    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)

    models = {
        "LR": LogisticRegression(C=2, max_iter=1000, n_jobs=-1),
        "BNB": BernoulliNB(alpha=2),
        "SVC": LinearSVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return vectoriser, models, X_test, y_test

def save_models(vectoriser, models):
    pickle.dump(vectoriser, open("models/vectoriser.pickle", "wb"))
    pickle.dump(models["LR"], open("models/lr_model.pickle", "wb"))
    pickle.dump(models["BNB"], open("models/bnb_model.pickle", "wb"))
