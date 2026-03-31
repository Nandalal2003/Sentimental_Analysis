import pickle
import pandas as pd
from src.preprocess import preprocess

def load_models():
    vectoriser = pickle.load(open("models/vectoriser.pickle", "rb"))
    model = pickle.load(open("models/lr_model.pickle", "rb"))
    return vectoriser, model

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    data = [(t, s) for t, s in zip(text, sentiment)]
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])

    return df
