from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from joblib import dump, load

def train_model(X, y): 
    model = MultinomialNB()
    model.fit(X, y)
    return model

def save_model(model, path):
    dump(model, path)

def load_model(path): 
    return load(path)