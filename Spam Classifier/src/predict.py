from preprocessing import clean_text

def predict(email, model, vectorizer): 
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    return "SPAM" if model.predict(vector)[0] == 1 else "HAM"