from data_loader import load_data
from preprocessing import clean_text
from vectorizer import build_vectorizer
from model import train_model
from predict import predict

def main(): 
    df = load_data("data/enron_spam_data.csv")
    df["clean"] = df["text"].apply(clean_text)

    vectorizer = build_vectorizer()

    X = vectorizer.fit_transform(df["clean"])
    y = df["label"]

    model = train_model(X, y)

    while True: 
        email = input("\nEnter email (or exit): ")
        if email.lower() == "exit": 
            break
        print(predict(email, model, vectorizer))

if __name__ == "__main__": 
    main()