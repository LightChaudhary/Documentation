import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text): 
    text = text.lower()
    text = re.sub(r'\d+', '', text) #removes digits
    text = re.sub(r'[^\w\s]', '', text) #removes punctuations
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)
