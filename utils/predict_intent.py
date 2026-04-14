from .clean_text import clean_text
import pickle

intent_encoder = pickle.load(open("./models/intent_encoder.pkl", "rb"))
intent_model = pickle.load(open("./models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("./models/vectorizer.pkl", "rb"))

def predict_intent(query):
    query_clean = clean_text(query)
    vec = vectorizer.transform([query_clean])

    probs = intent_model.predict_proba(vec)[0]
    idx = probs.argmax()

    intent = intent_encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])

    return intent, confidence