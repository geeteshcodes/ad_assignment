import streamlit as st
import pickle
import re

# Load
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    confidence = max(prob)
    
    return pred, confidence

# UI
st.set_page_config(page_title="AI Review Analyzer", layout="centered")

st.title("🎬 AI Review Analyzer")
st.write("Analyze sentiment of movie reviews using Machine Learning")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        pred, confidence = predict_sentiment(user_input)

        if pred == 1:
            st.success(f"Positive 😊")
        else:
            st.error(f"Negative 😠")

        st.write(f"Confidence: {confidence*100:.2f}%")

        # 🔹 Progress bar (visual upgrade)
        st.progress(float(confidence))