
import streamlit as st
import numpy as np
import pickle
import json
import random
import datetime
import requests
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and preprocessing assets
model = load_model('chatbot_model.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Get max sequence length used during training
max_length = model.input_shape[1]

# News API
def get_airbus_news():
    news_api_key = 'fb0bfd4c2cf14463b04f68cff6390a22'  # Replace with your own API key
    url = f"https://newsapi.org/v2/everything?q=airbus&language=en&sortBy=publishedAt&apiKey={news_api_key}"

    response = requests.get(url).json()

    if 'articles' in response:
        airbus_news = [
            f"{article['title']} - {article['source']['name']}"
            for article in response['articles']
            if "Airbus" in article['title']
        ]

        return "\n".join(airbus_news) if airbus_news else "No recent Airbus-related news found."
    else:
        return "Error fetching Airbus news."
def stock_airbus():
    airbus = yf.Ticker("AIR.PA")  # Airbus is listed on Euronext Paris
    stock_history = airbus.history(period="1d")

    if not stock_history.empty:
        stock_price = stock_history['Close'].iloc[0]
        return f"Airbus Stock Price: €{stock_price}"
    else:
        return "No stock data available for Airbus today."

# Get chatbot response
def chatbot_response(text):
    words = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]
    seq = tokenizer.texts_to_sequences([" ".join(words)])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')

    prediction = model.predict(padded_seq, verbose=0)[0]
    tag = classes[np.argmax(prediction)]

    for intent in data['intents']:
        if intent['tag'] == tag:
            if tag == "date":
                return str(datetime.date.today())
            elif tag == "latest_news":
                return get_airbus_news()
            elif tag == "stock":
                return stock_airbus()
            else:
                return random.choice(intent['responses'])  # Return a random response

    return "I'm not sure about that. Can you ask something else?"

# Streamlit App Setup
st.set_page_config(page_title="Chatbot 🤖", page_icon="💬")
st.title("🤖 Airbus Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chatbot_response(prompt)

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
