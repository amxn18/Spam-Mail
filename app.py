import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("mail_data.csv")
mailData = df.where(pd.notnull(df), '')

# Encode labels
encoder = LabelEncoder()
mailData['Category'] = encoder.fit_transform(mailData['Category'])  # 1 --> spam, 0 --> ham

# Features and labels
x = mailData['Message']
y = mailData['Category']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Vectorize the text
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train_vectorized, y_train)

# Streamlit app
st.set_page_config(page_title="📧 Spam Mail Detector", layout="centered")
st.title("📬 Spam Mail Detection App")
st.markdown("Enter a message below to check whether it's spam or not.")

# Input box
user_input = st.text_area("✍️ Enter your message here:", height=150)

# Predict
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message before clicking Predict.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("🚨 This mail is SPAM!")
        else:
            st.success("✅ This mail is NOT SPAM.")
