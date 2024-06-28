import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit code to get user input
st.title("SMS Classifier")
input_text = st.text_area("Enter the SMS text")

if st.button("PREDICT"):
    if input_text:
        # Transform the input text using the loaded vectorizer
        vector_input = vectorizer.transform([input_text])

        # Predict using the loaded model
        result = model.predict(vector_input)[0]

        # Display the result
        if result == 1:
            st.write("Classification Result: Spam")
        else:
            st.write("Classification Result: Not Spam")
    else:
        st.write("Please enter some text to classify.")
