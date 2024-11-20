import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load the TF-IDF vectorizer and the trained model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and description
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("Email/SMS Spam Classifier")
st.markdown("""
This application classifies messages as **Spam** or **Not Spam** based on the content you enter. 
Please input the message below and click on the **Predict** button.
""")

# Text area for user input with improved styling
input_sms = st.text_area("Enter the message", placeholder="Type your message here...", height=150)

# Button for prediction
if st.button('Predict', key='predict_button'):
    if input_sms:
        # Preprocess the input message
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the preprocessed input
        vector_input = tfidf.transform([transformed_sms])
        
        # Make a prediction using the trained model
        result = model.predict(vector_input)[0]
        
        # Display the result with a conditional format
        if result == 1:
            st.success("**Result:** This message is classified as **Spam**! 🚫")
        else:
            st.success("**Result:** This message is classified as **Not Spam**. ✅")
    else:
        st.warning("Please enter a message to classify.")

# Optional: Add footer with additional information or links
st.markdown("---")
st.markdown("Created by [Manohar Singh](#) | [ManoharSingh1311](https://github.com/ManoharSingh1311) | [Contact](+91 6399121342)")
