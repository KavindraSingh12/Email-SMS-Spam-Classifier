
# Email/SMS Spam Classifier

## Overview
The **Email/SMS Spam Classifier** is a machine learning project that uses Natural Language Processing (NLP) techniques to classify messages as either **Spam** or **Not Spam**. Built with Python, it leverages TF-IDF vectorization and a trained model to analyze message content and make predictions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Model Training](#model-training)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
Spam messages are unsolicited communications often sent in bulk, usually for advertising or malicious purposes. This project aims to provide a tool that detects and classifies messages as spam or non-spam, helping users efficiently filter unwanted messages.

## Features
- User-friendly web interface built with Streamlit.
- Text preprocessing, including lowercasing, tokenization, stopword removal, and stemming.
- Message classification using machine learning with TF-IDF vectorization.
- Real-time feedback on message classification (Spam or Not Spam).

## Technologies Used
- **Python**: Programming language.
- **Streamlit**: Framework for creating web applications.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **scikit-learn**: Machine learning library for model training.
- **Pandas & NumPy**: Libraries for data manipulation and analysis.
- **Pickle**: Used to save and load the trained model and vectorizer.

## Project Structure
```
SMS-Spam-Detection/
├── app.py                 # Streamlit app for user interaction
├── preprocess.py          # Script for data preprocessing
├── trainmodel.py          # Script for model training
├── spam.csv               # Dataset for training and testing
├── vectorizer.pkl         # Pickled TF-IDF vectorizer
├── model.pkl              # Pickled trained classification model
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── venv/                  # Virtual environment (optional)
└── downloadnltk.py        # NLTK resource downloader
```

## Installation
To set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SMS-Spam-Detection.git
   cd SMS-Spam-Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or `source venv/bin/activate` for macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## How to Use
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Enter a message in the text box and click the Predict button. The app will classify the input as either Spam or Not Spam and display the result.

## Model Training
The training process includes:
- **Data Cleaning**: Removing unnecessary columns and duplicates.
- **Text Preprocessing**: Lowercasing, tokenizing, removing stopwords/punctuation, and stemming words.
- **Vectorization**: TF-IDF vectorizer to convert text into numerical features.
- **Model Building**: Training a classifier (e.g., Naive Bayes, Logistic Regression) using scikit-learn.
- **Saving the Model**: Pickling the trained model and vectorizer for future use.

## Results
The trained model accurately distinguishes spam from non-spam messages. The app provides immediate feedback upon message input.

## Future Improvements
- Enhance the interface with additional styling and visual elements.
- Integrate additional models for ensemble learning.
- Implement a feedback loop for user corrections to further improve model accuracy.

## Acknowledgements  
- **NLTK** for text preprocessing tools.
- **scikit-learn** for machine learning model implementation.
- Dataset providers for valuable training data.

## Contact
For any questions or feedback, please reach out to: `kavindrakapkoti5@gmail.com`

---

This `README.md` provides a comprehensive overview, installation steps, and guidance on using and further developing the project.
