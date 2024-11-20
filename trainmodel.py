import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Load preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load the vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Train the model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Model Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Precision:", precision)

# Save the trained model
pickle.dump(mnb, open('model.pkl', 'wb'))
print("Model training complete. Model saved.")
