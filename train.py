import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
X_train = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)", 
           "Nah I don't think he goes to usf, he lives around here though", 
           "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!", 
           "I have had enough of this stuff"]
y_train = [1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Train the MultinomialNB model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Save the trained model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
