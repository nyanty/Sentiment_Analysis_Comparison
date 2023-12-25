import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("trainc.csv")
X = data.iloc[:,1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train.astype('U'))
X_test_bow = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)

y_pred = classifier.predict(X_test_bow)

reviews = [
    "I love this product. It's amazing!",
    "The quality of this product is excellent.",
    "I'm not satisfied with the purchase. The customer support is unresponsive.",
    "This is the worst product I've ever bought. Do not recommend it.",
    "It's an okay product. Nothing special.",
]

overall = 0
for review in reviews:
    new_sentence_bow = vectorizer.transform([review])
    prediction = classifier.predict(new_sentence_bow)
    print("Review:")
    print(review)
    print(f"Sentiment: {prediction[0]}")
    print()
    if(prediction[0] == 'positive'):
      overall += 1
    if(prediction[0] == 'negative'):
      overall -= 1
if(overall==0):
  print("overall neutral")
if(overall>0):
  print("overall positive")
if(overall<0):
  print("overall negetive")

ns = input("Enter sentence: ")
new_sentence = [ns]

new_sentence_bow = vectorizer.transform(new_sentence)

prediction = classifier.predict(new_sentence_bow)
print(f"Sentiment: {prediction[0]}")