import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
 ])

reviews = [
    "I love this product. It's amazing! But the service was terrible.",
    "The quality of this product is excellent.",
    "I'm not satisfied with the purchase. The customer support is unresponsive.",
    "This is the worst product I've ever bought. Do not recommend it.",
    "It's an okay product. Nothing special.",
]

print()
print("==========================================================")
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

print(f"First Algorithm: Multinomial Naive Bayes:")

#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

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

print("==========================================================")
print("Second Algorithm: TextBlob")

total_sentiment = 0

for review in reviews:
    analysis = TextBlob(review)
    sentiment = analysis.sentiment.polarity
    total_sentiment += sentiment

    if sentiment > 0:
        sentiment_label = "Positive"
    elif sentiment < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    print("Review:")
    print(review)
    print(f"Sentiment: {sentiment_label} (Polarity: {sentiment})")
    print()

average_sentiment = total_sentiment / len(reviews)

if average_sentiment > 0:
    overall_sentiment_label = "Overall Positive"
elif average_sentiment < 0:
    overall_sentiment_label = "Overall Negative"
else:
    overall_sentiment_label = "Overall Neutral"

print("Average Sentiment Analysis Result:")
print(f"Overall Sentiment: {overall_sentiment_label} (Polarity: {average_sentiment})")

print("==========================================================")
print(f"Third Algorithm: NLTK:")
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
stopwords = nltk.corpus.stopwords.words("english")
words = [w for w in words if w.lower() not in stopwords]
sia = SentimentIntensityAnalyzer()
for review in reviews:
    print("Review:")
    print(review)
    print(sia.polarity_scores(review))
    print()
