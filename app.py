# import libraries
import numpy as np
from flask import Flask, request, render_template
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.naive_bayes import MultinomialNB
import pickle
import string
# nltk.download('stopwords')
# Initialize the flask App
app = Flask(__name__)


ps = PorterStemmer()
sw = {'your', 'each', 'too', 'during', 'that', 'again', 'won', "weren't", 'aren', "mightn't", 'own', 'mightn', 'ours', 'it', 'between', "you've", 'will', 'once', 'wouldn', 'its', 'did', 'some', "wasn't", 'because', "shouldn't", "you'd", 'on', 'shan', 'yourselves', 'does', 'having', 'to', 'hers', 'll', 'he', 'itself', 'if', 'had', 'which', 'no', 'are', 'haven', 'don', "you're", "hadn't", "you'll", 'in', 'as', 'and', 'all', 'been', 'through', 'yours', 'be', "won't", 're', "hasn't", 'above', 've', 'me', 'why', 'a', 'shouldn', 'below', 'you', 'doing', "needn't", 'has', 's', 'weren', 'i', "that'll", 'themselves', "don't", 'needn', 'their', 'not', 'the', 'very', 'both', 'y', 'just', "wouldn't", 'other', 'mustn', 'then', 'd', 'she', 'didn', 'an', 'same', 'being', 'up', 'against', 'they', 'is', 'or', "should've", 'o', "shan't", 'whom', 'these', 'herself', 'but', 'under', 'him', 'have', "didn't", 'hasn', 'myself', 'off', 'who', 'when', 'ma', 'most', 't', 'm', 'couldn', "mustn't", 'with', 'was', 'himself', 'until', 'more', "doesn't", 'by', 'now', 'there', 'do', 'after', 'few', 'we', 'any', 'my', 'from', 'yourself', 'wasn', "aren't", "couldn't", "she's", 'out', 'such', 'before', 'where', 'isn', 'down', 'ourselves', 'what', 'while', 'them', 'those', 'only', 'further', "it's", 'nor', 'can', 'theirs', 'his', 'her', 'ain', 'doesn', 'hadn', "haven't", 'should', 'than', 'am', 'so', 'were', 'of', 'for', 'at', 'into', 'how', 'here', 'our', "isn't", 'this', 'about', 'over'}

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in sw and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')






# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """

    p = request.form.get("message")
    transformed_sms = transform_text(p)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    spamresult = ''
    if result == 1:
        spamresult = 'Spam'
    else:
        spamresult = 'Ham'
    return render_template('index.html', query='Message: {}'.format(p), prediction_text='The message is a {}'.format(spamresult))


if __name__ == "__main__":
    app.run(debug=True)
