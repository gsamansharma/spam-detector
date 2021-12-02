#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
import pickle


#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

data = df[['v1','v2']]

X = data.iloc[:, 1]
y = data.iloc[:, 0]


tokenizer = RegexpTokenizer('\w+')
sw = set(stopwords.words('english'))
ps = PorterStemmer()

def getStem(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review) 
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review
def getDoc(document):
    d = []
    for doc in document:
        d.append(getStem(doc))
    return d

def prepare(message):
    d = getDoc(message)
    return cv.transform(d)

stemmed_doc = getDoc(X)
cv = CountVectorizer()

vc = cv.fit_transform(stemmed_doc)

X = vc.todense()
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)

naivee = MultinomialNB()
naivee.fit(X_train, y_train)
naivee.score(X_test, y_test)

pickle.dump(naivee, open('model.pkl', 'wb'))






    


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])


def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    p = request.form.get("message")
    
    model = pickle.load(open('model.pkl', 'rb'))

    # message1 = [
    # """
    # England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/̼1.20"""
    # ]
    # print(type(message1))
    # print(message1)
    message=[""""""]
    message.insert(0, p)
    
    print(type(message))
    print(message)
    # message.remove(1)
    print(message)
    message = prepare(message)

    y_pred = model.predict(message)
    print(y_pred)
    output = y_pred[0]
    print(type(output))
    
    return render_template('index.html', prediction_text='The message is a {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


