from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
le = LabelEncoder()

data = df[['v1','v2']]

X = data.iloc[:, 1]
y = data.iloc[:, 0]

tokenizer = RegexpTokenizer('\w+')
sw = {'both', 'their', "wasn't", "doesn't", 'shan', 'until', 'it', 'any', 'herself', 'me', 'very', 'hers', 'when', 'wouldn', 'no', 'but', 'more', 'ma', 'an', 'there', 'hadn', 'my', "it's", 'yours', 'shouldn', "that'll", 'up', 'over', 'her', 'be', 's', 'so', "don't", 'at', 'needn', 'doing', 'can', 'further', "shan't", 'about', 'how', 'is', 'above', 'haven', 'such', 'yourself', 'was', 'ourselves', 'does', 'weren', 'm', 'this', 'during', 'all', 'didn', 'they', 'by', 'off', 'for', 'that', 'ain', 'd', "you've", 'theirs', 're', "didn't", 'he', 'isn', 'a', "couldn't", 'do', 'which', 'of', 'why', 'from', 'than', 'few', 'only', 'his', 'own', "aren't", 'them', 'o', 'what', 'because', 'mustn', 'hasn', "weren't", 'to', 'on', 't', 'don', 'doesn', 'before', 'we', "should've", 'y', 'some', "hadn't", 'where', 'being', 'your', 'through', 'down', 'in', 'each', 'if', 'with', 'its', 'out', 've', 'aren', 'myself', 'yourselves', 'whom', "mightn't", 'and', 'mightn', 'should', 'did', "wouldn't", 'i', 'those', 'between', 'once', "won't", 'been', 'or', 'you', 'too', 'have', 'having', 'couldn', 'just', 'were', 'll', 'other', 'themselves', 'same', "shouldn't", 'nor', 'itself', 'as', "she's", "mustn't", 'has', 'wasn', 'not', "needn't", "you'll", "you'd", 'here', 'most', 'ours', 'again', 'had', "hasn't", "haven't", 'him', 'himself', 'below', 'then', 'under', 'now', 'while', 'these', 'will', 'won', 'who', 'she', 'am', 'against', "you're", 'after', 'are', 'the', "isn't", 'our', 'into'}

ps = PorterStemmer()

def getStem(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review) # breaking into small words
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review

# get a clean document
def getDoc(document):
    d = []
    for doc in document:
        d.append(getStem(doc))
    return d

stemmed_doc = getDoc(X)

cv = CountVectorizer()

vc = cv.fit_transform(stemmed_doc)

X = vc.todense()


from sklearn import model_selection


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.naive_bayes import MultinomialNB
import pickle

naivee = MultinomialNB()
naivee.fit(X_train, y_train)
naivee.score(X_test, y_test)

pickle.dump(naivee, open('model.pkl', 'wb'))




model = pickle.load(open('model.pkl', 'rb'))



message = [
    """
    England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/̼1.20
    """
]

def prepare(message):
    d = getDoc(message)
    # dont do fit_transform!! it will create new vocab.
    return cv.transform(d)

message = prepare(message)


y_pred = model.predict(message)
print(y_pred)



























