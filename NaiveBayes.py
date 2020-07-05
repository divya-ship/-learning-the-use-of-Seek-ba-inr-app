import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix
import re
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import string

def processTweet(tweet):
    tweet = re.sub(r'\&\w*;', '', tweet)
    tweet = re.sub('@[^\s]+', '', tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet)
    tweet = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', tweet)
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = tweet.lstrip(' ')
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet


class NaiveBays:
    accuracy = 0
    precision = 0
    fmeasure = 0
    recall = 0
    imagePath = ''
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
    label_encoder = LabelEncoder()

    def trainModel(self, filePath):
        print('Inside trainModel')
        df_inputdata = pd.read_csv('D:\Internships\project\spamham.csv',usecols = [0,1],encoding='latin-1' )
        df_inputdata.rename(columns = {'v1':'Category','v2': 'Message'}, inplace = True)
        df_inputdata['Message'] = df_inputdata['Message'].apply(processTweet)
        # convert the labels from text to numbers
        NaiveBays.label_encoder= preprocessing.LabelEncoder()
        NaiveBays.label_encoder.fit(df_inputdata['Category'])
        df_inputdata['Category']=NaiveBays.label_encoder.transform(df_inputdata['Category'])
        X = df_inputdata.Message
        y = df_inputdata.Category
        # Split the dataset into 80% and 20% for training and testing respectively
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        NaiveBays.tfidf_vect.fit(X_train)
        xtrain_tfidf = NaiveBays.tfidf_vect.transform(X_train)
        xvalid_tfidf = NaiveBays.tfidf_vect.transform(X_test)
        model = naive_bayes.MultinomialNB()
        model.fit(xtrain_tfidf, y_train)
        y_pred = model.predict(xvalid_tfidf)
        NaiveBays.accuracy = metrics.accuracy_score(y_test, y_pred)
        # Save the trained model into hard disk
        modelfilename = 'NaiveBayesModel.sav'
        pickle.dump(model, open(modelfilename, 'wb'))
        # Confusion metrix
        # Get the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
        #plt.figure(figsize=(8, 5))
        #sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
        #imageFile = 'NaiveBayes_Confusion.jpg'
        #plt.savefig(imageFile)
        # perfomance parameters
        imagePath = ''
        print('Image fil path', imagePath)

        NaiveBays.precision = metrics.precision_score(y_test, y_pred, average=None)
        NaiveBays.recall = metrics.recall_score(y_test, y_pred, average=None)
        NaiveBays.fmeasure = metrics.f1_score(y_test, y_pred, average=None)

        print('NaiveBayes Model', 'Training Completed')

    def getAccuracy(self):
        return NaiveBays.accuracy

    def getPerfmatrix(self):
        return NaiveBays.precision, NaiveBays.recall, NaiveBays.fmeasure, NaiveBays.imagePath

    def getPrediction(self, inputTweet):
        myData = np.array([inputTweet])
        myData = NaiveBays.tfidf_vect.transform(myData)
        filename = 'NaiveBayesModel.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(myData)
        print(y_pred)
        vals = NaiveBays.label_encoder.inverse_transform([y_pred[0]])
        print(vals[0])
        return vals[0]
        #return NaiveBays.label_encoder.inverse_transform([y_pred[0]])