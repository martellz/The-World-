from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.stem.porter import *
from wordcloud import WordCloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import csv
import time

class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
        
    def remove_pattern(self, input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)    
            
        return input_txt    
 
if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    print("*****************************************************")
    train_size = 9846
 
    tweet_analyzer = TweetAnalyzer()
    stemmer = PorterStemmer()
    
    #Train = pd.read_csv("Train.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False) 
    #Test = pd.read_csv("Test.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False) 
    
    Train = pd.read_csv("Train.csv") 
    Test = pd.read_csv("Test.csv") 
    
    print(Train.shape)
    print(Test.shape)
    
    Train['airline_sentiment_int'] = np.where(Train['airline_sentiment']=='neutral', 0,np.where(Train['airline_sentiment']=='positive', 1,-1))
    Test['airline_sentiment_int'] = np.where(Test['airline_sentiment']=='neutral', 0,np.where(Test['airline_sentiment']=='positive', 1,-1))

    data = Train.append(Test, ignore_index=True)

    # remove twitter handles (@user)
    data['tidy_Text'] = np.vectorize(tweet_analyzer.remove_pattern)(data['text'], "@[\w]*")
    
    # remove special characters, numbers, punctuations
    data['tidy_Text'] = data['tidy_Text'].str.replace("[^a-zA-Z#]", " ")
        
    # Removing Short Words
    data['tidy_Text'] = data['tidy_Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    # Tokenization
    tokenized_tweet = data['tidy_Text'].apply(lambda x: x.split())
    #print(tokenized_tweet.head())
    
    # Stemming
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    print(tokenized_tweet.head())
    
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    data['tidy_Text'] = tokenized_tweet
    
    data['airline_sentiment_int'] = np.where(data['airline_sentiment']=='neutral', 0,np.where(data['airline_sentiment']=='positive', 1,-1))
        
    #data.to_csv("export.csv")
    
    """
    fig, ax = plt.subplots(figsize=(5, 6))

    # Plot histogram of the polarity values
    data['airline_sentiment_int'].hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")
    
    plt.title("airline_sentiment_int")
    plt.show()
    """
    
    ########################################################################################################
    # Word Cloud
    all_words = ' '.join([text for text in data['tidy_Text']])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('All words', loc='center', fontsize=20)
    plt.show()
    
    positive_words =' '.join([text for text in data['tidy_Text'][data['airline_sentiment_int'] == 1]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Positive feedback', loc='center', fontsize=20)
    plt.show()
    
    negative_words =' '.join([text for text in data['tidy_Text'][data['airline_sentiment_int'] == -1]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Negative feedback', loc='center', fontsize=20)
    plt.show()
    
    neutral_words =' '.join([text for text in data['tidy_Text'][data['airline_sentiment_int'] == 0]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neutral_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Neutral feedback', loc='center', fontsize=20)
    plt.show()

    ########################################################################################################
    # TextBlob
    TextBlob_Test = data[train_size:]
    #Use TextBlob for analyze
    TextBlob_Test['TextBlob_sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in TextBlob_Test['tidy_Text']])
    print("TextBlob accuracy score : {}".format(accuracy_score(TextBlob_Test['airline_sentiment_int'], TextBlob_Test['TextBlob_sentiment'])))
    
    """
    fig, ax = plt.subplots(figsize=(5, 6))

    # Plot histogram of the polarity values
    data['TextBlob_sentiment'].hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")
    
    plt.title("TextBlob_sentiment")
    plt.show()
    """
       
    ########################################################################################################
    # Bag-of-Words Features
    start_time = time.time()
    
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(data['tidy_Text'])
    
    #print(bow.shape)
    
    train_bow = bow[:train_size,:]
    test_bow = bow[train_size:,:]
    
    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, Train['airline_sentiment_int'], random_state=42, test_size=0.3)
    
    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain) # training the model
    
    prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
        
    prediction = np.where(prediction>=0.6,1,prediction)    
    prediction = np.where(prediction<=0.4,-1,prediction)
    prediction = np.where((prediction>0.4) & (prediction<0.6),0,prediction)
    
    prediction_int = prediction[:,1]    
    prediction_int = prediction_int.astype(np.int)
        
    #print(prediction)
    #print(prediction_int)    
    print("Bag-of-Words accuracy score (Validing): {}".format(f1_score(yvalid, prediction_int, average='micro'))) # calculating f1 score
    print("Bag-of-Words Training time %s seconds." % (time.time() - start_time))
    
    start_time = time.time()
    # Test the Prediction set
    test_pred = lreg.predict_proba(test_bow)
    test_pred_int = test_pred[:,1] >= 0.3
    
    test_pred = np.where(test_pred>=0.6,1,test_pred)    
    test_pred = np.where(test_pred<=0.4,-1,test_pred)
    test_pred = np.where((test_pred>0.4) & (test_pred<0.6),0,test_pred)
    
    test_pred_int = test_pred[:,1]    
    
    test_pred_int = test_pred_int.astype(np.int)
    Test['label'] = test_pred_int
    #submission = Test[['tweet_id','airline_sentiment_int','label']]
    #submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file
    
    print("Bag-of-Words accuracy score (Testing): {}".format(accuracy_score(Test['airline_sentiment_int'], Test['label'])))
    print("Bag-of-Words Testing time %s seconds." % (time.time() - start_time))
    
    ########################################################################################################
    # TF-IDF Features
    """
    train_tfidf = bow[:train_size,:]
    test_tfidf = bow[train_size:,:]
    
    xtrain_tfidf = train_tfidf[ytrain.index]
    xvalid_tfidf = train_tfidf[yvalid.index]
    
    lreg.fit(xtrain_tfidf, ytrain)
    
    prediction = lreg.predict_proba(xvalid_tfidf)
    
    prediction = np.where(prediction>=0.6,1,prediction)    
    prediction = np.where(prediction<=0.4,-1,prediction)
    prediction = np.where((prediction>0.4) & (prediction<0.6),0,prediction)
    
    prediction_int = prediction[:,1]       
    prediction_int = prediction_int.astype(np.int)
    
    print("TF-IDF accuracy score (Validing): {}".format(f1_score(yvalid, prediction_int, average='micro')))
    """
    print("*****************************************************")

