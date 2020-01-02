import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import re
import seaborn as sns
import xlrd
import csv
import sys
import getopt


def main(argv):
    loc = 'Tweets-airline-sentiment.csv'
    
    try:
        opts, args = getopt.getopt(argv, "l:", ["location="])
    except getopt.GetoptError:
        print('run.py -l <path of training set>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-l", "--location"):
            loc = arg

    df = load_sheet_data(loc)
    print(df.head())
    
    train_df = df
    
    # Training input on the whole training set (i.e. Tweets.csv) with no limit on training epochs.
    #https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/inputs/pandas_input_fn
    
    train_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)
    
    # Prediction on the whole training set.
    prediction_train_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
    
    embedded_text_feature_column = hub.text_embedding_column(
        key="tweets", 
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    
    estimator = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=100,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003))
    
    estimator.train(input_fn=train_fn, steps=1000);
    
    train_eval_result = estimator.evaluate(input_fn=prediction_train_fn)
    print("Training set accuracy: {accuracy}".format(**train_eval_result))

def load_sheet_data(location):
    data = {}
    data["tweets"] = []
    data["sentiment"] = []
    data["polarity"] = []
    with open(location, newline='') as csvfile:
        rows = csv.reader(csvfile)
        
        row_count = 0
        tweets_index = 0
        sentiment_index = 0
        
        for row in rows:
            if row_count == 0:
                #print(str(row))
                tweets_index = row.index("text")
                sentiment_index = row.index("airline_sentiment")
            else:
                data["sentiment"].append(row[sentiment_index])
                
                tweets_lst = row[tweets_index].split(' ')
                
                tweets_str = ''
                
                for i in tweets_lst:
                    if len(i) > 0:
                        if i[0] != '@':
                            tweets_str = "{} {}".format(tweets_str, i)
                        
                print(tweets_str)
                
                data["tweets"].append(tweets_str)
                
                if "negative" in row[sentiment_index].lower():
                    data["polarity"].append(0)
                elif "neutral" in row[sentiment_index].lower():
                    data["polarity"].append(1)
                elif "positive" in row[sentiment_index].lower():
                    data["polarity"].append(2)
                else:
                    data["polarity"].append(-1)
                    
            row_count += 1
        
        #data["tweets"] = data["tweets"][1:]
        #data["sentiment"] = data["sentiment"][1:]
        #data["polarity"] = data["polarity"][1:]      

        print(len(data["tweets"]))
        print(len(data["sentiment"]))
        print(len(data["polarity"]))
    return pd.DataFrame.from_dict(data)

if __name__ == "__main__":
    main(sys.argv[1:])
