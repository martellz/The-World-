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
import math


def main(argv):
    #train_loc = ''
    #test_loc = ''
    
    loc = ''
    
    try:
        opts, args = getopt.getopt(argv, "p:", ["path="])
        #opts, args = getopt.getopt(argv, "r:e:", ["train=", "test="])
    except getopt.GetoptError:
        print('run.py -p <path of training set>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-p", "--path"):
            loc = arg

    r_df, e_df = load_sheet_data(loc)
    train_df = r_df
    test_df = e_df
    #sys.exit(2)
    #print(r_df.head())

    
    # Training input on the whole training set (i.e. Tweets.csv) with no limit on training epochs.
    #https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/inputs/pandas_input_fn
    print("Training input on the whole training set, please wait")
    train_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)
    
    # Convert the csv data into input function that need by the model
    prediction_train_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
    prediction_test_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

    # processing the tweet text
    # https://www.tensorflow.org/hub/api_docs/python/hub/text_embedding_column
    # https://tfhub.dev/google/nnlm-en-dim128/1
    # https://databricks.com/glossary/hash-buckets
    embedded_text_feature_column = hub.text_embedding_column(
        key="tweets", 
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    
    # Initialize a classifier for Tensorflow DNN model
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/DNNClassifier
    estimator = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=100,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003))
    
    # Train the model 
    estimator.train(input_fn=train_fn, steps=1000);
    
    #pip3 uninstall gast
    #pip3 install gast==0.2.2
    
    train_eval_result = estimator.evaluate(input_fn=prediction_train_fn)
    test_eval_result = estimator.evaluate(input_fn=prediction_test_fn)
    
    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Testing set accuracy: {accuracy}".format(**test_eval_result))

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
                
                # data pre-processing, remove "@*" and special chars 
                for i in tweets_lst:
                    if len(i) > 0:
                        if i[0] != '@':
                            tweets_str = "{} {}".format(tweets_str, re.sub('[^A-Za-z0-9]+', '', i))

                        
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

        print("Origin dataset length is {}".format(len(data["tweets"])))
        #print(len(data["sentiment"]))
        #print(len(data["polarity"]))

    train_dict = {}
    train_dict["tweets"] = data["tweets"][:(-(math.floor(len(data["tweets"])/4)))]
    train_dict["sentiment"] = data["sentiment"][:(-(math.floor(len(data["sentiment"])/4)))]
    train_dict["polarity"] = data["polarity"][:(-(math.floor(len(data["polarity"])/4)))]
    
    print("Training dataset length is {}".format(len(train_dict["tweets"])))
        
    test_dict = {}
    test_dict["tweets"] = data["tweets"][(-(math.floor(len(data["tweets"])/4))):]
    test_dict["sentiment"] = data["sentiment"][(-(math.floor(len(data["sentiment"])/4))):]
    test_dict["polarity"] = data["polarity"][(-(math.floor(len(data["polarity"])/4))):]
    
    print("Testing dataset length is {}".format(len(test_dict["tweets"])))
        
    return pd.DataFrame.from_dict(train_dict), pd.DataFrame.from_dict(test_dict), 

if __name__ == "__main__":
    main(sys.argv[1:])
