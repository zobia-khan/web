import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import uuid
import pickle
import seaborn as sns
from fpdf import FPDF
import csv
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

from PredictionandAnalysisEnglish import visualizations,clean_text,report
unique_filename = str(uuid.uuid4())


def pre_process_roman_urdu(filename):
    df = pd.read_csv(filename)
    df['text'] = df['text'].apply(str)
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(lambda x: x.split())
    df['length'] = df['text'].apply(lambda x: len(x))
    word_count=[1,2,3]
    df = df[df["length"].isin(word_count) == False]
    df=df.drop('length', axis=1)
    scrapped_file_path =os.path.join('media','clean', unique_filename + '_clean'+ '.csv')
    df.to_csv(scrapped_file_path)
    visualizations(scrapped_file_path,unique_filename)
    feature_extraction_roman_urdu(scrapped_file_path)
    # report(scrapped_file_path,unique_filename)
    
def feature_extraction_roman_urdu(filename):
    df=pd.read_csv(filename)
    df.drop_duplicates(inplace = True)
    data = [i for i in df['text']]
    sent = []
    import ast
    for i in data:
        sent.append(ast.literal_eval(i))
    print (sent[0:20])
    
    # Now, In order to prepare vector of the data, we take the average of the vectors of the words of data.
    w2v_model = Word2Vec.load('media/models/word_embeddings_roman_urdu/word2vec.model')
    def word_vector(tokens, size):
        vec,count = np.zeros(size).reshape((1, size)),0
        for word in tokens:
            try:
                vec += w2v_model.wv[word].reshape((1, size))
                count += 1
            except KeyError: # handling the case where the token is not in the vocabulary             
                continue
                
        if count != 0:
            vec /= count
        return vec
    # # Now, preparing Word2vec feature set for test features:

    data_array_w2v = np.zeros((len(data), 300))

    for i in range(len(data)):
        data_array_w2v[i,:] = word_vector(data[i], 300)
        
    data_features_w2v = pd.DataFrame(data_array_w2v) 
    scaler = MinMaxScaler()
    scaleddata = scaler.fit_transform(data_features_w2v)
     # Loading model
    lr_roman_urdu_model = pickle.load(open('media/models/roman_urdu/lr_romanurdu_2.pkl','rb')) #give path of best model
    lr_roman_urdu_model_pred=lr_roman_urdu_model.predict(scaleddata)
        # Count the occurrences of each class
    class_counts = np.bincount(lr_roman_urdu_model_pred)

    # Calculate the percentage of each class
    total_samples = len(lr_roman_urdu_model_pred)
    class_percentages = (class_counts / total_samples) * 100

    # Define the class labels for the x-axis
    class_labels = ['Class 0', 'Class 1']

    # Generate the x-axis indices based on the number of classes
    x_indices = np.arange(len(class_labels))

    # Create the bar plot
    plt.figure(figsize=(6,6))

    plt.bar(x_indices, class_counts, align='center')

    # Display the percentage of each class on the plot
    for i, count in enumerate(class_counts):
        plt.text(x_indices[i], count, f'{class_percentages[i]:.2f}%', ha='center', va='bottom')

    # Customize the plot
    plt.xticks(x_indices, class_labels)
    plt.xlabel('Predicted Classes')
    plt.ylabel('Count')
    plt.title('Bar Plot of Predicted Classes')
    image_path = os.path.join('media','report_images', unique_filename + '_BarPlot'+ '.png')
    plt.savefig(image_path)

pre_process_roman_urdu('media/scrapped/zobia_mehengai_2021_10001.csv')
#feature_extraction_roman_urdu('')