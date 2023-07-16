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


from sklearn.feature_extraction.text import TfidfVectorizer

# Generate a unique filename using uuid
unique_filename = str(uuid.uuid4())

def clean_text(text):
    pat1 = r'@[^ ]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = r'\'s'
    pat4 = r'\#\w+'
    pat5 = r'&amp '
    pat6 = r'[^A-Za-z\s]'
    combined_pat = r'|'.join((pat1,pat2,pat3,pat4,pat5,pat6))
    text = re.sub(combined_pat,"",text).lower()
    return text.strip()

def pre_process(filename):
    #df=pd.read_csv('scrapped/zobia_fifa_khel_2021',encoding='latin-1')
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
    visualizations(scrapped_file_path)
    feature_extraction_english(scrapped_file_path)
    report(scrapped_file_path,unique_filename)


    

def visualizations(filename,uniqueFilename):
    df=pd.read_csv(filename)
    df.drop_duplicates(inplace = True)


    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                collocations=False).generate(" ".join(df['text']))

    plt.title("WordCloud for most used words", size = 24)
    plt.imshow(wc)

    # Generate a unique filename using uuid
    
    # Save the figure with the unique filename
    image_path = os.path.join('media','report_images', uniqueFilename + '_wordCloud'+ '.png')
    plt.savefig(image_path)

    plt.style.use('ggplot')
    plt.figure(figsize=(14,6))
    freq=pd.Series(" ".join(df["text"]).split()).value_counts()[:30]
    freq.plot(kind="bar", color = "teal")
    plt.title("30 most frequent words",size=20)
    image_path = os.path.join('media','report_images', uniqueFilename + '_frequentWords'+ '.png')
    plt.savefig(image_path)
    

def feature_extraction_english(filename):
    df=pd.read_csv(filename)
    df.drop_duplicates(inplace = True)
    data=df.text
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(data)
    feature_extracted_data = vectoriser.transform(data)
    # Loading model
    lr_english_model = pickle.load(open('C:/Users/Zobia/FYP Web/media/models/lr_english.pkl','rb')) #give path of best model
    lr_english_model_pred=lr_english_model.predict(feature_extracted_data)
    
        # Count the occurrences of each class
    class_counts = np.bincount(lr_english_model_pred)

    # Calculate the percentage of each class
    total_samples = len(lr_english_model_pred)
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
   
# def report(filename):
#         # Custom class to overwrite the header and footer methods
#     class PDF(FPDF):
#         def __init__(self):
#             super().__init__()
#         def header(self):
#             self.set_font('Arial', 'B', 12)
#             self.cell(0, 10, 'Report', 0, 1, 'C')

#         def footer(self):
#             self.set_y(-15)
#             self.set_font('Arial', 'I', 8)
#             self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
#     df = pd.read_csv(filename)

#     # Extract text column
#     text_column = df['text']

#     # Create a PDF report
#     pdf = PDF()
#     pdf.add_page()

#     # Add CSV data to the PDF
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(40, 10, 'CSV Data', 0, 1)
#     pdf.set_font('Arial', '', 12)
#     for text_value in text_column[:5]:
#         pdf.cell(0, 10, text_value, 1)
#         pdf.ln()

    
    
#     report_path = os.path.join('media','reports', unique_filename + '_report'+ '.pdf')
#     pdf.output(report_path, 'F')




def report(filename,uniqueFilename):
    # Custom class to overwrite the header and footer methods
    class PDF(FPDF):
        def __init__(self):
            super().__init__()

        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Report', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    
    df = pd.read_csv(filename)
    # Extract text column
    text_column = df['text']

    # Create a PDF report
    pdf = PDF()
    pdf.add_page()
    pdf.image('media/logo.png', x=40, y=100, w=55, h=50)
    pdf.text(110, 160, 'ABSTRACT')
    pdf.text(40, 160, 'DATA ANALYTICS')
    pdf.text(40, 170, 'REPORT')
    absText = ['The following report contains the analytics',
               'performed on the given data set by the customer.',
               'The report not only contains textual information',
               'but also graphical representation of the insights to',
               'help the individual understand the performance.']
    pdf.set_line_width(1)
    pdf.line(100, 70, 100, 200)
    pdf.set_font('Times', '', 12)
    for i in range(len(absText)):
        pdf.text(110, 170 + i*5, absText[i])

    pdf.add_page()

    # Add CSV data to the PDF
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'CSV Data', 0, 1)
    pdf.set_font('Arial', '', 12)
    left_margin = pdf.l_margin
    right_margin = pdf.r_margin
    cell_width = pdf.w - left_margin - right_margin
    for text_value in text_column[:5]:
        pdf.multi_cell(cell_width, 10, text_value, 1)

    #page2
    # Fetch Word Cloud image from file location
    wordcloud_path = os.path.join('media','report_images', uniqueFilename + '_wordCloud'+ '.png')

    # Add Word Cloud image to the PDF
    pdf.add_page()
    #pdf.cell(40, 10, 'Word Cloud', 0, 1)
    pdf.image(wordcloud_path, x=pdf.l_margin, y=pdf.get_y(), w=pdf.w - left_margin - right_margin)

    # Fetch Frequency image from file location
    frequency_path = os.path.join('media','report_images', uniqueFilename + '_frequentWords'+ '.png')

    # Add Frequency image to the PDF
    #pdf.add_page()
    #pdf.cell(140, 10, 'Frequency', 0, 1)
    pdf.image(frequency_path, x=pdf.l_margin, y=pdf.get_y()+100, w=pdf.w - left_margin - right_margin)

    #page3
    # Fetch bar plot image from file location
    barplot_path = os.path.join('media','report_images', uniqueFilename + '_BarPlot'+ '.png')

    # Add bar plot image to the PDF
    pdf.add_page()
    #pdf.cell(40, 10, 'Word Cloud', 0, 1)
    pdf.image(barplot_path, x=pdf.l_margin, y=pdf.get_y(), w=pdf.w - left_margin - right_margin)

    report_path = os.path.join('media', 'reports', uniqueFilename + '_report' + '.pdf')
    pdf.output(report_path, 'F')



#pre_process('media/scrapped/zobia_mehengai_2021_10001.csv')    
#visualizations('media/clean/cleaned_withstopwords.csv')
#feature_extraction_english('media/clean/cleaned_withstopwords.csv')
#report('media/clean/cleaned_withstopwords.csv,unique_filename')

    
    







