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
import emoji


# Import Data Preprocessing Libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Remove stop words from text
from typing import FrozenSet

import urduhack
#urduhack.download()
from urduhack.normalization import normalize
from urduhack.normalization import normalize_characters
from urduhack.preprocessing import normalize_whitespace, remove_punctuation, remove_accents, replace_urls, replace_emails, replace_numbers, replace_currency_symbols, remove_english_alphabets
from urduhack.models.lemmatizer import lemmatizer

from PredictionandAnalysisEnglish import report

# Generate a unique filename using uuid
unique_filename = str(uuid.uuid4())

def remove_stopwords(text: str):
    # Urdu Language Stop words list
    STOP_WORDS: FrozenSet[str] = frozenset("""
    آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
    ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
    اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
    بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
    تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
    جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
    جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
    دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
    رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
    سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
    فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
    لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
    مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
    نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
    وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
    چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
    کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
    کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
    گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
    ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
    """.split())

    return " ".join(word for word in text.split() if word not in STOP_WORDS)

def lemitizeStr(str):
    lemme_str = ""
    temp = lemmatizer.lemma_lookup(str)
    for t in temp:
        lemme_str += t[0] + " "
    
    return lemme_str

def pre_process_urdu(filename):
        data_urdu = pd.read_csv(filename)
        data_urdu['text'] = data_urdu['text'].apply(str)
        data_urdu['text'] = data_urdu['text'].apply(normalize_characters) # To normalize some text, all you need to do pass unicode text. It will return a str with normalized characters both single and combined, proper spaces after digits and punctuations and diacritics(Zabar - Paish) removed.
        data_urdu['text'] = data_urdu['text'].apply(remove_punctuation) # Remove punctuation from text by removing all instances of marks. marks=',;:'
        data_urdu['text'] = data_urdu['text'].apply(remove_accents) # Remove accents from any accented unicode characters in text str, either by transforming them into ascii equivalents or removing them entirely.
        data_urdu['text'] = data_urdu['text'].apply(replace_urls)# Replace all URLs in text str with replace_with str.
        data_urdu['text'] = data_urdu['text'].apply(lambda s: emoji.replace_emoji(s, ''))# Replace all emojis in text str with replace_with str.
        data_urdu['text'] = data_urdu['text'].apply(replace_emails) # Replace all emails in text str with replace_with str.
        # df['review'] = df['review'].apply(replace_numbers) # Replace all numbers in text str with replace_with str.
        data_urdu['text'] = data_urdu['text'].apply(replace_currency_symbols) # Replace all currency symbols in text str with string specified by replace_with str.
        data_urdu['text'] = data_urdu['text'].apply(remove_english_alphabets) # Removes English words and digits from a text
        data_urdu['text'] = data_urdu['text'].apply(normalize_whitespace) ## Given text str, replace one or more spacings with a single space, and one or more linebreaks with a single newline. Also strip leading/trailing whitespace.
        data_urdu['text'] =  data_urdu['text'].apply(remove_stopwords)
        data_urdu['lemmatized_text'] = data_urdu['text'].apply(lemitizeStr)
        scrapped_file_path =os.path.join('media','clean', unique_filename + '_clean'+ '.csv')
        data_urdu.to_csv(scrapped_file_path)
 
def visualizations_urdu(filename):
    df=pd.read_csv(filename)
    print(df.text[3])
    df.drop_duplicates(inplace = True)
   # Convert the Pandas Series to a single string
    text = " ".join(df['text'].astype(str))
    print(text)
    # Word Cloud
    wc = WordCloud(font_path='NotoNastaliqUrdu-Regular.ttf', max_words=1000, width=1600, height=800, collocations=False).generate(text)


    # wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
    #             collocations=False).generate(df['text'])

    plt.title("WordCloud for most used words", size = 24)
    plt.imshow(wc)

    # Generate a unique filename using uuid
    
    # Save the figure with the unique filename
    image_path = os.path.join('media','report_images', unique_filename + '_wordCloud'+ '.png')
    plt.savefig(image_path)

    plt.style.use('ggplot')
    plt.figure(figsize=(14,6))
    freq=pd.Series(" ".join(df["text"].astype(str)).split()).value_counts()[:30]
    freq.plot(kind="bar", color = "teal")
    plt.title("30 most frequent words",size=20)
    image_path = os.path.join('media','report_images', unique_filename + '_frequentWords'+ '.png')
    plt.savefig(image_path)

def feature_extraction_urdu(filename):
    data_urdu = pd.read_csv(filename)
    data_urdu.drop_duplicates(inplace = True)
    data_urdu['lemmatized_text'].fillna('', inplace=True)

    max_feature_num = 3000
    # vectorizer = TfidfVectorizer(max_features=max_feature_num)
    data_predict = TfidfVectorizer(max_features=max_feature_num).fit_transform(data_urdu.lemmatized_text)    
    # Loading model
    lr_urdu_model = pickle.load(open('media/models/LR_model_urdu.sav','rb')) #give path of best model
    lr_urdu_model_pred=lr_urdu_model.predict(data_predict)
            # Count the occurrences of each class
    class_counts = np.bincount(lr_urdu_model_pred)

    # Calculate the percentage of each class
    total_samples = len(lr_urdu_model_pred)
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
    
def report(filename,uniqueFilename):
    # Custom class to overwrite the header and footer methods
    class PDF(FPDF):
        def __init__(self):
            super().__init__()

        def header(self):
            self.set_font('Arial', 'B', 12)
            self.set_font('NotoNastaliqUrdu','B', 12)

            self.cell(0, 10, 'Report', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_font('NotoNastaliqUrdu','I', 8)

            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    
    df = pd.read_csv(filename)
    # Extract text column
    text_column = df['text']

    # Create a PDF report
    pdf = PDF()
    pdf.add_page()
    # Set the Urdu font
    pdf.add_font('NotoNastaliqUrdu', '', 'NotoNastaliqUrdu-Regular.ttf', uni=True)
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

    
visualizations_urdu('media/clean/047390f8-38ea-4287-9621-52d3bfdbffe0_clean.csv')       
#pre_process_urdu('media/scrapped/combined_urdu.csv')
feature_extraction_urdu('media/clean/047390f8-38ea-4287-9621-52d3bfdbffe0_clean.csv')
report('media/clean/047390f8-38ea-4287-9621-52d3bfdbffe0_clean.csv',unique_filename)