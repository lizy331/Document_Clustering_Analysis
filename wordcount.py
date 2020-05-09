import os
import re
import numpy as np
import matplotlib.pylab as plt
import csv
from tqdm import tqdm
from tkinter import filedialog
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# modifying stopwords
my_stopwords = ['the','when','mater','id','rms','she','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
    'q','r','s','t','u','v','w','x','y','z','us','on','one','two','three','four','five','six','seven','eight','nine',
    'go','get','got','take','token','st','head','given','']
stopwords = stopwords.words('english')
stopwords.extend(my_stopwords)
# print(stopwords)

# format two digits year
def conv(yy):
    if int(yy) < 20:
        return int('20' + yy)
    else:
        return int('19' + yy)

# count the words
def wordcount(words_list):
    count_dic = {}
    words_count = []
    for word in words_list:
        if word not in count_dic:
            count_dic[word] = 1
        else:
            count_dic[word] += 1
    for key in count_dic:
        words_count.append([key,count_dic.get(key)])
    return words_count  # 返还值为[word,count]

# Apply sentiment analysis
input_dir = filedialog.askdirectory(title='Select the input directory')
# input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1-21-20'
report_list = os.listdir(input_dir)
# print(report_list)
dict = {}
for elem in tqdm(report_list):
    # print(elem)
    complete_name = os.path.join(input_dir, elem)
    # print(complete_name)
    with open(complete_name, 'r', encoding='utf-8') as f:
        report_content = f.read()
    date = re.search(r'RMS\d{2}',elem)
    date = re.search(r'\d{2}',date.group())
    full_date = conv(date.group())
    words_token = word_tokenize(report_content)
    words_alph = [word.lower() for word in words_token if word.isalpha()]
    words = [word for word in words_alph if word not in stopwords]
    if int(full_date) > 1990:
        if full_date not in dict:
            dict[full_date] = wordcount(words)
        else:
            dict.get(full_date).extend(wordcount(words))

# sum count from all reports
for key in tqdm(dict):
    word_list = {}
    count = []
    index = []
    for value in dict.get(key):
        if value[0] not in word_list:
            word_list[value[0]] = value[1]
        else:
            word_list[value[0]] += value[1]
    for word in word_list:
        if word_list.get(word) > 10: # 筛选词频超过3次
            count.append([word,int(word_list.get(word))])
    dict[key] = count
    with open(str(key) + '.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        header = ['WORDS','FREQUENCY']
        writer.writerow(header)
        for word in count:
            writer.writerow(word)

    # print(count






# print(dict)
# print(dict.keys())