import sentiment
import os
import re
import numpy as np
import matplotlib.pylab as plt
import csv
from tqdm import tqdm
from tkinter import filedialog

# Download necessary packages
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# format two digits year
def conv(yy):
    if int(yy) < 20:
        return int('20' + yy)
    else:
        return int('19' + yy)

# Initiate SentimentAnalysis
stmnt = sentiment.SentimentAnalysis()

# Apply sentiment analysis
input_dir = filedialog.askdirectory(title='Select the input directory')
# input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1-21-20'
report_list = os.listdir(input_dir)
# print(report_list)
dict = {}
for elem in tqdm(report_list):
    complete_name = os.path.join(input_dir, elem)
    # print(complete_name)
    with open(complete_name, 'r', encoding='utf-8') as f:
        report_content = f.read()
    sco = stmnt.score(report_content)
    date = re.search(r'RMS\d{2}',elem)
    date = re.search(r'\d{2}',date.group())
    full_date = conv(date.group())
    if int(full_date) > 1990:
        if full_date not in dict:
            dict[full_date] = [sco]
        else:
            dict.get(full_date).append(sco)


# Compute the average of score and format the year
sorted_dict = {}
size = []
scores_year = []
x = []
ave_score = []
for key in sorted(dict.keys()):
    size.append(len(dict.get(key)))
    scores_year.append(dict.get(key))
    sorted_dict[key] = stmnt.average(dict.get(key))
    x.append(key)
    ave_score.append(stmnt.average(dict.get(key)))

print(size)
print(sorted_dict)

# Write out the csv file
zipped = zip(x,ave_score,size)
zipped = list(zipped)
print(zipped)
os.remove('SentimentScore.csv')
with open('SentimentScore.csv','w',encoding='utf-8',newline='') as f:
    header = ['Year','Score',"Size"]
    writer = csv.writer(f)
    writer.writerow(header)
    for elem in zipped:
        writer.writerow(elem)

# Scatter Plot of the score
plt.figure(1)
plt.style.use('seaborn')
plt.scatter(x, ave_score, s = size, edgecolors='black', linewidths=1,alpha=0.75)

# zip joins x and y coordinates in pairs
for i,j in zip(x,ave_score):
    label = "{:.4f}".format(j)
    plt.annotate(label, # this is the text
                 (i,j), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Sentiment Plot')
plt.xlabel('Year')
plt.ylabel('Sentiment Score')
plt.tight_layout()
plt.xticks(np.arange(1990, 2021, 1))

# Bar Chart of the size
plt.figure(2)
xpos = np.arange(len(x))
plt.xticks(xpos,x)
plt.bar(xpos,size)
# zip joins x and y coordinates in pairs
for i,j in zip(xpos,size):
    label = "{:d}".format(j)
    plt.annotate(label, # this is the text
                 (i,j), # this is the point to label
                 textcoords="offset pixels", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Report Size')
plt.xlabel('Year')
plt.ylabel('Size')
plt.show()











