import re
import nltk
import os
import string
import statistics
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
# 输入police_report文件夹路径
# input_dir = filedialog.askdirectory(title='Select the input directory')
input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1-21-20'

# 将yy返回YYYY年份
def conv(yy):
    if int(yy) < 20:
        return int('20' + yy)
    else:
        return int('19' + yy)

# 将文章分为段落
def ParagraphSplit(report):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(report)
    length = len(sentences)
    spliter = round(length/3) #将文章大致分为三段
    paragraph = [' '.join(sentences[0:spliter]),' '.join(sentences[spliter+1:2*spliter]),' '.join(sentences[2*spliter+1:])]
    return paragraph

# 词性统计
tags = set(['RB','IN','VBD'])

#文章分段演示
with open(r'/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1/M510-5424_RMS96-45754.txt','r',encoding='utf-8') as f:
    reader = f.read()
    test = ParagraphSplit(reader)
# print(test)

# 输入文本路径
# input_dir = filedialog.askdirectory(title='Select the input directory')
input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1-21-20'
report_list = os.listdir(input_dir)


# 对分段清洗并对其进行向量化处理
stopwords = stopwords.words('english')   #生成停用词
punctuation_map = dict((ord(char), None) for char in string.punctuation) #生成标点符号
s = nltk.stem.SnowballStemmer('english') # 使用英语提取词干
def word2vec(text):    #输入段落
    l_text = text.lower()     #全部转化为小写以方便处理
    without_punctuation = l_text.translate(punctuation_map)    #去除文章标点符号
    tokens = nltk.word_tokenize(without_punctuation)        #将文章进行分词处理,将一段话转变成一个list
    words_alph = [word.lower() for word in tokens if word.isalpha()]       #去除数字
    without_stopwords = [w for w in words_alph if not w in stopwords]    #去除文章的停用词
    length = len(without_stopwords)   #记录清洗之后段落词量 为得出段落次数的中数
    cleaned_text = []
    for i in range(len(without_stopwords)):
        cleaned_text.append(s.stem(without_stopwords[i]))    #提取词干
    pos_tags = nltk.pos_tag(cleaned_text)   #提取词性
    pos_tags = [pos_tags[i][1] for i in range(len(pos_tags))]
    count = Counter(pos_tags)   #对词性进行计数
    word_vec = []
    for tag in tags:
        if tag not in count:
            word_vec.append(0)
        else:
            word_vec.append(count.get(tag))
    return word_vec,length,count

# 依照年份分段报告 并对报告进行向量化处理
dict = {}
paragraph_list = []
paragraph_vec = []
length_of_paragraph = []
#写出词性统计
os.remove('word_tag.csv')
with open('word_tag.csv','w',encoding='utf-8') as file:
    writer = csv.writer(file)
    for elem in tqdm(report_list):
        complete_name = os.path.join(input_dir, elem)
        with open(complete_name, 'r', encoding='utf-8') as f:
            report_content = f.read()
        date = re.search(r'RMS\d{2}',elem)   #识别年份
        date = re.search(r'\d{2}',date.group())    #识别年份
        full_date = conv(date.group())
        paragraph = ParagraphSplit(report_content)  #将文章分段
        paragraph_list.extend(paragraph)   #保存分段
        paragraph_vec.extend([word2vec(par)[0] for par in paragraph])   #分段向量化
        length_of_paragraph.extend([word2vec(par)[1] for par in paragraph])     #记录分段中词量以计算中位数
        writer.writerow(word2vec(par) for par in paragraph)         #记录词性统计以观察段落词性特征
        if int(full_date) > 1990:
            if full_date not in dict:
                dict[full_date] = paragraph
            else:
                dict.get(full_date).extend(paragraph)


median = statistics.median(length_of_paragraph)    #段落词量中位数
for i in range(len(paragraph_vec)):    #向量除以段落词数的中位数 以获得相对频率
    paragraph_vec[i] = np.divide(paragraph_vec[i],median)

#### 使用PCA对向量进行降维
# 1.数据标准化处理
scaler = StandardScaler()
scaler.fit(paragraph_vec)
paragraph_vec = scaler.transform(paragraph_vec)
# 2.执行PCA降维
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(paragraph_vec)
print(principalComponents)
print(pca.explained_variance_ratio_)

#### 使用Kmeans对段落分类
kmeans = KMeans(n_clusters=3, random_state=0).fit(principalComponents)

plt.scatter(principalComponents[:,0],principalComponents[:,1])
plt.show()
y_kmeans = kmeans.predict(principalComponents)

plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()