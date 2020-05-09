import os
import re
import csv
import math
import nltk
import string
import nltk.stem
from collections import Counter
from tqdm import tqdm
from tkinter import filedialog
from nltk.corpus import stopwords

#将每年频率高的词移除
spc_stopwords = ['rms','rape','mater','id','suspect','state','states','stated','male','sex','case','time','unit','crime','original','narrative','investigation','investigative','victim','victims','police']
stopwords = stopwords.words('english')
stopwords.extend(spc_stopwords)

# 将yy返回YYYY年份
def conv(yy):
    if int(yy) < 20:
        return int('20' + yy)
    else:
        return int('19' + yy)

####### 作为分段的另一种方法
    # n = 5    #每个自然段包含 n 个句子
    # sentences_group = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n)]
    # paragraph = [' '.join(sentences_group[i]) for i in range(len(sentences_group))]
    # print(len(sentences_group))
#######

# 将文章分为段落
def ParagraphSplit(report):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(report)
    n = 5    #每个自然段包含 n 个句子
    sentences_group = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n)]
    paragraph = [' '.join(sentences_group[i]) for i in range(len(sentences_group))]
    return paragraph #输出自然段list

punctuation_map = dict((ord(char), None) for char in string.punctuation)   #生成标点符号
s = nltk.stem.SnowballStemmer('english') # 使用英语提取词干

def stem_count(text):
    l_text = text.lower()     #全部转化为小写以方便处理
    without_punctuation = l_text.translate(punctuation_map)    #去除文章标点符号
    tokens = nltk.word_tokenize(without_punctuation)        #将文章进行分词处理,将一段话转变成一个list
    words_alph = [word.lower() for word in tokens if word.isalpha()]       #去除数字
    without_stopwords = [w for w in words_alph if not w in stopwords]    #去除文章的停用词
    cleaned_text = []
    for i in range(len(without_stopwords)):
        cleaned_text.append(s.stem(without_stopwords[i]))    #提取词干
    count = Counter(cleaned_text)                 #实现计数功能 对每个词进行计数 并返回一个dict
    return count

#定义TF-IDF的计算过程
def D_con(word, count_list):
    D_con = 0
    for count in count_list:
        if word in count:
            D_con += 1
    return D_con
def tf(word, count):
    return count[word] / sum(count.values())
def idf(word, count_list):
    return math.log(len(count_list)) / (1 + D_con(word, count_list))
def tfidf(word, count, count_list):
    return round(tf(word, count) * idf(word, count_list),6)

# 输入police_report文件夹路径
input_dir = filedialog.askdirectory(title='Select the input directory')
# input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1-21-20'
dict = {}
TargetWord = ['unnam','knew','friend','famili','partner','know','meet','roommat','name']   #目标词 用来计算tfidf
global_count_list = []  #段落词干计数
global_paragraph = []   #段落内容
paragraph_vec = []    #段落向量
paragraph_lab = []    #段落向量化标签
paragraph_name = []     #段落所属报告名称
paragraph_tfidf = []    #目标词的tfidf权重
paragraph_tfidf_top = []   #每个自然段的tfidf权重最大的word

date_list = []  #段落日期

for text in tqdm(os.listdir(input_dir)):
    complete_path = os.path.join(input_dir,text)
    with open(complete_path,'r',encoding='utf-8') as f:
        reader = f.read()
    paragraph = ParagraphSplit(reader)
    global_paragraph.extend(paragraph)     #记录分段内容
    date = re.search(r'RMS\d{2}', text)
    date = re.search(r'\d{2}', date.group())
    full_date = conv(date.group())    #从文件名中提取年份
    count_list = []  # 包含每个自然段中词频的统计 dict形式
    count_list.extend(stem_count(paragraph[i]) for i in range(len(paragraph)))    #将自然段提取词干
    global_count_list.extend(stem_count(paragraph[i]) for i in range(len(paragraph)))    #记录段落中提取所有的词 用来观察词干提取效果
    sort = []
    for i in range(len(count_list)):   #在count_list中每个段落提取词中寻找目标词
        date_list.append(full_date)    #记录自然段年份
        paragraph_name.append(str(text).replace('.txt',''))
        tf_idf = {}  #记录目标词的权重
        full_tfidf = {}   #记录所有词的权重
        for word in TargetWord:    #寻找目标词 若段落不包含目标词 tf_idf权重为0
            if word in count_list[i]:
                tf_idf[word] = tfidf(word, count_list[i], count_list)
            else:
                tf_idf[word] = 0
        for word in count_list[i]:  #计算所有词的tfidf并记录 以寻找最大权重的词
            full_tfidf[word] = tfidf(word,count_list[i],count_list)
        paragraph_vec.append([tf_idf[word] for word in tf_idf.keys()])    #记录段落tfidf向量化结果 顺序与目标词相同
        count_sorted = sorted(tf_idf.items(), key = lambda x: x[1], reverse=True)   #记录目标词排序结果
        count_total_sorted = sorted(full_tfidf.items(), key = lambda x: x[1], reverse=True)   #记录所有词排序结果
        paragraph_tfidf.append(count_sorted)
        paragraph_tfidf_top.append(count_total_sorted[0:9])
        paragraph_lab.append(count_sorted[0][0])
        sort.append(count_sorted)   #将集合按照TF-IDF值从大到小排列

    # if int(full_date) > 1990: # 根据年份将段落分类
    #     if full_date not in dict:
    #         dict[full_date] = sort    # dict包含了词以及对应tfidf权重 词顺序由权重大到小排列
    #     else:
    #         dict.get(full_date).extend(sort)

print(len(paragraph_vec))   #段落向量集合
print(len(paragraph_lab))   #段落标签集合
print(len(global_paragraph))    #段落内容集合
print(len(date_list))   #段落日期集合
print(len(paragraph_name))      #段落所属报告名称
print(len(paragraph_tfidf))     #段落tfidf权重
print(len(paragraph_tfidf_top))   #段落所有词tfidf权重


label_year = {}
for i in range(len(paragraph_tfidf)):
    if date_list[i] not in label_year:
        label_year[date_list[i]] = [paragraph_lab[i]]
    else:
        label_year.get(date_list[i]).append(paragraph_lab[i])
# print(label_year)

label_year_count = {}
for year in label_year:
    label_year_count[year] = [['investig',label_year.get(year).count('investig')]]
    label_year_count[year].append(['crime',label_year.get(year).count('crime')])
    label_year_count[year].append(['victim',label_year.get(year).count('victim')])
# print(label_year_count)

# 写出分段内容，标签，TFIDF权重
with open('paragraphTargetWords.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['PoliceReport','Date','Paragraph','Label','TF-IDF']
    writer.writerow(header)
    for i in range(len(paragraph_vec)):
        writer.writerow([paragraph_name[i],date_list[i],global_paragraph[i],paragraph_lab[i],paragraph_tfidf[i]])

# # 写出分段标签计数
# with open('LabelCount.csv','w',encoding='utf-8') as f:
#     writer = csv.writer(f)
#     header = ['Date','ParagraphLabel','CountLabel']
#     writer.writerow(header)
#     for year in label_year_count:
#         for i in range(len(TargetWord)):
#             writer_content = []
#             writer_content.append(year)
#             writer_content.extend(label_year_count.get(year)[i])
#             writer.writerow(writer_content)

# 写出段落所有词的权重 以研究聚类问题
with open('paragraph_tfidf.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['PoliceReport','Date','Paragraph','TF-IDF']
    writer.writerow(header)
    for i in range(len(paragraph_vec)):
        writer.writerow([paragraph_name[i],date_list[i],global_paragraph[i],paragraph_tfidf_top[i]])