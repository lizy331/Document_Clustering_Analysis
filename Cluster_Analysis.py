import sentiment
import os
import re
import math
import csv
from tqdm import tqdm
from tkinter import filedialog
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#将每年频率高的词移除
spc_stopwords = ['rms','received','told','car','assigned','arrested','rape','mater','id','suspect','state','states','stated','male','sex','case','time','unit','crime','crimes','original','narrative','investigation','investigative','victim','victims','police']
stopwords = stopwords.words('english')
stopwords.extend(spc_stopwords)

# 启动情感分析
stmnt = sentiment.SentimentAnalysis()

# 词频统计
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

# 报告文件夹位置
input_dir = '/Users/liziyang/Downloads/NJIT_Sandbox-selected/Report_population'
report_list = os.listdir(input_dir)

# 报告类别文件夹位置
cluster_dir = r'/Users/liziyang/Downloads/Cluster_csv_data'
cluster_list = os.listdir(cluster_dir)

# 报告类型
output = []
for elem in tqdm(cluster_list):
    TargetRMS = [] # 记录RMS搜寻报告文件
    complete_name = os.path.join(cluster_dir,elem)
    print(complete_name)
    with open(complete_name, 'r', encoding='utf-8') as csvf:
        cluster = csv.reader(csvf)
        print()
        for line in cluster:
            if line[2] == '1':
                TargetRMS.append('RMS' + line[1])
    words = []     # 搜寻目标文件RMS 并对文件进行清洗 单词计数
    score = []  # 情感分析得分
    for item in report_list:
        RMS_ext = re.search(r'RMS\d{2}-?\d{2,8}',item)
        RMS = RMS_ext.group()
        if RMS in TargetRMS:
            complete_name = os.path.join(input_dir, item)
            with open(complete_name, 'r', encoding='utf-8') as f:
                report_content = f.read()
            sco = stmnt.score(report_content)   # 情感分析
            score.append(sco)
            words_token = word_tokenize(report_content)
            words_alph = [word.lower() for word in words_token if word.isalpha()]
            words_local = [word for word in words_alph if word not in stopwords]
            words.extend(words_local)
    avg_score = stmnt.average(score)
    output.append([elem.replace('.csv',''),avg_score])
    count_list = wordcount(words)   # 对目标类别报告进行词频统计
    output_path = r'/Users/liziyang/Downloads/Cluster_output/'
    # with open(str(output_path) + str(elem),'w',encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     header = ['word','frequency']
    #     writer.writerow(header)
    #     for i in count_list:
    #         writer.writerow(i)

with open('Cluster_Analysis.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['Cluster','Avg_Score']
    writer.writerow(header)
    for line in output:
        writer.writerow(line)

