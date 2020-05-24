import json
import os
import csv
import pandas as pd
from tqdm import tqdm
from my_wordcount import words_cleaned_count

# 报告文件夹位置
report_path = 'train_sak_geo.json'
with open(report_path,'r',encoding='utf-8') as jsonf:
    reader = jsonf.read()
report = json.loads(reader)      # 载入json数据

# print('report length:',len(report))          # 检查报告篇数
# print('report type:',type(report))         # 检查数据结构
# for i in range(len(report)):
#     print(report[i]['document'])         # 载入报告内容

# 报告类别文件夹位置
cluster_dir = r'Cluster_csv_data'
input_list = os.listdir(cluster_dir)      # 聚类文件列 每行内容格式需统一为 [MasterID, RMS, '1'(默认其他报告类型为 '0')]


# 对同类报告内容词频统计
def cluster_wordcount(cluster_list):
    count_dict = {}     # 记录聚类词频计数结果
    print('start counting word frequency')
    for elem in tqdm(cluster_list):
        TargetRMS = []  # 记录RMS搜寻报告文件
        cluster_content = ''    # 合并同类报告内容 以统一词频计数
        complete_name = os.path.join(cluster_dir,elem)      # 文件夹位置与文件位置结合
        with open(complete_name, 'r', encoding='utf-8') as csvf:    # 打开报告聚类csv文件
            cluster = csv.reader(csvf)
            for line in cluster:    # 每行内容为 [MasterID, RMS, Cluster '1' 默认其他报告类型为 '0']
                if line[2] == '1':
                    TargetRMS.append(line[1])       # 记录'1'类报告
        for i in range(len(report)):        # 在json库中寻找属于此聚类的报告
            if report[i]['rms'] in TargetRMS:       # 使用rms匹配报告
                cluster_content += report[i]['document']        # 合并同类报告
        counter = words_cleaned_count(cluster_content)          # 对同类报告计数
        cluster_name = elem.replace('.csv','')              # 提取聚类名称
        count_dict[cluster_name] = counter              # 以聚类名称为key，计数结果为value 储存词频计数结果
    return count_dict

# 记录所有单词
def record_word(dict):
    word_list = []
    print('start collecting different words')
    for key in tqdm(dict):
        for item in dict.get(key):      # dict.get(key) 是一个聚类counter， item是一个单词
            # print(item,dict.get(key)[item])     # dict.get(key)[item] 得到单词数量
            if item not in word_list:       # 记录不同单词
                word_list.append(item)
    print('all different words:', len(word_list))  # 检查单词种数
    return word_list

# 创建dataframe 储存数据
def word_df(word_list):
    words_dict = {}     # 创建哈希表记录每个单词在不同聚类数量
    print('start collecting word frequency in different cluster')
    for word in tqdm(word_list):
        words_dict[word] = []        # 为每个单词创建空list以记录单词数量
        for key in dict:
            if word in dict.get(key):       # 检查此类报告是否包含单词
                words_dict[word].append(dict.get(key)[word])        # 记录此单词数量
            else:
                words_dict[word].append(0)      # 若不包含即为0
    return words_dict


if __name__ == "__main__":
    print('program started')
    dict = cluster_wordcount(input_list)        # 聚类报告词频计数
    word_list = record_word(dict)               # 收集不同单词
    words_dict = word_df(word_list)             # 创建dataframe储存数据
    print('Clusters:', len(dict))       # 检查聚类数量
    print('Word_in_clusters', len(words_dict['victim']))      # 检查每个单词所属聚类数量 应与以上结果相同
    word_df = pd.DataFrame(words_dict,index=[item for item in dict.keys()])         # 将每类报告的单词计数以dataframe形式记录
    print(word_df)          # 展示dataframe
    word_df.to_csv('wordcount.csv')         # 写出dataframe
    print('program ended')