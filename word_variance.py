import pandas as pd
import  csv
from statistics import variance,median


cluster_word = pd.read_csv('wordcount.csv')     # 载入聚类单词
df = cluster_word.drop(['Unnamed: 0'],axis=1)       # 去除聚类名称列

def IQR_Range(df):      # 将输入的dataframe每列求和 以1.5倍IQR方法筛选outliers
    cluster_word_sum = {}             # 以哈希表形式记录每列求和结果
    for col in df.columns:
        cluster_word_sum[col] = sum(df[col])        # 每列求和
    word_sum_sorted = sorted(cluster_word_sum.items(), key = lambda x: x[1])   # 按升序排列求和结果
    word_sum_list = [item[1] for item in word_sum_sorted]       # 只提取数值
    Q1 = median(word_sum_list[0:(int(len(word_sum_list)/2))])       # Q1 前半部分中位数
    Q3 = median(word_sum_list[(int(len(word_sum_list)/2)):int(len(word_sum_list))])     # Q3 后半部分中位数
    IQR = Q3-Q1         # IQR
    IQR_range = [Q1-1.5*IQR,Q3+1.5*IQR]         # 计算1.5倍IQR范围 超出此范围被视为outliers
    drop_col = []           # 记录outlier列
    for col in df.columns:
        if sum(df[col]) not in range(int(IQR_range[0]),int(IQR_range[1])):     # 超出1.5IQR范围被视为outliers
            drop_col.append(str(col))
    new_df = df.drop(drop_col, axis=1)      # 同时drop多列 （axis=1）
    return new_df


# 计算variance
def word_variance(df):
    word_variance = {}      # 记录每列方差结果
    for col in df.columns:              # 以每列为循环
        word_variance[col] = variance(df[col])      # 使用statistics 的variance计算方差
    word_variance_sorted = sorted(word_variance.items(), key = lambda x: x[1], reverse=True)   # 记录所有词方差排序结果
    # print(word_variance_sorted)
    return word_variance_sorted[0:99]       # 返回前100位方差较大的单词


if __name__ == '__main__':
    print('program started')
    new_df = IQR_Range(df)
    variance_list = word_variance(new_df)
    with open('word_variance.csv','w',encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['word','variance']
        writer.writerow(header)
        for item in variance_list:
            writer.writerow([item[0],item[1]])
    print('program ended')