import os
import csv
import nltk.stem
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 输入police_report文件夹路径
data = pd.read_json('train_sak_geo.json')                           # 读取报告数据 得到 pandas dataframe

# 报告类别文件夹位置
cluster_dir = r'Cluster_csv_data'
cluster_list = os.listdir(cluster_dir)

stopwords_file = './stopwords.txt'                                  # 本地stopwords可根据情况修改
default_stopwords = set(nltk.corpus.stopwords.words('english'))     # nltk内置stopwords
with open(stopwords_file, 'r', encoding='utf-8') as f:              # 打开本地stopwords
    custom_stopwords = set(f.read().splitlines())                   # 将每行内容整合成list
all_stopwords = default_stopwords | custom_stopwords                # 将stopwords合并

### 定义工作流
# SVC分类器
SVC_pipeline = Pipeline([                                                           # 设置pipeline 类似于workflow将多步骤运算整合
    ('tfidf', TfidfVectorizer(stop_words=all_stopwords,                             # 输入多篇文章 根据tfidf返回一个word list
                              ngram_range=(1, 2),                                   # ngram 设定元单词 （1,2) 为一元和二元词汇 (1,1) 只显示为一元词汇 （2，2）只显示为二元词汇
                              token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b'         # 去除标点
                              )
     ),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),                # 将每个class视为一类 其他所有classes视为另一类 组成binary classification
])

# 贝叶斯分类器
NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=all_stopwords, ngram_range=(1, 2))),
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),
])

# 逻辑分类器
LogReg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=all_stopwords)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
])

# 合并报告聚类信息
def merge_cluster(train_data,cluster_list):
    cluster_label = {}                                                  # 合并标签
    for i in range(len(cluster_list)):                                  # 每个报告聚类为循环
        zeros = [0] * int(train_data.shape[0])                          # 记录binary label, 单个聚类为'1' 其他聚类为'0'
        TargetRMS = []                                                  # 记录RMS搜寻报告文件
        complete_name = os.path.join(cluster_dir,cluster_list[i])       # 文件夹位置与文件位置结合
        cluster_name = cluster_list[i].replace('.csv','')               # 得到聚类名称
        with open(complete_name, 'r', encoding='utf-8') as csvf:        # 打开报告的同类csv文件
            cluster_csv = csv.reader(csvf)                              # 读取报告聚类文件
            for line in cluster_csv:                                    # 聚类文件每行内容为 materID，RMS，labels
                if line[2] == '1':                                      # '1' 类报告为一类 默认其他报告为 '0'
                    TargetRMS.append(line[1])                           # 储存RMS用以搜寻属于此类的报告
        for h in range(len(train_data['rms'])):                         # 循环每篇报告
            if train_data['rms'][h] in TargetRMS:                       # 检查报告是否属于此聚类
                zeros[h] = 1                                            # 聚类标签化'1'
        cluster_label[cluster_name] = zeros                             # 记录每篇报告在所有聚类标签
    df = pd.DataFrame(cluster_label)
    return df

def word_top10(vectorizer, clf, class_labels):                          # 返回分类所依据的前十位单词
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()                      # get_feature_names 返回文章出现的所有单词
    for i, class_label in enumerate(class_labels):
        if i == class_labels.shape[0]-1:
            break
        top10 = np.argsort(clf.coef_[i])[-10:]                          # coef_ 从小到大排列 所以top10中单词重要性从左往右增加
        print('Top 10 words:',",".join(feature_names[j] for j in top10))
    return ",".join(feature_names[j] for j in top10)

def write_out(df,train_data,model):                                     # 选择使用的pipeline进行分类
    document_list = np.array(list(train_data['document']))              # 提取报告内容
    write_output = []                                                   # 记录 [聚类名，前十位词汇，精确度]
    for col in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(document_list,list(df[col]),random_state=0, test_size=0.1, shuffle=True)            # 将数据随机分配成 训练集与测试集
        model.fit(X_train, y_train)                              # 输入测试集数据和标签 （数据为报告内容 标签为当前聚类'1' 其他聚类'0'）
        prediction = model.predict(X_test)                       # 预测模型
        print()
        print(col + ': ' + str(accuracy_score(y_test, prediction)))      # 打印 OVR 准确值
        write_output.append([col,word_top10(model.steps[0][1], model.steps[1][1], model.classes_),str(accuracy_score(y_test, prediction))])        # 记录 [聚类名，前十位词汇，精确度]
    return write_output



if __name__ == '__main__':
    print('program started')
    new_df = merge_cluster(data,cluster_list)
    write_output = write_out(new_df,data,SVC_pipeline)
    with open('ClassifierBigram_SVC.csv','w',encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Cluster Name','Top10_words','accuracy']
        writer.writerow(header)
        for line in write_output:
            writer.writerow(line)
    print('program ended')


