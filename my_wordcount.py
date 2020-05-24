import string
import nltk
from collections import Counter
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
punctuation_map = dict((ord(char), None) for char in string.punctuation)   #生成标点符号

def words_cleaned_count(text):       # 词频统计
    l_text = text.lower()     #全部转化为小写以方便处理
    without_punctuation = l_text.translate(punctuation_map)    #去除文章标点符号
    tokens = nltk.word_tokenize(without_punctuation)        #将文章进行分词处理,将一段话转变成一个list
    words_alph = [word.lower() for word in tokens if word.isalpha()]       #去除数字
    without_stopwords = [w for w in words_alph if not w in stopwords]    #去除文章的停用词
    count = Counter(without_stopwords)                 #实现计数功能 对每个词进行计数 并返回一个dict
    return count

