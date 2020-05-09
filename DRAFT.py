# import datetime
# import re
# df = ['19', '69']
# print([datetime.datetime.strptime(x,'%y').strftime('%Y') for x in df])
#
# ID = 'RMS09'
# date = re.search('\d{2}',ID)
# print(date.group())

# import matplotlib.pylab as plt
#
# plt.boxplot([[1,21,36,49,55,60,37,18,100,0.99],[90,89,170,130,98,77,200,16]])
# plt.show()

import io
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# my_stopwords = ['the','when','mater','id','rms','she','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
#     'q','r','s','t','u','v','w','x','y','z','us','on','one','two','three','four','five','six','seven','eight','nine',
#     'go','get','got','take','token','st']
# stopwords = stopwords.words('english')
# stopwords.extend(my_stopwords)
# print(stopwords)
#
input_file = r'/Users/liziyang/Downloads/NJIT_Sandbox-selected/NARRATIVES_BATCH_1/M504-2972_RMS03-118961.txt'
with open(input_file,'r',encoding='utf-8') as f:
    content = f.read()
#     content_split = content.split()
#     words_token = word_tokenize(content)
#     words_alph = [word.lower() for word in words_token if word.isalpha()]
#     words = [word for word in words_alph if word not in stopwords]
#
# def wordcount(words_list):
#     count_dic = {}
#     words_count = []
#     for word in words_list:
#         if word not in count_dic:
#             count_dic[word] = 1
#         else:
#             count_dic[word] += 1
#     for key in count_dic:
#         words_count.append([key,count_dic.get(key)])
#     return words_count
#
# word_count = wordcount(words)
#
# print("WORDS:",len(words))
# print("WORDS:",words)
# print(word_count)

#
# print("WORDS_alph:",len(words_alph))
# print("WORDS_alph:",words_alph)

# word_token = []
# # removing stopwords
# for word in words:
#     if word not in stopwords:


# my_list = [1, 2, 3, 4, 5,
#            6, 7, 8, 9]
#
# # How many elements each
# # list should have
# n = 4
#
# # using list comprehension
# final = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
# print(final)


# # 将文章分为段落
# def ParagraphSplit(report):
#     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     sentences = tokenizer.tokenize(report)
#     n = 5    #将文章大致分为 n 段
#     sentences_group = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n)]
#     paragraph = [' '.join(sentences_group[i]) for i in range(len(sentences_group))]
#     print(len(sentences_group))
#     print(len(paragraph))
#     return paragraph #输出自然段list
#
# par = ParagraphSplit(content)
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
        cleaned_text.append(s.stem(without_stopwords[i]))    #提取词干                 #实现计数功能 对每个词进行计数 并返回一个dict
    return cleaned_text

my_words = ['knew','friends','family','boyfriends','partner','know','father','unnamed','rape/named']
for word in my_words:
    print(s.stem(word))

string = 'Rape/Name Suspect'
l_text = string.lower()
tokens = nltk.word_tokenize(l_text)

print(tokens)