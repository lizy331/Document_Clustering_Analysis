import jieba as jieba
import matplotlib
import wordcloud as wordcloud
import matplotlib.pyplot as plt  #绘制图像的模块
import jieba #jieba分词
import os
import tqdm
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 设置停用词
stopwords = set(STOPWORDS)
print(stopwords)
# stopwords.add('AND','THE','SEX','TO','VICTIM','NARRATIVE','CASE',"STATE",'STATES','CRIME','SUSPECT','STATED','THIS','AT','HER','FOR','ON','THAT','WITH')
# stopwords.add('AND')
# stopwords.add('THE')
# stopwords.add('SEX')
# stopwords.add('VICTIM')
# stopwords.add('TO')
# stopwords.add('NARRATIVE')
# stopwords.add('CASE')
# stopwords.add('SUSPECT')
# stopwords.add('CRIME')
# stopwords.add('STATE')

cluster_content = '/Users/liziyang/Downloads/CaseWestern-master/cluster_content'
content_list = os.listdir(cluster_content)
# print(content_list)

for elem in content_list:
    complete_name = os.path.join(cluster_content,elem)
    cluster_name = elem.replace('.csv','')
    print(cluster_name)
    f = open(complete_name,'r').read()
    f = f.lower()
    f.replace('and','')
    f.replace('the','')
    f.replace('victim','')
    f.replace('the suspect','')
    f.replace('the victim','')
    f.replace('sex crime','')
    f.replace('victim states','')

    wordcloud = WordCloud(background_color="white",width=1000, height=860, margin=2,stopwords=stopwords).generate(f)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(cluster_name)
    plt.show()
    wordcloud.to_file('test.png')
# 保存图片,但是在第三模块的例子中 图片大小将会按照 mask 保存
