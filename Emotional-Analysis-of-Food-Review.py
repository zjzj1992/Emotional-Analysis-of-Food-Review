#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
import nltk
from nltk import ngrams
from itertools import chain
from wordcloud import WordCloud


# In[33]:


data = pd.read_csv('D:/amazon-fine-food-reviews/Reviews.csv')


# In[5]:


data.head()


# 根据HelpfulnessNumerator和HelpfulnessDenominator计算好评率

# In[34]:


data['helpful percentage'] = np.where(data['HelpfulnessDenominator'] > 0,data['HelpfulnessNumerator'] / data['HelpfulnessDenominator'],-1)


# In[8]:


data.head()


# In[35]:


data['upvote'] = pd.cut(data['helpful percentage'],bins=[-1,0,0.2,0.4,0.6,0.8,1.0],labels=['Empty','0-20%','20%-40%','40%-60%','60%-80%','80%-100%'],include_lowest=True)


# 上面计算出了评论的好评率，然后根据这个特征使用分位数的方式进行分箱，对数据进行了离散化的处理，下面进行可视化的时候会需要这个特征

# In[10]:


data.head()


# In[8]:


data_1 = data.groupby(['Score','upvote']).agg({'Id':'count'})


# In[9]:


data_1 = data_1.unstack()


# In[11]:


data_1.columns = data_1.columns.get_level_values(1)


# In[13]:


fig = plt.figure(figsize=(15,10))

sns.heatmap(data_1[data_1.columns[::-1]].T, cmap = 'YlGnBu', linewidths=.5, annot = True, fmt = 'd', cbar_kws={'label': '# reviews'})
plt.yticks(rotation=0)
plt.title('How helpful users find among user scores')


# 这是一个热力图，从这个可视化的结果可以看出：
# #评论偏向于正向
# #有很多零票
# #超过半数的人给了5分

# 下面先看一下数据中有没有缺失的情况

# In[5]:


total = data.isnull().sum().sort_values(ascending=False)
percent = round(data.isnull().sum().sort_values(ascending=False)/len(data)*100,4)
pd.concat([total,percent],axis=1,keys=['total','percent'])


# 数据中的缺失数据很少，可以不做处理

# In[7]:


data.describe()


# 下面对数据进行清洗，去除掉重复数据

# In[36]:


data[data.duplicated(subset={'UserId','ProfileName','Time','Text','Score'})].count()


# In[37]:


data = data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'})


# 下面处理目标变量

# 所有评论的得分在1-5之间，1、2为负面评论；3为中性评论；4、5为正面评论，所以，为了训练一个可以区分正面还是负面的评论模型，下面我打算将评分为3的中性评论剔除掉，将训练重点只关注到正面和负面的评论上

# 还有一点需要住，就是关于预测的目标变量的问题，正面评论的分数是4或5，而负面的评论是1或2，但是如果用这个分数进行预测的话，那么预测的结论是不准确的，因为，分数这个目标变量会受到很多因素的影响，会导致模型预测的复杂度很高，所以一般有两种方式处理这种问题：
# #二值化的方式来处理，用1代表正面评论，用0代表负面评论
# #分箱的方式处理，如果不能用二值来处理的话，那么可以使用这种形式

# 这里可以采用二值化的方式处理目标变量

# In[38]:


data_score = data[data['Score'] != 3]
X = data_score['Text']
y_mapping = {1:0,2:0,4:1,5:1}
y = data_score['Score'].map(y_mapping)


# 目前所获取的数据都是非数值类型的数据，这种数据是不能直接应用到模型中去的，所以必须进行特征工程，将文本数据转化为数值的形式才可以，而比较常用的方式有：
# #CountVectorizer:该方式是一种比较简单的方式，是只考虑每个单词的计数，可以将文本文档转化为文档向量的形式，是一种扁平向量。这种方式虽然简单，但是在处理分类任务和判断文档的相似性时这种方式已经可以有不错的表现了
# #Tfidf：这种方式的特点是考虑了词频和逆文档频率，是CountVectorizer的一种简单的扩展。因为有时候，只用词汇的计数是很难凸显出数据的主要特点的，所以Tfidf可以更好的强调那些有意义的单词，这种方式会凸显出罕见词，并忽略常见词
# #ngram：这种方式加入了联想的功能，意思就是，每个词的出现都跟前一个或前若干个词有关，之前的技术都是不考虑语序的，但是这种技术考虑了词之间的顺序，在理解语义上要更加强大

# 下面先用CountVectorizer来处理文本数据

# In[11]:


c = CountVectorizer(stop_words='english')
lr_c = LogisticRegression()


# In[12]:


X_c = c.fit_transform(X)
print("feature:{}".format(X_c.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_c,y,random_state=0)
clf = lr_c.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# 通过逻辑回归这种简单的模型预测的结果还不错，有92.5%的准确率

# 但是这里有一点需要注意，原始样本是不平衡的，所以如果仅采用准确率，那么这种评估并不会很准确，但是我在这里并没有对样本进行处理，而是使用了一种新的评估标准

# MCC也是一种准确率的评估指标，但是这种指标对不平衡数据是比较友好的，所以哪怕数据是不平衡的，这个评估结果也是更具参考性的

# In[19]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# MCC的结果为0.7，而这种评估指标的取值范围在[-1,1]之间，越接近1代表越好，而这个结果为0.7，所以也能证明，这个模型的预测准确率还是不错的

# 下面打算展示一下前20的正向词汇和负向词汇，看看有什么特点

# In[15]:


def show_p_n_top20(model,clf_model):
    words = model.get_feature_names()
    coef = clf_model.coef_.tolist()[0]
    coef_data = pd.DataFrame({'Word':words,'Coef':coef}).sort_values(['Coef','Word'],ascending=[0,1])
    print('Top 20 positive')
    print(coef_data.head(20).to_string(index=False))
    print("")
    print("Top 20 negative")
    print(coef_data.tail(20).to_string(index=False))


# In[13]:


show_p_n_top20(c,clf)


# 从上面的结果可以看出，有一些词汇是没有意义的，并且词汇对应的系数太小，没有凸显出对结果的预测能力，需要换一种技术

# 下面用Tfidf来处理文本数据

# In[21]:


tfidf = TfidfVectorizer(stop_words='english')
lr_tfidf = LogisticRegression()


# In[22]:


X_tfidf = tfidf.fit_transform(X)
print("feature:{}".format(X_tfidf.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf,y,random_state=0)
clf = lr_tfidf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[16]:


show_p_n_top20(tfidf,clf)


# In[23]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 从上面的结果可以看出来，准确率是没有什么变化的，但是筛选出来的词汇变得更有意义了，每个词汇对应的系数都提高了，说明Tfidf模型的效果要更好一些

# 下面使用带有n-grams的技术尝试效果

# In[24]:


tfidf_ngram = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
lr_tfidf_ngram = LogisticRegression()


# In[25]:


X_tfidf_ngram = tfidf_ngram.fit_transform(X)
print("feature:{}".format(X_tfidf_ngram.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_ngram,y,random_state=0)
clf = lr_tfidf_ngram.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[19]:


show_p_n_top20(tfidf_ngram,clf)


# In[26]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 从上面的结果可以看出来，在使用了n-grams之后，模型预测的准确率没有什么变化，但是每个词汇对应的系数都提升了，证明这个模型可以找到更关键的词汇

# 下面更改了n-gram中N的值来看看有什么效果

# 之前n-grams使用的是（1，2），模型的准确率没有什么变化，但是词汇对应的系数都提高了，证明选择出来的词汇更有意义，对结果的影响力更大。现在尝试使用其他的参数值来看看有什么效果

# 尝试一下n-grams=(1,3)的情况

# In[27]:


tfidf_ngram_1_3 = TfidfVectorizer(ngram_range=(1,3),stop_words='english')
lr_tfidf_1_3 = LogisticRegression()


# In[28]:


X_tfidf_ngram_1_3 = tfidf_ngram_1_3.fit_transform(X)
print("feature:{}".format(X_tfidf_ngram_1_3.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_ngram_1_3,y,random_state=0)
clf = lr_tfidf_1_3.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[14]:


show_p_n_top20(tfidf_ngram_1_3,clf)


# In[29]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 模型的预测准确率有了比较明显的下降，对于词汇的系数有了一点提升，但是提升效果已经不明显了

# In[8]:


tfidf_ngram_1_4 = TfidfVectorizer(ngram_range=(1,4),stop_words='english')
lr_tfidf_ngram_1_4 = LogisticRegression()


# In[9]:


X_tfidf_ngram_1_4 = tfidf_ngram_1_4.fit_transform(X)
print("feature:{}".format(X_tfidf_ngram_1_4.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_ngram_1_4,y,random_state=0)
clf = lr_tfidf_ngram_1_4.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracty:{}".format(acc))


# In[18]:


show_p_n_top20(tfidf_ngram_1_4,clf)


# In[10]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 可以看出来，对于n-gram模型中的N来说，当N越大时，可以提取出来的信息量就越多，但是随着N的增加，数据的信息量增长的速度放缓，所以从这个实验可以看出一味的提升N并不一定可以提升模型的性能。最常用的N一般是2或3，从实际来看当N等于2时，预测的准确率是最高的

# 之前对文本数据的预处理阶段只是做了去重的操作，下面想对文本做一些其他的处理，比如去掉一些额外的符号、转化小写、去除停用词等等操作，然后再看看这个文本数据的表现

# In[8]:


'''
data_3 = data[data['Score'] != 3]
y_mapping = {'0-20%':0,'20%-40%':0,'60%-80%':1,'80%-100%':1}
data_3['upvote'] = data_3['upvote'].map(y_mapping)
downvote_word2vec_data = len(data_3[data_3['upvote'] == 0])
downvote_word2vec_indices = np.array(data_3[data_3['upvote'] == 0].index)
upvote_word2vec_indices = data_3[data_3['upvote'] == 1].index
random_upvote_indices = np.random.choice(upvote_word2vec_indices,downvote_word2vec_data,replace=False)
random_upvote_indices = np.array(random_upvote_indices)

#combine
sample_word2vec_data = np.concatenate([downvote_word2vec_indices,random_upvote_indices])
sample_word2vec_data = data_3.ix[sample_word2vec_data,:]
X_sample_word2vec_data = sample_word2vec_data['Text']
sample_word2vec_data['upvote'] = sample_word2vec_data['upvote'].astype(int)
y_sample_word2vec_data = sample_word2vec_data['upvote']

#Display data status
print("upvote percentage:",len(sample_word2vec_data[sample_word2vec_data['upvote'] == 1])/len(sample_word2vec_data))
print("downvote pencentage:",len(sample_word2vec_data[sample_word2vec_data['upvote'] == 0])/len(sample_word2vec_data))
print("record:",len(sample_word2vec_data))
'''


# In[22]:


stop_words = set(stopwords.words('english'))


# In[23]:


import re
words = []
snow = nltk.stem.SnowballStemmer('english')
for sentence in X:
    #转换为小写
    sentence = sentence.lower()
    r = re.compile('<.*?>')
    sentence = re.sub(r,' ',sentence)
    sentence = re.sub(r'[?|!|\|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)
    temp = [snow.stem(word) for word in sentence.split() if word not in stop_words]
    words.append(temp)
modify_data = words  


# In[24]:


sentence = []
for row in modify_data:
    seq = ''
    for word in row:
        seq  = seq + ' ' + word
    sentence.append(seq)
modify_data = sentence


# In[25]:


modify_data_ngram_2 = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
modify_lr = LogisticRegression()


# In[26]:


modify_data_ngram_2 = modify_data_ngram_2.fit_transform(modify_data)
print("feature:{}".format(modify_data_ngram_2.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(modify_data_ngram_2,y,random_state=0)
clf = modify_lr.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracty:{}".format(acc))


# In[27]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 上面的处理并没有提升模型的表现

# In[ ]:


'''w2v_data = modify_data

split = []
for row in w2v_data:
    split.append([word for word in row.split()])
    
w2v_data = Word2Vec(split,min_count=5,workers=2)
'''


# 下面我打算找出成绩为5的数据，并基于此数据找出不满意的评论，并根据这个促进销售

# In[39]:


data_score_5 = data[data['Score'] == 5]
data_score_5 = data_score_5[data_score_5['upvote'].isin(['0-20%','20%-40%','60%-80%','80%-100%'])]

X = data_score_5['Text']
y_dict = {'0-20%':0,'20%-40%':0,'60%-80%':1,'80%-100%':1}
y = data_score_5['upvote'].map(y_dict)

print("Classs:")
print(y.value_counts())


# 从上面的结果可以很明显的看出，正面和负面的评论数量有着严重的不平衡问题，需要先解决这个问题，需要对样本进行重新采样，以获得平衡数据

# 对于不平衡数据来说，比较常用的处理方式可以分为：
# #欠采样：通过减少多数类别的样本数量，随机的从多数类样本中挑选与少数类数目一样的样本，然后与少数类组合起来构成平衡样本
# #过采样：使用随机的方式来增加稀疏样本的数量，随机的从少数类的样本中抽取样本，产生的样本数量要与多数类的样本数量保持一致，然后在组合起来

# 下面我打算两种方式都尝试一下

# 欠采样

# In[40]:


data_5 = pd.DataFrame([X,y]).T


# In[41]:


downvote = len(data_5[data_5['upvote'] == 0])
downvote_indices = np.array(data_5[data_5['upvote'] == 0].index)
upvote_indices = data_5[data_5['upvote'] == 1].index
random_upvote_indices = np.random.choice(upvote_indices,downvote,replace=False)
random_upvote_indices = np.array(random_upvote_indices)

#combine
undersampling_sample_data = np.concatenate([downvote_indices,random_upvote_indices])
undersampling_sample_data = data_5.ix[undersampling_sample_data,:]
X_sample_data = undersampling_sample_data['Text']
undersampling_sample_data['upvote'] = undersampling_sample_data['upvote'].astype(int)
y_sample_data = undersampling_sample_data['upvote']

#Display data status
print("upvote percentage:",len(undersampling_sample_data[undersampling_sample_data['upvote'] == 1])/len(undersampling_sample_data))
print("downvote pencentage:",len(undersampling_sample_data[undersampling_sample_data['upvote'] == 0])/len(undersampling_sample_data))
print("record:",len(undersampling_sample_data))


# In[42]:


c = CountVectorizer(stop_words='english')
lr_c = LogisticRegression()


# In[43]:


X_c = c.fit_transform(X_sample_data)
print("feature:{}".format(X_c.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_c,y_sample_data,random_state=0)
clf = lr_c.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[16]:


show_p_n_top20(c,clf)


# In[44]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 模型准确率为60.8%，模型预测的结果很差，而且top20的词汇都是一些没有意义的，且系数很小，说明这个模型并不好，下面换一个tfidf+ngram的模型

# In[45]:


tfidf_ngram = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
lr_tfidf_ngram = LogisticRegression()


# In[46]:


X_tfidf_ngram = tfidf_ngram.fit_transform(X_sample_data)
print("feature:{}".format(X_tfidf_ngram.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_ngram,y_sample_data,random_state=0)
clf = lr_tfidf_ngram.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[19]:


show_p_n_top20(tfidf_ngram,clf)


# In[47]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 结果的预测准确率会有所改善，为61.4%，但是依然很差，并且词汇仍然都是一些没有意义的

# 下面尝试用过采样的方式

# In[48]:


downvote = len(data_5[data_5['upvote'] == 0])
downvote_indices = np.array(data_5[data_5['upvote'] == 0].index)
upvote_indices = data_5[data_5['upvote'] == 1].index
upvote = len(data_5[data_5['upvote'] == 1])
random_downvote_indices = np.random.choice(downvote_indices,upvote,replace=True)
random_downvote_indices = np.array(random_downvote_indices)

#combine
oversampling_sample_data = np.concatenate([random_downvote_indices,upvote_indices])
oversampling_sample_data = data_5.ix[oversampling_sample_data,:]
X_sample_data = oversampling_sample_data['Text']
oversampling_sample_data['upvote'] = oversampling_sample_data['upvote'].astype(int)
y_sample_data = oversampling_sample_data['upvote']

#Display data status
print("upvote percentage:",len(oversampling_sample_data[oversampling_sample_data['upvote'] == 1])/len(oversampling_sample_data))
print("downvote pencentage:",len(oversampling_sample_data[oversampling_sample_data['upvote'] == 0])/len(oversampling_sample_data))
print("record:",len(oversampling_sample_data))


# 下面使用tfidf+ngram的模式来看看效果

# In[49]:


tfidf_ngram_2 = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
lr_tfidf_ngram_2 = LogisticRegression()


# In[50]:


X_tfidf_ngram_2 = tfidf_ngram_2.fit_transform(X_sample_data)
print("feature:{}".format(X_tfidf_ngram_2.shape[1]))
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_ngram_2,y_sample_data,random_state=0)
clf = lr_tfidf_ngram_2.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# In[23]:


show_p_n_top20(tfidf_ngram_2,clf)


# In[51]:


y_pred = clf.predict(X_test)
metrics.matthews_corrcoef(y_test,y_pred)


# 从结果可以看出，这种过采样的方式可以带来99.7%的预测准确率，乍一看这个模型是很好的，但是之所以有这样的准确率我觉得主要还是因为随机样本的选择方式上，因为我使用了有放回的方式，那么很有可能会随机到很多重复的样本，那么这时候训练数据和测试数据中就一定会有重复的情况，所以预测的结果自然就很高，另外这种差异也与过采样可以带来比欠采样要多的多的数据有一定关系，因为数据量的增加是可以提升模型性能的。所以，这个模型的准确率其实并没有什么可参考性，对于未知数据的泛化能力不会很高

# 再来看词汇对应的系数，也不是很大，所以词汇的意义还是比较小，预测的能力并不强

# 鉴于上面的模型表现不佳，没有找出什么有意义的词汇，也就是基于现有的特征是无法有更好的表现了，所以下面我打算尝试构造一些新的特征，看看能不能带来更好的表现

# 下面使用欠采样的数据

# In[52]:


#每条评论的单词数量
undersampling_sample_data['words_count'] = undersampling_sample_data['Text'].apply(lambda x:len(x.split()))
#带有问好的评论
undersampling_sample_data['question'] = undersampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c == '?'))
#带有感叹号的评论
undersampling_sample_data['exclamation'] = undersampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c == '!'))
#带有大写字母的评论
undersampling_sample_data['capitial_count'] =undersampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c.isupper()))


# In[53]:


print(undersampling_sample_data.groupby('upvote').agg({'words_count':'mean','capitial_count':'mean','question':'mean','exclamation':'mean'}).T)


# 从结果可以看出，反对评论的词汇数量有77，而点赞评论的词汇数量有87；还有不管是什么评论，带有疑问的评论是很少的，但是带有感叹号的评论却很多

# In[54]:


#筛选出需要的数据
X_counts = undersampling_sample_data[undersampling_sample_data.columns.difference(['upvote','Text'])]
y_counts = undersampling_sample_data['upvote']


# In[55]:


X_train,X_test,y_train,y_test = train_test_split(X_counts,y_counts,random_state=0)
clf_lr = LogisticRegression()
clf_lr.fit(X_train,y_train)
acc = clf_lr.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# 下面使用过采样的数据

# In[56]:


#每条评论的单词数量
oversampling_sample_data['words_count'] = oversampling_sample_data['Text'].apply(lambda x:len(x.split()))
#带有问好的评论
oversampling_sample_data['question'] = oversampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c == '?'))
#带有感叹号的评论
oversampling_sample_data['exclamation'] = oversampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c == '!'))
#带有大写字母的评论
oversampling_sample_data['capitial_count'] =oversampling_sample_data['Text'].apply(lambda x:sum(1 for c in x if c.isupper()))


# In[57]:


print(oversampling_sample_data.groupby('upvote').agg({'words_count':'mean','capitial_count':'mean','question':'mean','exclamation':'mean'}).T)


# 从结果可以看出，反对评论的词汇数量有77，而点赞评论的词汇数量有85；还有不管是什么评论，带有疑问的评论是很少的，但是带有感叹号的评论却很多

# In[58]:


#筛选出需要的数据
X_counts = oversampling_sample_data[oversampling_sample_data.columns.difference(['upvote','Text'])]
y_counts = oversampling_sample_data['upvote']


# In[59]:


X_train,X_test,y_train,y_test = train_test_split(X_counts,y_counts,random_state=0)
clf_lr = LogisticRegression()
clf_lr.fit(X_train,y_train)
acc = clf_lr.score(X_test,y_test)
print("Model Accuracy:{}".format(acc))


# 但是说实话，这种构造出来的特征有点随便，是不会给预测带来什么好的提升的，这两种方式的预测准确率只有60%而已

# 需要思考更多，找出更有意义的特征

# In[60]:


data_user = data.groupby(['UserId','ProfileName']).agg({'Score':['count','mean']})
data_user.columns = data_user.columns.get_level_values(1)
data_user.columns = ['score count','score mean']


# In[61]:


data_user = data_user.sort_values(by = 'score count',ascending=False)


# In[62]:


print(data_user.head())


# 从结果可以看出，评论最多的用户是一个叫Gary Peterson的人，一共有329条，其平均评分为4.66

# In[40]:


def plot_score(UserId):
    data_user = data[data['UserId'] == UserId]['Score']
    data_user_plot = data_user.value_counts(sort=False)
    ax = data_user_plot.plot(kind='bar',figsize=(15,10),title="User {} score".format(data[data['UserId'] == UserId]['ProfileName'].iloc[0]))

plot_score('AY12DBB0U420B')


# 从上面的直方图中可以看出，该用户给了很多高分，1分的一个都没有，2分的只有几个，中立分3也很少，4和5的高分却给了很多，所以该用户非常容易给高分，那么这种用户不具备一般性，参考价值不大

# 下面我要筛选一下平均分在2.5到3.5之间的用户，这种用户分析起来会更方便

# In[51]:


print(data_user[(data_user['score mean'] < 3.5) & (data_user['score mean'] > 2.5)].head())


# 根据结果，我选择了出现次数最多，且在2.5到3.5之间的用户A. Bennett

# In[52]:


plot_score('A2M9D9BDHONV3Y')


# 从可视化的结果可以看出，这个用户更值得分析，下面进行一些具体的分析工作

# In[53]:


def get_token_words(score,mark,userid='all'):
    #判断是某个用户还是全部用户
    if userid != 'all':
        data_u_s = data[(data['UserId'] == userid) & (data['Score'] == score)]['Text']
    else:
        data_u_s = data[data['Score'] == score]['Text']
        
    count = len(data_u_s)
    text = ' '.join(data_u_s)
    #将文本转化为小写形式
    text = text.lower()
    #生成停用词列表
    stop_words = set(stopwords.words('english'))
    #建立分词
    words_list = nltk.word_tokenize(text)
    #筛选出长度大于3且不在停用词列表中的词汇
    words_list = [word for word in words_list if word not in stop_words and len(word) >= 3]
    #词性还原
    lemmatizer = WordNetLemmatizer()
    #动词
    words_list = [lemmatizer.lemmatize(w,'v') for w in words_list]
    #ngrams模型
    bigrams = ngrams(words_list,2)
    trigrams = ngrams(words_list,3)
    
    combine = chain(bigrams,trigrams)
    texts = nltk.Text(combine)
    #统计combine中相同元素的个数
    fdist = nltk.FreqDist(texts)
    
    return sorted([(w,fdist[w],str(round(fdist[w]/count*100,2))+'%') for w in set(texts) if fdist[w] >= count * mark],key=lambda x: -x[1])


# 该函数用于找出最常出现的词汇，可以用于分析数据

# 下面找出A. Bennett用户最常使用的词汇

# In[54]:


indexs = ['Phrase','Count','Occur']

#1-5的所有成绩
for i in range(1,6):
    test = pd.DataFrame()
    messages = get_token_words(i,0.25,'A2M9D9BDHONV3Y')
    print("Score {} most popular 2grams and 3grams:".format(i))
    for j in messages:
        test = test.append(pd.Series(j,index=indexs),ignore_index=True)
        test = test.sort_values('Count',ascending=False)
    print(test)


# 这个用户在低平分中总是喜欢用省略号这样的标记，而在高评分基本就只使用了“强烈推荐”这个词

# 下面找出所有评论的最常用词汇

# In[55]:


index = ['Phrase','Count','Occur']

for i in range(1,6):
    test = pd.DataFrame()
    messages = get_token_words(i,0.02)
    print("Score {} most popular 2grams and 3grams:".format(i))
    for j in messages:
        test = test.append(pd.Series(j,index=indexs),ignore_index=True)
        test = test.sort_values("Count",ascending=False)
    print(test)


# 与之前结果最明显的差异就是在3、4、5的评论中页面链接出现的频繁了，但是在评分1和评分2中没有什么其他的特别有意义的词汇出现

# 后来发现了一点，就是应该将重点放在形容词上，这样的结果应该会好

# In[56]:


def get_token_adj(score,mark,userid='all'):
    #判断是单个用户还是全部用户
    if userid != 'all':
        data_u_s = data[(data['UserId'] == userid) & (data['Score'] == score)]['Text']
    else:
        data_u_s = data[data['Score'] == score]['Text']
        
    count = len(data_u_s)
    text = ' '.join(data_u_s)
    #将文件转化为小写形式
    text = text.lower()
    #提炼出停用词列表
    stop_words = set(stopwords.words('english'))
    #分词
    words_list = nltk.word_tokenize(text)
    #筛选出长度大于3且不在停用词列表中的词汇
    words_list = [word for word in words_list if word not in stop_words and len(word) >= 3]
    #词性还原
    lemmatizer = WordNetLemmatizer()
    #形容词
    words_list = [lemmatizer.lemmatize(w,'a') for w in words_list]
    words_list = [word for word,form in nltk.pos_tag(words_list) if form == 'JJ']
    
    texts = nltk.Text(words_list)
    #统计combine中相同元素的个数
    fdist = nltk.FreqDist(texts)
    
    return sorted([(w,fdist[w],str(round(fdist[w]/count*100,2))+'%') for w in set(texts) if fdist[w] >= count * mark],key=lambda x: -x[1])


# In[64]:


indexs = ['Phrase','Count','Occur']

for i in range(1,6):
    test = pd.DataFrame()
    messages = get_token_adj(i,0.2,'A2M9D9BDHONV3Y')
    print("Score {} most popular adj words:".format(i))
    for j in messages:
        test = test.append(pd.Series(j,index=indexs),ignore_index=True)
        test = test.sort_values('Count',ascending=False)
    print(test)


# 那么最终推测的结论是：用户在吐槽的点是，食物量比较少、食物搭配不好，导致脂肪含量大、价格昂贵、食物口感偏硬等问题，那么找到问题后，就可以根据这些出现的问题来改善食物，进而促进销量

# 第二种尝试

# In[ ]:


data_score_1_2 = data[(data['Score'] == 1) & (data['Score'] == 2)]
data_score_1_2 = data_score_1_2[data_score_1_2['upvote'].isin(['0-20%','20%-40%','60%-80%','80%-100%'])]

X = data_score_1_2['Text']
y_dict = {'0-20%':0,'20%-40%':0,'60%-80%':1,'80%-100%':1}
y = data_score_1_2['upvote'].map(y_dict)

print("Classs:")
print(y.value_counts())


# In[ ]:





# In[ ]:




