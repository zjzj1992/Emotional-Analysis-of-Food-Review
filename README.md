# Emotional-Analysis-of-Food-Review
#这是一份关于各种美食的评论数据，一共有数据60多万条，而我分析的目的有两个：
#1、对评论是正面还是负面进行预测。
#2、对负面评论进行分析，找出顾客的差评原因，进而可以对其进行改善，从而促进销量。
#原始数据都是文本数据，所以如果想要利用机器学习模型进行分析的话，那么数据必须转换为数值类型的数据，否则模型是无法利用原始数据的，转换过程主要使用了CountVectorizer、Tfidf、N-grams、Word2Vec 、WordNetLemmatizer 、Snowball这些方式，另外还使用了一些其他的基础工具，比如Numpy、Pandas、Matplotlib、Seaborn、Scikit-Learn等
#还在改进中
