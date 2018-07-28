---
layout: post
title: PyTorch 序列数据和文本的深度学习
date: 2018-07-29
categories: blog
tags: [PyTorch,序列数据和文本的深度学习]
description: PyTorch 序列数据和文本的深度学习
---
# 第六章 序列数据和文本的深度学习
在上一章中，我们介绍了如何使用卷积神经网络（CNN）处理空间数据以及如何构建图像分类器。 在本章中，我们将介绍以下主题：
* 对构建深度学习模型有用的文本数据的不同表示
* 了解递归神经网络（RNN）和RNN的不同实现，例如长期短期记忆（LSTM）和门控递归单位（GRU），它们为大多数基于文本和序列数据的深度学习模型提供支持
* 使用一维卷积来处理顺序数据

可以使用RNN构建的一些应用程序是：
* 文档分类器：识别推文或评论的情绪，对新闻文章进行分类
* 序列到序列学习：用于语言翻译，将英语转换为法语等任务
* 时间序列预测：根据前几天的商店销售的详细信息，预测商店的销售情况

## 使用文本数据

文本是常用的顺序数据类型之一。 文本数据可以看作是一系列字符或一系列单词。 对于大多数问题而言，将文本视为的单词序列是很常见的。 深度学习序列模型（如RNN及其变体）能够从文本数据中学习重要模式，这些模式可以解决以下领域中的问题：
* 自然语言理解
* 文献分类
* 情感分类

这些顺序模型还充当各种系统的重要构建块，例如问答（QA）系统。

虽然这些模型在构建这些应用程序时非常有用，但它们无法理解固有的复杂性的人类语言。 这些顺序模型能够成功找到有用的模式，然后用于执行不同的任务。 将深度学习应用于文本处理是一个快速发展的领域，每个月都会有很多新技术到来。 我们将介绍为大多数现代深度学习应用程序提供支持的基本组件。

与任何其他机器学习模型一样，深度学习模型不理解文本，因此我们需要将文本转换为数字表示。 将文本转换为数字表示的过程称为矢量化，可以通过不同的方式完成，如下所述：

* 将文本转换为单词并将每个单词表示为向量
* 将文本转换为字符并将每个字符表示为向量
* 创建n-gram单词并将其表示为向量

文本数据可以分解为这些表示之一。 每个较小的文本单元称为标记，将文本分解为标记的过程称为标记化。 Python中有许多强大的库可以帮助我们进行标记化。 一旦我们将文本数据转换为标记，我们就需要将每个标记映射到一个向量。 One-Hot编码和词嵌入是将标记映射到向量的两种最常用的方法。 下图总结了将文本转换为矢量表示的步骤：

![](https://bennix.github.io/imgs/6_1.png)

让我们看一下标记化，n-gram表示和向量化的更多细节。

## 标记化

给定一个句子，将其分为字符或单词称为标记化。 有一些库，例如spaCy，它们为标记化提供了复杂的解决方案。 让我们使用简单的Python函数（如split和list）将文本转换为标记。

为了演示标记化如何对字符和单词起作用，让我们考虑对电影Thor：Ragnarok的一个小评论。 我们将使用以下文本：

```
The action scenes were top notch in this movie. Thor has never been this epic in the MCU. He does some pretty epic sh*t in this movie and he is definitely not under-powered anymore. Thor in unleashed in this, I love that.
```

# 将文本转换为字符

Python列表函数接受一个字符串并将其转换为单个字符列表。 这样做可以将文本转换为字符。 以下代码块显示了使用的代码和结果：


```python
thor_review ="""the action scenes were top notch in this movie. Thor has never been this epic in the MCU. 
He does some pretty epic sh*t in this movie and he is definitely not under-powered anymore. Thor in unleashed 
in this, I love that."""

```


```python
print(list(thor_review))
```

    ['t', 'h', 'e', ' ', 'a', 'c', 't', 'i', 'o', 'n', ' ', 's', 'c', 'e', 'n', 'e', 's', ' ', 'w', 'e', 'r', 'e', ' ', 't', 'o', 'p', ' ', 'n', 'o', 't', 'c', 'h', ' ', 'i', 'n', ' ', 't', 'h', 'i', 's', ' ', 'm', 'o', 'v', 'i', 'e', '.', ' ', 'T', 'h', 'o', 'r', ' ', 'h', 'a', 's', ' ', 'n', 'e', 'v', 'e', 'r', ' ', 'b', 'e', 'e', 'n', ' ', 't', 'h', 'i', 's', ' ', 'e', 'p', 'i', 'c', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'M', 'C', 'U', '.', ' ', '\n', 'H', 'e', ' ', 'd', 'o', 'e', 's', ' ', 's', 'o', 'm', 'e', ' ', 'p', 'r', 'e', 't', 't', 'y', ' ', 'e', 'p', 'i', 'c', ' ', 's', 'h', '*', 't', ' ', 'i', 'n', ' ', 't', 'h', 'i', 's', ' ', 'm', 'o', 'v', 'i', 'e', ' ', 'a', 'n', 'd', ' ', 'h', 'e', ' ', 'i', 's', ' ', 'd', 'e', 'f', 'i', 'n', 'i', 't', 'e', 'l', 'y', ' ', 'n', 'o', 't', ' ', 'u', 'n', 'd', 'e', 'r', '-', 'p', 'o', 'w', 'e', 'r', 'e', 'd', ' ', 'a', 'n', 'y', 'm', 'o', 'r', 'e', '.', ' ', 'T', 'h', 'o', 'r', ' ', 'i', 'n', ' ', 'u', 'n', 'l', 'e', 'a', 's', 'h', 'e', 'd', ' ', '\n', 'i', 'n', ' ', 't', 'h', 'i', 's', ',', ' ', 'I', ' ', 'l', 'o', 'v', 'e', ' ', 't', 'h', 'a', 't', '.']


## 将文本转换为单词

我们将使用Python字符串对象中可用的split函数将文本分解为单词。 split函数接受一个参数，根据该参数将文本拆分为标记。 对于我们的示例，我们将使用空格作为分隔符。 以下代码块演示了如何使用Python split函数将文本转换为单词：


```python
print(thor_review.split())

```

    ['the', 'action', 'scenes', 'were', 'top', 'notch', 'in', 'this', 'movie.', 'Thor', 'has', 'never', 'been', 'this', 'epic', 'in', 'the', 'MCU.', 'He', 'does', 'some', 'pretty', 'epic', 'sh*t', 'in', 'this', 'movie', 'and', 'he', 'is', 'definitely', 'not', 'under-powered', 'anymore.', 'Thor', 'in', 'unleashed', 'in', 'this,', 'I', 'love', 'that.']


在前面的代码中，我们没有使用任何分隔符; 默认情况下，split函数在空格上分割。

## N-gram表示

我们已经看到文本如何表示为字符和单词。 有时一起查看两个，三个或更多单词是有用的。 N-gram是从给定文本中提取的单词组。 在n-gram中，n表示可以一起使用的单词数。 让我们看一下bigram（n = 2）的样子。 我们使用Python nltk包为thor_review生成一个bigram。 以下代码块显示了bigram的结果以及用于生成它的代码：


```python
from nltk import ngrams

print(list(ngrams(thor_review.split(),2)))

```

    [('the', 'action'), ('action', 'scenes'), ('scenes', 'were'), ('were', 'top'), ('top', 'notch'), ('notch', 'in'), ('in', 'this'), ('this', 'movie.'), ('movie.', 'Thor'), ('Thor', 'has'), ('has', 'never'), ('never', 'been'), ('been', 'this'), ('this', 'epic'), ('epic', 'in'), ('in', 'the'), ('the', 'MCU.'), ('MCU.', 'He'), ('He', 'does'), ('does', 'some'), ('some', 'pretty'), ('pretty', 'epic'), ('epic', 'sh*t'), ('sh*t', 'in'), ('in', 'this'), ('this', 'movie'), ('movie', 'and'), ('and', 'he'), ('he', 'is'), ('is', 'definitely'), ('definitely', 'not'), ('not', 'under-powered'), ('under-powered', 'anymore.'), ('anymore.', 'Thor'), ('Thor', 'in'), ('in', 'unleashed'), ('unleashed', 'in'), ('in', 'this,'), ('this,', 'I'), ('I', 'love'), ('love', 'that.')]


ngrams函数接受一个单词序列作为其第一个参数，并将要分组的单词数作为第二个参数。 以下代码块显示了三个词组成的元组以及用于它的代码：


```python
print(list(ngrams(thor_review.split(),3)))

```

    [('the', 'action', 'scenes'), ('action', 'scenes', 'were'), ('scenes', 'were', 'top'), ('were', 'top', 'notch'), ('top', 'notch', 'in'), ('notch', 'in', 'this'), ('in', 'this', 'movie.'), ('this', 'movie.', 'Thor'), ('movie.', 'Thor', 'has'), ('Thor', 'has', 'never'), ('has', 'never', 'been'), ('never', 'been', 'this'), ('been', 'this', 'epic'), ('this', 'epic', 'in'), ('epic', 'in', 'the'), ('in', 'the', 'MCU.'), ('the', 'MCU.', 'He'), ('MCU.', 'He', 'does'), ('He', 'does', 'some'), ('does', 'some', 'pretty'), ('some', 'pretty', 'epic'), ('pretty', 'epic', 'sh*t'), ('epic', 'sh*t', 'in'), ('sh*t', 'in', 'this'), ('in', 'this', 'movie'), ('this', 'movie', 'and'), ('movie', 'and', 'he'), ('and', 'he', 'is'), ('he', 'is', 'definitely'), ('is', 'definitely', 'not'), ('definitely', 'not', 'under-powered'), ('not', 'under-powered', 'anymore.'), ('under-powered', 'anymore.', 'Thor'), ('anymore.', 'Thor', 'in'), ('Thor', 'in', 'unleashed'), ('in', 'unleashed', 'in'), ('unleashed', 'in', 'this,'), ('in', 'this,', 'I'), ('this,', 'I', 'love'), ('I', 'love', 'that.')]


在前面的代码中唯一改变的是n值，即函数的第二个参数。

许多受监督的机器学习模型，例如朴素贝叶斯，使用n-gram来改善其特征空间。 n-gram还用于拼写纠正和文本摘要任务。

n-gram表示的一个挑战是它失去了文本的顺序性。 它通常与浅机器学习模型一起使用。 这种技术很少用于深度学习，因为RNN和Conv1D等架构会自动学习这些表示。

## 矢量化

将生成的标记映射到数字向量有两种流行的方法，称为One-Hot编码和词嵌入。 让我们通过编写一个简单的Python程序来理解如何将标记转换为这些向量表示。 我们还将讨论每种方法的各种优缺点。

An apple a day keeps doctor away said the doctor.

这一句的One-hot编码可以表示为表格格式，如下所示：

|An	|00000000|
|---|--------|
|apple|10000000|
|a	|001000000|
|day|	000100000|
|keeps|	000010000|
|doctor|	000001000|
|away	|000000100|
|said	|000000010|
|the	|000000001|

该表描述了标记及其One-Hot编码表示。 向量长度为9，因为句子中有九个唯一的单词。 许多机器学习库已经简化了创建一个One-Hot编码变量的过程。 我们将编写自己的实现以使其更易于理解，并且我们可以使用相同的实现来构建后续示例所需的其他功能。 以下代码包含Dictionary类，其中包含用于创建唯一单词字典的功能以及用于返回特定单词的One-Hot编码向量的函数。 我们来看看代码，然后浏览每个功能：


```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {} 
        self.idx2word = [] 
        self.length = 0 
    
    def add_word(self,word):
        if word not in self.idx2word:
            self.idx2word.append(word) 
            self.word2idx[word] = self.length + 1 
            self.length += 1 
            return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word) 
    
    def onehot_encoded(self,word): 
        vec = np.zeros(self.length) 
        vec[self.word2idx[word]] = 1 
        return vec
            
        
    
```

上述代码提供了三个重要功能：

* 初始化函数`__init__`创建一个`word2idx`字典将所有唯一单词与索引一起存储。 `idx2word`列表存储所有唯一单词，`length`变量包含文档中唯一单词的总数。

* add_word函数接受一个单词并将其添加到word2idx和idx2word，并增加词汇表的长度，前提是单词是唯一的。

* onehot_encoded函数接受一个单词并返回一个长度为N的向量，其中包含整个零，除了单词的索引。 如果传递的单词的索引是2，那么索引2处的向量的值将是1，并且所有剩余的值将是0。

正如我们已经定义了Dictionary类，让我们在thor_review数据上使用它。 以下代码演示了如何构建word2idx以及如何调用onehot_encoded函数：


```python
dic = Dictionary()

for tok in thor_review.split(): 
    dic.add_word(tok)

print(dic.word2idx)
```

    {'the': 1, 'action': 2, 'scenes': 3, 'were': 4, 'top': 5, 'notch': 6, 'in': 7, 'this': 8, 'movie.': 9, 'Thor': 10, 'has': 11, 'never': 12, 'been': 13, 'epic': 14, 'MCU.': 15, 'He': 16, 'does': 17, 'some': 18, 'pretty': 19, 'sh*t': 20, 'movie': 21, 'and': 22, 'he': 23, 'is': 24, 'definitely': 25, 'not': 26, 'under-powered': 27, 'anymore.': 28, 'unleashed': 29, 'this,': 30, 'I': 31, 'love': 32, 'that.': 33}


该单词的One-hot编码如下：


```python
import numpy as np
```


```python
# One-hot representation of the word 'were' 
dic.onehot_encoded('were')
```




    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



One-hot表示的挑战之一是数据太稀疏，并且随着词汇表中唯一单词数量的增加，向量的大小迅速增加，这被认为是一种限制，因此很少在深入学习中使用。

##  词嵌入

Word嵌入是在深度学习算法解决的问题中表示文本数据的一种非常流行的方式。 Word嵌入提供了填充浮动数字的单词的密集表示。 矢量维度根据词汇量大小而变化。 通常使用尺寸为50,100,256,300，有时为1,000的字嵌入。 尺寸大小是我们在训练阶段需要使用的超参数。

如果我们试图用单热表示来表示大小为20,000的词汇表，那么我们最终将得到20,000 x 20,000个数字，其中大部分将为零。 相同的词汇表可以在单词嵌入中表示为20,000 x维度大小，其中维度大小可以是10,50,300等。

创建词嵌入的一种方法是从包含随机数的每个标记的密集向量开始，然后训练诸如文档分类器或情感分类器的模型。 表示标记的浮点数将以一种方式调整，使得语义上更接近的单词将具有相似的表示。 为了理解它，让我们看看下图，我们在五部电影的二维图上绘制了嵌入向量字：

![](https://bennix.github.io/imgs/6_2.png)

上图显示了如何调整密集向量，以便在语义上相似的单词具有较小的距离。 由于超人，雷神和蝙蝠侠等电影片是基于漫画的动作电影，这些词的嵌入更接近，而电影“泰坦尼克号”的嵌入远离动作电影，更接近电影片头笔记本，因为它们是 浪漫电影。

当您的数据太少时，学习单词嵌入可能不可行，在这种情况下，我们可以使用由其他机器学习算法训练的单词嵌入。 从另一个任务生成的嵌入称为预训练单词嵌入。 我们将学习如何构建自己的单词嵌入并使用预训练的单词嵌入。

## 通过构建情绪分类器训练单词嵌入

在上一节中，我们简要地了解了嵌入字而没有实现它。 在本节中，我们将下载一个名为imdb的数据集，其中包含评论，并构建一个情绪分类器，用于计算评论的情绪是正面，负面还是未知。 在构建过程中，我们还将为imdb数据集中的单词训练单词嵌入。 我们将使用一个名为torchtext的库，它可以使下载，文本矢量化和批处理等许多过程变得更加容易。 培训情绪分类器将涉及以下步骤：

1.下载IMDB数据并执行文本标记化

2.建立词汇

3.生成批量的向量

4.使用词嵌入创建网络模型

5.训练模型

下载IMDB数据并执行文本标记化

对于与计算机视觉相关的应用，我们使用了torchvision库，它为我们提供了许多实用功能，有助于构建计算机视觉应用程序。 同样，有一个名为torchtext的库，它是PyTorch的一部分，它与PyTorch一起工作，通过提供不同的数据加载器和文本抽象，简化了许多与自然语言处理（NLP）相关的活动。 在撰写本文时，PyTorch 没有安装torchtext，需要单独安装。 您可以在计算机的命令行中运行以下代码以安装torchtext：

```
pip install torchtext

```

一旦安装，我们将能够使用它。 Torchtext提供了两个重要的模块叫做torchtext.data和torchtext.datasets。

>我们可以从以下链接下载IMDB Movies数据集： https://www.kaggle.com/orgesleka/imdbmovies

## torchtext.data

torchtext.data实例定义了一个名为Field的类，它帮助我们定义如何读取和标记数据。 让我们看一下以下示例，我们将使用它来准备IMDB数据集：


```python
from torchtext import data

TEXT = data.Field(lower=True, batch_first=True,fix_length=20)

LABEL = data.Field(sequential=False)

```

在前面的代码中，我们定义了两个Field对象，一个用于实际文本，另一个用于标签数据。 对于实际文本，我们希望torchtext将所有文本小写，标记化文本，并将其修剪为最大长度为20.如果我们正在为生产环境构建应用程序，我们可能会将长度固定为更大的数字。 但是，对于玩具示例，它运作良好。 Field构造函数还接受另一个名为tokenize的参数，该参数默认使用str.split函数。 我们还可以指定spaCy作为参数或任何其他标记化器。 对于我们的例子，我们将坚持使用str.split。

## torchtext.datasets

torchtext.datasets实例提供了使用不同数据集的包装器，如IMDB，TREC（问题分类），语言建模（WikiText-2）和一些其他数据集。 我们将使用torch.datasets下载IMDB数据集并将其拆分为训练和测试数据集。 以下代码执行此操作，当您第一次运行它时，可能需要几分钟，具体取决于您的宽带连接，因为它从Internet下载imdb数据集：



```python
from torchtext import datasets
train, test = datasets.IMDB.splits(TEXT, LABEL)
```

    downloading aclImdb_v1.tar.gz


之前的数据集的imdb类抽象出了下载，标记化和将数据库拆分为训练和测试数据集所涉及的所有复杂性。 train.fields包含一个字典，其中TEXT是键，值是LABEL。 让我们看看train.fields和训练集的每个元素包含：


```python
print('train.fields', train.fields)
```

    train.fields {'text': <torchtext.data.field.Field object at 0x1a1173a710>, 'label': <torchtext.data.field.Field object at 0x1a1173a780>}



```python
print(vars(train[0]))

```

    {'text': ['for', 'a', 'movie', 'that', 'gets', 'no', 'respect', 'there', 'sure', 'are', 'a', 'lot', 'of', 'memorable', 'quotes', 'listed', 'for', 'this', 'gem.', 'imagine', 'a', 'movie', 'where', 'joe', 'piscopo', 'is', 'actually', 'funny!', 'maureen', 'stapleton', 'is', 'a', 'scene', 'stealer.', 'the', 'moroni', 'character', 'is', 'an', 'absolute', 'scream.', 'watch', 'for', 'alan', '"the', 'skipper"', 'hale', 'jr.', 'as', 'a', 'police', 'sgt.'], 'label': 'pos'}


我们可以从这些结果中看到，单个元素包含字段，文本以及表示文本的所有标记，以及包含文本标签的标签字段。 现在我们已准备好imdb数据集进行批处理。

## 建立词汇

当我们为thor_review创建一个One-hot编码时，我们创建了一个word2idx字典，它被称为词汇表，因为它包含文档中唯一字的所有细节。 torchtext实例使我们更容易。 加载数据后，我们可以调用build_vocab并传递必要的参数，这些参数将负责构建数据的词汇表。 以下代码显示了如何构建词汇表：


```python
from torchtext.vocab import GloVe

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)

LABEL.build_vocab(train)
```

    .vector_cache/glove.6B.zip: 862MB [06:34, 2.19MB/s]                               
    100%|██████████| 400000/400000 [00:34<00:00, 11555.73it/s]


在前面的代码中，我们传入了我们需要构建词汇表的train对象，并且我们还要求它使用预先训练的300维度词嵌入来初始化向量.build_vocab对象只是下载并创建稍后将使用的维度， 当我们使用预训练权重训练情绪分类器时。 max_size实例限制词汇表中的单词数，min_freq删除任何未发生十次以上的单词，其中10可配置。


一旦构建了词汇，我们就可以获得不同的值，例如频率，单词索引和每个单词的向量表示。 以下代码演示了如何访问这些值：


```python
print(TEXT.vocab.freqs)
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


以下代码演示了如何访问结果：


```python
print(TEXT.vocab.vectors)

```

    tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0466,  0.2132, -0.0074,  ...,  0.0091, -0.2099,  0.0539],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.7724, -0.1800,  0.2072,  ...,  0.6736,  0.2263, -0.2919],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]])



```python
print(TEXT.vocab.stoi)
```

    defaultdict(<function _default_unk_index at 0x1a50ab9c80>, {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6, 'is': 7, 'in': 8, 'i': 9, 'this': 10, 'that': 11, 'it': 12, '/><br': 13, 'was': 14, 'as': 15, 'for': 16, 'with': 17, 'but': 18, 'on': 19, 'movie': 20, 'his': 21, 'are': 22, 'not': 23,  ... ...,  'politicians': 9979, 'portuguese': 9980, 'preachy': 9981, 'prefers': 9982, 'pressed': 9983, 'proceedings.': 9984, 'prolific': 9985, 'question:': 9986, 'questions.': 9987, 'rampant': 9988, 'replacement': 9989, 'replacing': 9990, 'repulsive': 9991, 'retrieve': 9992, 'reunited': 9993, 'rivers': 9994, 'sammy': 9995, 'sarandon': 9996, 'seconds.': 9997, 'secure': 9998, 'seeing,': 9999, 'self-indulgent': 10000, 'sequels,': 10001})


stoi允许访问包含单词及其索引的字典。

## 生成批量矢量

Torchtext提供Bucketlterator，它有助于批处理所有文本并用单词的索引号替换单词。 Bucketlterator实例附带了许多有用的参数，如batch_size，设备（GPU或CPU）和shuffle（数据是否必须洗牌）。 以下代码演示了如何创建为训练和测试数据集生成批处理的迭代器：


```python
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=128, device=-1,shuffle=True)

#device = -1 represents cpu , if u want gpu leave it to None.
```

上面的代码为训练和测试数据集提供了Bucketlterator对象。 以下代码将显示如何创建批处理并显示批处理的结果：


```python
batch = next(iter(train_iter)) 
batch.text

```




    tensor([[ 143,    9,   61,  ...,    0,   17,    0],
            [  10, 1156, 4622,  ...,   34, 6589,   27],
            [  52,   22,  109,  ...,   24,  287,   12],
            ...,
            [   9,  680,  119,  ...,    9,   86,   98],
            [  15,    3,  252,  ...,    9, 1344,   94],
            [7327,  625, 1008,  ...,    4,    0,    5]])




```python
batch.label
```




    tensor([1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1,
            1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2,
            2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2,
            2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2,
            1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
            1, 2, 1, 1, 1, 1, 2, 1])



从前面代码块中的结果，我们可以看到文本数据如何转换为大小的矩阵（batch_size * fix_len），即（128x20）。

## 使用词嵌入创建网络模型

我们之前简要讨论过词嵌入。 在本节中，我们把词嵌入作为我们创建的网络架构的一部分，并训练整个模型以预测每个评论的情绪。 在训练结束时，我们将有一个情感分类器模型以及imdb数据集的嵌入词。 以下代码演示了如何使用词嵌入来创建网络体系结构来预测情绪：


```python
train, test = datasets.IMDB.splits(TEXT, LABEL)
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))
```

    train.fields {'text': <torchtext.data.field.Field object at 0x1a1173a710>, 'label': <torchtext.data.field.Field object at 0x1a1173a780>}
    len(train) 25000
    vars(train[0]) {'text': ['for', 'a', 'movie', 'that', 'gets', 'no', 'respect', 'there', 'sure', 'are', 'a', 'lot', 'of', 'memorable', 'quotes', 'listed', 'for', 'this', 'gem.', 'imagine', 'a', 'movie', 'where', 'joe', 'piscopo', 'is', 'actually', 'funny!', 'maureen', 'stapleton', 'is', 'a', 'scene', 'stealer.', 'the', 'moroni', 'character', 'is', 'an', 'absolute', 'scream.', 'watch', 'for', 'alan', '"the', 'skipper"', 'hale', 'jr.', 'as', 'a', 'police', 'sgt.'], 'label': 'pos'}



```python
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device=-1)

train_iter.repeat = False
test_iter.repeat = False
```


```python
import torch.nn as nn

class EmbNet(nn.Module):
    def __init__(self,emb_size,hidden_size1,hidden_size2=200):
        super().__init__()
        self.embedding = nn.Embedding(emb_size,hidden_size1)
        self.fc = nn.Linear(hidden_size2,3)
        
    def forward(self,x):
        embeds = self.embedding(x).view(x.size(0),-1)
        out = self.fc(embeds)
        return F.log_softmax(out,dim=-1)
    
    
        
```

在上面的代码中，EmbNet创建了情绪分类模型。在`__init__`函数内部，我们初始化nn.Embedding类的一个对象，该对象有两个参数，即词汇表的大小和我们希望为每个单词创建的维度。由于我们限制了唯一单词的数量，因此词汇量大小将为10,000，我们可以从10的小嵌入大小开始。为了快速运行程序，小的嵌入大小很有用，但是当您尝试构建应用程序时生产系统，使用大尺寸的嵌入。我们还有一个线性图层，将单词嵌入映射到类别（正面，负面或未知）。

forward函数确定输入数据的处理方式。对于32的批量大小和20个字的最大长度的句子，我们将具有32×20的形状的输入。第一嵌入层充当查找表，用相应的嵌入向量替换每个单词。对于10的嵌入维度，当每个字被其相应的嵌入替换时，输出变为32×20×10。` view()`函数将展平嵌入层的结果。传递给view的第一个参数将保持该维度不变。在我们的例子中，我们不希望组合来自不同批次的数据，因此我们保留第一个维度并将张量中的其余值展平。应用视图功能后，张量形状将更改为32 x 200.密集层将展平的嵌入映射到类别数。一旦定义了网络，我们就可以像往常一样训练网络。

>请记住，在这个网络中，我们失去了文本的连续性，我们只是将它们用作一个Bag of Words。

## 训练模型

训练模型与我们在构建图像分类器时看到的非常类似，因此我们将使用相同的功能。 我们通过模型传递批量数据，计算输出和损失，然后优化模型权重，包括嵌入权重。 以下代码执行此操作：


```python
from torch.optim import optimizer 
```


```python
is_cuda = False

if torch.cuda.is_available():
    is_cuda=True
```


```python
def fit(epoch,model,data_loader,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx , batch in enumerate(data_loader):
        text , target = batch.text , batch.label
        if is_cuda:
            text,target = text.cuda(),target.cuda()
        
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output,target)
        
        running_loss += F.nll_loss(output,target,size_average=False).data.item()
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy}')
    return loss,accuracy
            


    
    
```


```python
model = EmbNet(len(TEXT.vocab.stoi),10)
model = model.cuda()
```


```python
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device=-1,shuffle=True)
train_iter.repeat = False
test_iter.repeat = False
optimizer = optim.Adam(model.parameters(),lr=0.001)
```


```python
import torch
import torch.optim as optim
import torch.nn.functional as F

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

for epoch in range(1,10):

    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
```
