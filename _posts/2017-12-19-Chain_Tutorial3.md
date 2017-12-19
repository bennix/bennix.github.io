---
ilayout: post
title: Chainer 入门教程（3）MNIST数据集
date: 2017-12-19
categories: blog
tags: [Chainer,入门教程（3), MNIST数据集]
descrption: Chainer 入门教程（3）MNIST数据集
---

# MNIST 数据集

MNIST（美国国家标准与技术研究院（NIST）混合数据集）数据库是手写数字数据集，由Yann Lecun所创建的手写数字网站的MNIST数据库分发。

数据集由“手写数字图像”和“标签”组成。数字范围从0到9，共10个模式。

手写数字图像：这是尺寸为28 x 28像素的灰度图像。
标签：这是手写数字图像代表的实际数字号码。它是0到9。

![](https://bennix.github.io/imgs/mnist_plot-700x525.png)

MNIST数据集广泛用于“分类”，“图像识别”任务。这被认为是相对简单的任务，并且经常用于机器学习类中的“Hello world”程序。它也经常被用来比较算法在研究中的表现。

## 用Chainer处理MNIST数据集

对于像MNIST这样的著名数据集，Chainer提供了实用的函数来准备数据集。所以你不需要自己编写预处理代码，从网上下载数据集，并提取它，然后格式化等等。Chainer函数已经为你做好了一切！

目前，

MNIST
CIFAR-10, CIFAR-100
Penn Tree Bank (PTB)
数据集在Chainer中默认支持，请参考数据集的官方文档。

我们首先熟悉MNIST数据集的处理。准备MNIST数据集，您只需调用chainer.datasets.get_mnist函数。


```python
import numpy as np
import chainer

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist()
```

如果这是第一次执行，则首先开始几分钟的下载数据集以备日后使用。从此，chainer将自动参考缓存的内容，使其运行速度更快。
你将得到2个返回数据，分别对应“训练数据集”和“测试数据集”。MNIST共有70000条数据，其中训练数据集大小为60000，测试数据集大小为10000。



```python
# train[i] represents i-th data, there are 60000 training data
# test data structure is same, but total 10000 test data
print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))
```

    len(train), type  60000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>
    len(test), type  10000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>


我只解释下面的训练数据集，但是测试数据集有相同的数据集格式。
`train[i]`表示第i个数据，type =元组（$ x_i $，$ y_i $），其中$ x_i $是数组格式，大小为784的图像数据，$ y_i $是标签数据，表示图像的实际数字。



```python
print('train[0]', type(train[0]), len(train[0]))
# print(train[0])  # x_i = long array and y_i = label
```

    train[0] <class 'tuple'> 2


$ x_i $信息。您可以看到，图像仅表示为从0到1范围内的浮点数的数组。MNIST图像大小为28×28像素，因此它被表示为784 一维数组。



```python
# train[i][0] represents x_i, MNIST image data,
# type=numpy(784,) vector <- specified by ndim of get_mnist()
print('train[0][0]', train[0][0].shape)
np.set_printoptions(threshold=10)  # set np.inf to print all.
print(train[0][0])
```

    train[0][0] (784,)
    [ 0.  0.  0. ...,  0.  0.  0.]


$ y_i $信息。在下面的情况下，您可以看到第0个图像的标签为5。


```python
# train[i][1] represents y_i, MNIST label data(0-9), type=numpy() -> this means scalar
print('train[0][1]', train[0][1].shape, train[0][1])

```

    train[0][1] () 5


绘制MNIST
所以，每个第i个数据集由图像和标签组成
`train[i][0]`或`est[i][0]`：第i个手写图像
`train[i][1]`或`test[i][1]`：第i个标签
下面是一个绘图代码来检查图像（这只是python程序中的一个数组向量）的样子。此代码将生成本文顶部所示的MNIST映像。


```python
import os

import chainer
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

base_dir = ''

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist(ndim=1)

ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    # train[i][0] is i-th image data with size 28x28
    image = train[i][0].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, i+1)          # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('label = {}'.format(train[i][1]))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig(os.path.join(base_dir, 'mnist_plot.png'))
plt.show()
```


![png](https://bennix.github.io/imgs/output_14_0.png)

