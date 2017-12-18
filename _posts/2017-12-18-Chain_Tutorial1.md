---
ilayout: post
title: Chainer 入门教程（1）
date: 2017-12-18
categories: blog
tags: [Chainer,入门教程（1）]
descrption: Chainer 入门教程（1）
---

# Chainer 入门教程（1）

## Chainer的模块介绍

Chainer的模块介绍如下

|模块	|	功能|
|---|---|
|datasets|输入数据可以被格式化为这个类的模型输入。它涵盖了大部分输入数据结构的用例。|
|variable|它是一个函数/连接/Chain的输出。|
|functions|支持深度学习中广泛使用的功能的框架，例如 sigmoid, tanh, ReLU等|
|links| 支持深度学习中广泛使用的层的框架，例如全连接层，卷积层等|
|Chain|	连接和函数（层）连接起来形成一个“模型”。|
|optimizers| 指定用于调整模型参数的什么样的梯度下降方法，例如 SGD, AdaGrad, Adam.|
|serializers|保存/加载训练状态。例如 model, optimizer 等|
|iterators| 定义训练器使用的每个小批量数据。|
|training.updater|定义Trainer中使用的每个前向、反向传播的参数更新过程。|
|training.Trainer|管理训练器|

# 初始设置

下面是chainer模块的导入语句。



```python
# Initial setup following 
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
```


检查 chainer 版本


```python
print(chainer.__version__)
```

    3.0.0


## Variable

Chainer变量可以由Variable构造函数创建，它创建chainer.Variable类对象。
写变量时，它意味着Chainer的变量类。请不要混淆通常的“变量”名词。

> 注意：chainer之所以需要使用自己的Variable，Function类来计算，而不是使用numpy，是因为在深度学习训练中需要反向传播。变量保存其“计算历史”信息，函数具有作为差分函数的后向方法来处理反向传播。请参阅下面的更多细节


```python
from chainer import Variable
# creating numpy array
# this is `numpy.ndarray` class
a = np.asarray([1., 2., 3.], dtype=np.float32)

# chainer variable is created from `numpy.ndarray` or `cuda.ndarray` (explained later) 
x = Variable(a)

print('a: ', a, ', type: ', type(a))
print('x: ', x, ', type: ', type(x))
```

    a:  [ 1.  2.  3.] , type:  <class 'numpy.ndarray'>
    x:  variable([ 1.  2.  3.]) , type:  <class 'chainer.variable.Variable'>


在上面的代码中，numpy数据类型显式设置为dtype = np.float32。如果我们不设置数据类型，np.float64可能被用作64位环境中的默认类型。然而，这样的精确度通常是“太多”而不是机器学习所必需的。对于计算速度和内存使用情况，最好使用较低的精度。

## attribute

Chainer变量具有以下属性

* data
* dtype
* shape
* ndim
* size
* grad

它们与numpy.ndarray非常相似。您可以访问以下属性。


```python
# These attributes return the same

print('attributes', 'numpy.ndarray a', 'chainer.Variable x')
print('dtype', a.dtype, x.dtype)
print('shape', a.shape, x.shape)
print('ndim', a.ndim, x.ndim)
print('size', a.size, x.size)
```

    attributes numpy.ndarray a chainer.Variable x
    dtype float32 float32
    shape (3,) (3,)
    ndim 1 1
    size 3 3



```python
# Variable class has debug_print function, to show this Variable's properties.
x.debug_print()
```




    "<variable at 0x126cddd30>\n- device: CPU\n- backend: <class 'numpy.ndarray'>\n- shape: (3,)\n- dtype: float32\n- statistics: mean=2.00000000, std=0.81649661\n- grad: None"



一个例外是data属性，chainer变量的数据是指numpy.ndarray



```python
# x = Variable(a)

# `a` can be accessed via `data` attribute from chainer `Variable`
print('x.data is a : ', x.data is a)  # True -> means the reference of x.data and a are same. 
print('x.data: ', x.data)
```

    x.data is a :  True
    x.data:  [ 1.  2.  3.]


## Function

我们想要处理一些计算结果到变量。变量可以使用计算算术运算（例如+， - ，* ，/）方法是chainer.Function子类（例如。F.sigmoid，F.relu）



```python
# Arithmetric operation example
x = Variable(np.array([1, 2, 3], dtype=np.float32))
y = Variable(np.array([5, 6, 7], dtype=np.float32))

# usual arithmetric operator (this case `*`) can be used for calculation of `Variable`
z = x * y
print('z: ', z.data, ', type: ', type(z))
```

    z:  [  5.  12.  21.] , type:  <class 'chainer.variable.Variable'>


只有基本的计算可以用算术运算来完成。

Chainer通过chainer.functions提供了一系列广泛使用的功能，例如在深度学习中被广泛用作激活功能的Sigmoid函数或ReLU函数。



```python
# Functoin operation example
import chainer.functions as F

x = Variable(np.array([-1.5, -0.5, 0, 1, 2], dtype=np.float32))
sigmoid_x = F.sigmoid(x)  # sigmoid function. F.sigmoid is subclass of `Function`
relu_x = F.relu(x)        # ReLU function. F.relu is subclass of `Function`

print('x: ', x.data, ', type: ', type(x))
print('sigmoid_x: ', sigmoid_x.data, ', type: ', type(sigmoid_x))
print('relu_x: ', relu_x.data, ', type: ', type(relu_x))
```

    x:  [-1.5 -0.5  0.   1.   2. ] , type:  <class 'chainer.variable.Variable'>
    sigmoid_x:  [ 0.18242553  0.37754068  0.5         0.7310586   0.88079709] , type:  <class 'chainer.variable.Variable'>
    relu_x:  [ 0.  0.  0.  1.  2.] , type:  <class 'chainer.variable.Variable'>


>注意：您可以找到函数的大写字母，如F.Sigmoid或F.ReLU。基本上，这些大写字母是函数的实际类的实现，而小写字母的方法则是这些大写字母的实例getter方法。 当您使用F.xxx时，建议使用小写字母方法。sigmoid和ReLU函数是非线性函数，其形式是这样的。



```python
%matplotlib inline
import matplotlib.pyplot as plt


def plot_chainer_function(f, xmin=-5, xmax=5, title=None):
    """draw graph of chainer `Function` `f`

    :param f: function to be plotted
    :type f: chainer.Function
    :param xmin: int or float, minimum value of x axis
    :param xmax: int or float, maximum value of x axis
    :return:
    """
    a = np.arange(xmin, xmax, step=0.1)
    x = Variable(a)
    y = f(x)
    plt.clf()
    plt.figure()
    # x and y are `Variable`, their value can be accessed via `data` attribute
    plt.plot(x.data, y.data, 'r')
    if title is not None:
        plt.title(title)
    plt.show()

plot_chainer_function(F.sigmoid, title='Sigmoid')
plot_chainer_function(F.relu, title='ReLU')
```


    <matplotlib.figure.Figure at 0x1277ef0b8>



![png](https://bennix.github.io/output_19_1.png)



    <matplotlib.figure.Figure at 0x126d156d8>



![png](https://bennix.github.io/output_19_3.png)


## Link

Link类似于Function，但它拥有内部参数。这个内部参数在训练期间被调整。 Chainer通过 chainer.links提供了在流行论文中引入的全连接层，卷积层等层。

![](https://bennix.github.io/LinkLinear-700x137.png)

L.Linear是Link的一个例子。 
1. Linear拥有内部参数self.W和self.b 
2. Linear计算函数，F.linear。其输出取决于内部参数W和b。






```python
import chainer.links as L

in_size = 3  # input vector's dimension
out_size = 2  # output vector's dimension

linear_layer = L.Linear(in_size, out_size)  # L.linear is subclass of `Link`

"""linear_layer has 2 internal parameters `W` and `b`, which are `Variable`"""
print('W: ', linear_layer.W.data, ', shape: ', linear_layer.W.shape)
print('b: ', linear_layer.b.data, ', shape: ', linear_layer.b.shape)
```

    W:  [[ 0.01068367  0.58748239 -0.16838944]
     [-0.50624901  0.32139847  0.79277271]] , shape:  (2, 3)
    b:  [ 0.  0.] , shape:  (2,)


请注意，内部参数W是用一个随机值初始化的。所以每次执行上面的代码，结果都会不一样（试一下，检查一下！）。

该线性层将输入三维向量[x0，x1，x2 ...]（Variable 类）作为输入，并输出二维向量[y0，y1，y2 ...]（Variable 类）。
在等式形式中，$$ y_i = W * x_i + b $$ 其中i = 0,1,2 ...表示输入/输出的每个“小批量”。
>查看线性类的源代码，可以很容易地理解它只是调用F.linear
>```
  return linear.linear(x, self.W, self.b)
 ```


```python
x0 = np.array([1, 0, 0], dtype=np.float32)
x1 = np.array([1, 1, 1], dtype=np.float32)

x = Variable(np.array([x0, x1], dtype=np.float32))
y = linear_layer(x)
print('W: ', linear_layer.W.data)
print('b: ', linear_layer.b.data)
print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
```

    W:  [[ 0.01068367  0.58748239 -0.16838944]
     [-0.50624901  0.32139847  0.79277271]]
    b:  [ 0.  0.]
    x:  [[ 1.  0.  0.]
     [ 1.  1.  1.]]
    y:  [[ 0.01068367 -0.50624901]
     [ 0.42977661  0.6079222 ]]


让我强调一下Link和Function之间的区别。Function 的输入输出关系是固定的。另一方面，Link模块具有内部参数，可以通过修改（调整）该内部参数来改变功能行为。


```python
# Force update (set) internal parameters
linear_layer.W.data = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32)
linear_layer.b.data = np.array([3, 5], dtype=np.float32)

# below is same code with above cell, but output data y will be different
x0 = np.array([1, 0, 0], dtype=np.float32)
x1 = np.array([1, 1, 1], dtype=np.float32)

x = Variable(np.array([x0, x1], dtype=np.float32))
y = linear_layer(x)
print('W: ', linear_layer.W.data)
print('b: ', linear_layer.b.data)
print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
```

    W:  [[ 1.  2.  3.]
     [ 0.  0.  0.]]
    b:  [ 3.  5.]
    x:  [[ 1.  0.  0.]
     [ 1.  1.  1.]]
    y:  [[ 4.  5.]
     [ 9.  5.]]


输出y的值与上面的代码相比是不同的，尽管我们输入了相同的x值。
这些内部参数在机器学习训练期间被“调整”。通常情况下，我们不需要手动设置这些内部参数W或b，chainer会在训练过程中通过反向传播自动更新这些内部参数。

## Chain

Chain 构建神经网络。它通常由几个Link和Function模块组合而成。
我们来看一个例子，


```python
from chainer import Chain, Variable


# Defining your own neural networks using `Chain` class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 2)
            self.l2 = L.Linear(2, 1)
        
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
    

x = Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))

model = MyChain()
y = model(x)

print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
```

    x:  [[ 1.  2.]
     [ 3.  4.]]
    y:  [[-0.82586527]
     [-1.92913711]]


基于官方文档，Chain类提供以下功能
* 参数管理
* CPU / GPU移植支持
* 存储/加载功能

为您的神经网络代码提供方便的可重用性。
备注：上面的init_scope（）方法在chainer v2中引入，Link类实例在此范围内初始化。
在chainer v1中，Chain被初始化如下。具体来说，Link类实例在super方法的参数中被初始化。为了向后兼容，你也可以在chainer v2中使用这种类型的初始化。


```python
from chainer import Chain, Variable


# Defining your own neural networks using `Chain` class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(2, 2),
            l2=L.Linear(2, 1)
        )
        
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
    

x = Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))

model = MyChain()
y = model(x)

print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
```

    x:  [[ 1.  2.]
     [ 3.  4.]]
    y:  [[ 1.30976367]
     [ 2.13145947]]

