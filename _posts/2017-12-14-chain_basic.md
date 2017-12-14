---
layout: post
title: Chainer 介绍
date: 2017-12-14
categories: blog
tags: [Chainer,介绍]
description: Chainer 介绍
---


# Chainer 介绍

这里是 Chainer 教程的第一部分。 在此部分中，您将学习如下内容:

* 现行框架的优缺点以及我们为什么开发 Chainer
* 前向以及反向计算的简单的例子
* 连接的使用以及梯度计算
* chains 的构建(即. 大多数框架所指的“模型”)
* 参数优化
* 连接和优化器的串行化

读完此部分，您将能够:

* 计算一些算式的梯度
* 用 Chainer 写一个多层感知器


## 核心概念

正如前文所述， Chainer 是一个柔性的神经网络框架。我们的主要目标就是柔性，使得我们能够简单直观的写出复杂的网络。

当下已有的深度学习框架使用的是“定义后运行”机制。即意味着，首先定义并且固化一个网络，再周而复始地馈入小批量数据进行训练。由于网络是在任何前向、反向计算前静态定义的，所有的逻辑作为**数据**必须事先嵌入网络中。 意味着，在诸如Caffe这样的框架中通过声明的方法定义网络结构。（注：可以使用torch.nn, 基于 Theano框架, 以及 TensorFlow 的命令语句定义一个静态网络）

## 边定义边运行

 Chainer 对应地采用了一种叫做 “边定义边运行” 的机制, 即, 网络可以在实际进行前向计算的时候同时被定义。 更加准确的说, Chainer 存储的是计算的历史结果而不是计算逻辑。这个策略使我们能够充分利用Python中编程逻辑的力量。例如，Chainer不需要任何魔法就可以将条件和循环引入到网络定义中。 边定义边运行是Chainer的核心概念。 我们将在本教程中展示如何动态定义网络。

这个策略也使编写多GPU并行化变得容易，因为逻辑更接近于网络操作。我们将在本教程后面的章节中回顾这些设施。

Chainer 将网络表示为**计算图**上的执行路径。计算图是一系列函数应用，因此它可以用多个`Function`对象来描述。当这个`Function`是一个神经网络层时，功能的参数将通过训练来更新。因此，该函数需要在内部保留可训练的参数，因此Chainer具有Link类，它可以在类的对象上保存可训练参数。在`Link`对象中执行的函数的参数被表示为`Variable`对象。 简言之，`Link`和`Function`之间的区别在于它是否包含可训练参数。 神经网络模型通常被描述为一系列`Link`和`Function`。

您可以通过动态“链接”各种`Link`和`Function`来构建计算图来定义Chain。在框架中，通过运行链接图来定义网络，因此名称是Chainer。

> 在本教程的示例代码中，我们假定为了简单起见，已经预先导入了以下语句：


```python
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
```

这些导入广泛出现在Chainer代码和例子中。为了简单起见，我们在本教程中省略了这些导入。

## 前向/反向计算

如上所述，Chainer使用“边定义边运行”方案，因此前向计算本身即定义了网络。为了开始前向计算，我们必须将输入数组设置为一个`Variable`对象。这里我们从一个简单的ndarray开始，只有一个元素：



```python
x_data = np.array([5], dtype=np.float32)
```


```python
x = Variable(x_data)
```

`Variable` 对象具有基本的算术运算符。为了计算 $y = x^2 - 2x + 1$, 只需写：


```python
y = x**2 - 2 * x + 1
```

结果y也是一个`Variable`对象，其值可以通过访问`data`属性来提取：


```python
y.data
```




    array([ 16.], dtype=float32)



y所持有的不仅是结果的数值。它也保持计算的历史（即计算图），其能够计算其差分。这是通过调用它的`backward()`方法完成的：



```python
y.backward()
```

其运行错误反向传播（也称为反向传播或反向模式自动差分）。然后，计算梯度并将其存储在输入变量x的`grad`属性中：


```python
x.grad
```




    array([ 8.], dtype=float32)



我们也可以计算中间变量的梯度。请注意，Chainer默认情况下会释放中间变量的梯度数组以提高内存效率。为了保留梯度信息，请将`retain_grad`参数传递给`backward`方法：


```python
z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True)
z.grad
```




    array([-1.], dtype=float32)



否则，`z.grad`将为`None`，如下所示：


```python
z = 2*x
y = x**2 - z + 1
y.backward()
z.grad
```


```python
z.grad is None
```




    True



所有这些计算都很容易推广到多元素数组输入。请注意，如果我们想从一个包含多元素数组的变量开始向后计算，我们必须手动设置初始错误。 因为当一个变量的`size`（这意味着数组中元素的个数）是1时，它被认为是一个表示损失值的变量对象，所以变量的`grad`属性被自动填充为1。 另一方面，当一个变量的大小大于1时，`grad`属性保持为`None`，并且在运行`backward（）`之前需要明确地设置初始错误。这可以简单地通过设置输出变量的`grad`属性来完成，如下所示：


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
x.grad
```




    array([[  0.,   2.,   4.],
           [  6.,   8.,  10.]], dtype=float32)



> 在`functions`模块中定义了许多采用`Variable`对象的函数。您可以将它们结合起来，实现具有自动后向计算的复杂功能.

## 连接


为了编写神经网络，我们必须将函数与参数相结合，并优化参数。你可以使用连接来做到这一点。`Link`是保存参数（即优化目标）的对象。

最基本的是像常规函数一样的连接。我们将介绍更高层次的连接，但是在这里将连接看作简化的带有参数的函数。


最经常使用的连接之一是`Linear` 连接（也称为完全连接层或仿射变换）。它代表一个数学函数 $f（x）= Wx + b$
，其中`W`为矩阵和`b` 为矢量参数。这个连接对应于`linear（）`，它接受`x`，`W`，`b` 作为参数。从三维空间到二维空间的线性连接由以下行定义：



```python
f = L.Linear(3, 2)
```

> 大多数函数和链接只接受小批量输入，其中输入数组的第一个维度被视为批量维度。在上面的线性连接情况下，输入必须具有（N，3）的形状，其中N是最小批量大小。

连接的参数被存储为属性。每个参数都是`Variable`的一个实例。在`Linear`连接的情况下，存储两个参数`W`和`b`。默认情况下，矩阵`W`是随机初始化的，而向量`b`是用零初始化的。


```python
f.W.data
```




    array([[ 0.19792122,  0.29951876, -0.31833425],
           [-0.59501284, -0.65519476, -0.00605371]], dtype=float32)




```python
f.b.data
```




    array([ 0.,  0.], dtype=float32)



`Linear` 连接的一个实例就像一个通常的函数：


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
y.data
```




    array([[-0.15804404, -1.9235636 ],
           [ 0.37927318, -5.69234705]], dtype=float32)



> 有时计算输入空间的维数很麻烦。线性连接和一些（反）卷积连接可以在实例化时省略输入维度，并从第一个小批量中推断出输入维度来。

> 例如，以下行创建一个输出维度为两个的线性连接：


```python
g = L.Linear(2)
```

>如果我们输入一个小批量的形状为`（N，M）`，则输入维数将被推断为`M`，这意味着`g.W`将是`2×M`矩阵。 请注意，它的参数在第一个小批处理中以懒惰的方式初始化。因此，如果没有数据放入连接，则`f`不具有`W`属性。 

参数的梯度由`backward（）`方法计算。请注意，梯度是由方法累积而不是覆盖。所以首先你必须清除梯度来更新计算。可以通过调用`cleargrads（）`方法来完成。


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
g = L.Linear(2)
p=g(x)
```


```python
p
```




    variable([[-2.64461255,  2.90179563],
              [-6.81166267,  4.94405651]])




```python
g.cleargrads()
```


```python
g.grad = np.ones((2, 2), dtype=np.float32)
```


```python
g.W.grad
```


```python
g.b.grad
```

##  基于 chain 写一个模型

大多数神经网络体系结构包含多个连接。例如，多层感知器由多个线性层组成。我们可以通过组合多个连接来编写具有可训练参数的复杂过程：



```python
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_forward(x):
    h = l1(x)
    return l2(h)
```

这里的L表示`links`模块。以这种方式定义参数的过程很难重用。更多Pythonic的方式是将连接和程序组合成一个类：


```python
class MyProc(object):
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)
```

为了使其更加可重用，我们希望支持参数管理，CPU / GPU迁移，强大而灵活的保存/加载功能等。这些功能都由Chainer中的`Chain`类支持。那么，我们要做的就是将上面的类定义为 `Chain` 的子类：



```python
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)
            
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
```

它显示了一个复杂的连接是如何通过更连接的链接构建的。诸如`l1`和`l2`被称为MyChain的子连接。注意，`Chain`本身继承自`Link`。这意味着我们可以定义更复杂的连接，将MyChain对象作为子连接。


>我们经常通过__call__运算符定义一个前向连接。这样的连接和Chains是可调用的，并且像常规函数和变量一样。


另一种定义chain的方法是使用`ChainList`类，它的行为类似于连接列表：



```python
class MyChain2(ChainList):
    def __init__(self):
        super(MyChain2, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)
```

`ChainList`可以方便地使用任意数量的连接，但是如果连接的数量固定且与上述情况相同，则建议使用`Chain`类作为基类。


## 优化器

为了获得良好的参数值，我们必须通过优化器类来优化它们。它在给定的连接上运行数值优化算法。许多算法在优化器模块中实现。这里我们使用最简单的称为随机梯度下降（SGD）：



```python
model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

```


setup（）方法针对给定的连接准备对应的优化器。

一些参数/梯度操作，例如权重衰减和梯度剪切，可以通过设置钩子函数到优化器来完成。 钩子函数在梯度计算之后和实际更新参数之前调用。例如，我们可以通过预先运行下一行来设置权重衰减正则化： 


```python
 optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
```

当然，你可以编写自己的钩子函数。它应该是一个函数或一个可调用的对象，以优化器为参数。

有两种使用优化器的方法。一个是通过训练器使用它，我们将在下面的部分中看到。另一种方式是直接使用它。我们在这里回顾后一种情况。如果您有兴趣以简单的方式使用优化器，请跳过本节并转到下一节。

还有两种直接使用优化器的方法。一个是手动计算梯度，然后调用没有参数的 `update（）`方法。不要忘记事先清除梯度！



```python
x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
# compute gradient here...
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()
```

另一种方法是将损失函数传递给`update（）`方法。在这种情况下，`cleargrads()` 会被update方法自动调用，所以用户不必手动调用它。



```python
def lossfun(arg1, arg2):
    # calculate loss
    loss = F.sum(model(arg1 - arg2))
    return loss
```


```python
arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))
```

## 训练器

当我们想要训练神经网络时，我们必须运行训练循环多次更新参数。典型的训练循环包括以下过程：

1. 对训练数据集进行迭代
2. 提取小批量的预处理
3. 神经网络的前向/后向计算
4. 参数更新
5. 评估验证数据集上的当前参数
6. 记录和打印中间结果

Chainer提供了一个简单而强大的方法来使写这样的训练过程变得容易。训练循环抽象主要由两部分组成：




* **数据集抽象**。它在上面的列表中实现了1和2。核心组件在数据集模块中定义。数据集和迭代器模块中还有许多数据集和迭代器的实现。

* **训练器**。它在上面的列表中实现3,4,5和6。整个程序由Trainer执行。更新参数（3和4）的方式由`Updater`定义，可以自由定制。 5和6由`Extension`的实例来实现，它将一个额外的过程附加到训练循环中。用户可以通过添加扩展来自由定制训练程序。用户也可以实现自己的扩展。

## 序列化器


在继续第一个例子之前，我们介绍Serializer，这是本页中描述的最后一个核心功能。序列化器是一个简单的接口来序列化或反序列化一个对象。连接，优化器和训练器都支持序列化。

序列化器模块中定义了具体的序列化器。它支持NumPy NPZ和HDF5格式。

例如，我们可以通过serializers.save_npz（）函数将连接对象序列化成NPZ文件：



```python
serializers.save_npz('my.model', model)
```

它将模型的参数以NPZ格式保存到文件“my.model”中。保存的模型可以被serializers.load_npz（）函数读取：


```python
serializers.load_npz('my.model', model)
```

>请注意，只有参数和持久值由该序列化代码序列化。其他属性不会自动保存。您可以通过`Link.add_persistent（）`方法将数组，标量或任何可序列化的对象注册为持久值。注册的值可以通过传递给`add_persistent`方法的名称的属性来访问。 




优化器的状态也可以通过相同的函数来保存：


```python
serializers.save_npz('my.state', optimizer)
serializers.load_npz('my.state', optimizer)
```

>请注意，优化器的序列化只保存其内部状态，包括迭代次数，MomentumSGD的动量向量等。它不保存目标连接的参数和永久值。我们必须明确地保存与优化器的目标连接，从保存状态恢复优化。

如果安装了h5py软件包，则支持HDF5格式。 HDF5格式的序列化和反序列化与NPZ格式的序列化和反序列化几乎相同;只需用save_hdf5（）和load_hdf5（）分别替换save_npz（）和load_npz（）即可。



## 例子：基于MNIST的多层感知器

现在，您可以使用多层感知器（MLP）来解决多类分类任务。我们使用手写数字数据集称为MNIST，这是机器学习中长期使用的事实上的“hello world”示例之一。这个MNIST例子也可以在官方仓库的examples / mnist目录中找到。我们演示如何使用训练器来构建和运行本节中的训练循环。

我们首先必须准备MNIST数据集。 MNIST数据集由70,000个尺寸为28×28（即784个像素）的灰度图像和相应的数字标签组成。数据集默认分为6万个训练图像和10,000个测试图像。我们可以通过`datasets.get_mnist（）`获得矢量化版本（即一组784维向量）。


```python
train, test = datasets.get_mnist()
```

此代码自动下载MNIST数据集并将NumPy数组保存到 `$(HOME)/.chainer` 目录中。返回的训练集和测试集可以看作图像标签配对的列表（严格地说，它们是TupleDataset的实例）。

我们还必须定义如何迭代这些数据集。我们想要在数据集的每次扫描开始时对每个epoch的训练数据集进行重新洗牌。在这种情况下，我们可以使用`iterators.SerialIterator`。


```python
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
```

另一方面，我们不必洗牌测试数据集。在这种情况下，我们可以通过shuffle = False来禁止混洗。当底层数据集支持快速切片时，它使迭代速度更快。


```python
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
```

当所有的例子被访问时，我们停止迭代通过设定 repeat=False 。测试/验证数据集通常需要此选项;没有这个选项，迭代进入一个无限循环。

接下来，我们定义架构。我们使用一个简单的三层网络，每层100个单元。


```python
class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
```

该链接使用relu（）作为激活函数。请注意，“l3”链接是最终的全连接层，其输出对应于十个数字的分数。

为了计算损失值或评估预测的准确性，我们在上面的MLP连接的基础上定义一个分类器连接：


```python
class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
```

这个分类器类计算准确性和损失，并返回损失值。参数对x和t对应于数据集中的每个示例（图像和标签的元组）。 `softmax_cross_entropy（）`计算给定预测和基准真实标签的损失值。 `accuracy() `计算预测准确度。我们可以为分类器的一个实例设置任意的预测器连接。 

`report()` 函数向训练器报告损失和准确度。收集训练统计信息的具体机制参见 `Reporter`. 您也可以采用类似的方式收集其他类型的观测值，如激活统计。

请注意，类似上面的分类器的类被定义为`chainer.links.Classifier`。因此，我们将使用此预定义的`Classifier`连接而不是使用上面的示例。



```python
model = L.Classifier(MLP(100, 10))  # the input size, 784, is inferred
optimizer = optimizers.SGD()
optimizer.setup(model)
```

现在我们可以建立一个训练器对象。



```python
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
```

第二个参数（20，'epoch'）表示训练的持续时间。我们可以使用epoch或迭代作为单位。在这种情况下，我们通过遍历训练集20次来训练多层感知器。

为了调用训练循环，我们只需调用run（）方法。

这个方法执行整个训练序列。

上面的代码只是优化了参数。在大多数情况下，我们想看看培训的进展情况，我们可以在调用run方法之前使用扩展插入。





```python
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()  
```

    epoch       main/accuracy  validation/main/accuracy
    [J     total [..................................................]  0.83%
    this epoch [########..........................................] 16.67%
           100 iter, 0 epoch / 20 epochs
           inf iters/sec. Estimated time to finish: 0:00:00.
    [4A[J     total [..................................................]  1.67%
    this epoch [################..................................] 33.33%
           200 iter, 0 epoch / 20 epochs
        270.19 iters/sec. Estimated time to finish: 0:00:43.672168.
    [4A[J     total [#.................................................]  2.50%
    this epoch [#########################.........................] 50.00%
           300 iter, 0 epoch / 20 epochs
        271.99 iters/sec. Estimated time to finish: 0:00:43.017048.
    [4A[J     total [#.................................................]  3.33%
    this epoch [#################################.................] 66.67%
           400 iter, 0 epoch / 20 epochs
        274.82 iters/sec. Estimated time to finish: 0:00:42.209075.
    [4A[J     total [##................................................]  4.17%
    this epoch [#########################################.........] 83.33%
           500 iter, 0 epoch / 20 epochs
        275.19 iters/sec. Estimated time to finish: 0:00:41.789476.
    [4A[J1           0.6581         0.8475                    
    [J     total [##................................................]  5.00%
    this epoch [..................................................]  0.00%
           600 iter, 1 epoch / 20 epochs
        250.26 iters/sec. Estimated time to finish: 0:00:45.553447.
    [4A[J     total [##................................................]  5.83%
    this epoch [########..........................................] 16.67%
           700 iter, 1 epoch / 20 epochs
        251.78 iters/sec. Estimated time to finish: 0:00:44.879872.
    [4A[J     total [###...............................................]  6.67%
    this epoch [################..................................] 33.33%
           800 iter, 1 epoch / 20 epochs
        253.07 iters/sec. Estimated time to finish: 0:00:44.257362.
    [4A[J     total [###...............................................]  7.50%
    this epoch [#########################.........................] 50.00%
           900 iter, 1 epoch / 20 epochs
        253.97 iters/sec. Estimated time to finish: 0:00:43.706513.
    [4A[J     total [####..............................................]  8.33%
    this epoch [#################################.................] 66.67%
          1000 iter, 1 epoch / 20 epochs
        255.94 iters/sec. Estimated time to finish: 0:00:42.979372.
    [4A[J     total [####..............................................]  9.17%
    this epoch [#########################################.........] 83.33%
          1100 iter, 1 epoch / 20 epochs
        257.61 iters/sec. Estimated time to finish: 0:00:42.311793.
    [4A[J2           0.868483       0.8922                    
    [J     total [#####.............................................] 10.00%
    this epoch [..................................................]  0.00%
          1200 iter, 2 epoch / 20 epochs
        250.02 iters/sec. Estimated time to finish: 0:00:43.196043.
    [4A[J     total [#####.............................................] 10.83%
    this epoch [########..........................................] 16.67%
          1300 iter, 2 epoch / 20 epochs
        250.73 iters/sec. Estimated time to finish: 0:00:42.674737.
    [4A[J     total [#####.............................................] 11.67%
    this epoch [################..................................] 33.33%
          1400 iter, 2 epoch / 20 epochs
        250.76 iters/sec. Estimated time to finish: 0:00:42.271780.
    [4A[J     total [######............................................] 12.50%
    this epoch [#########################.........................] 50.00%
          1500 iter, 2 epoch / 20 epochs
        250.66 iters/sec. Estimated time to finish: 0:00:41.889907.
    [4A[J     total [######............................................] 13.33%
    this epoch [#################################.................] 66.67%
          1600 iter, 2 epoch / 20 epochs
        250.63 iters/sec. Estimated time to finish: 0:00:41.494966.
    [4A[J     total [#######...........................................] 14.17%
    this epoch [#########################################.........] 83.33%
          1700 iter, 2 epoch / 20 epochs
         250.3 iters/sec. Estimated time to finish: 0:00:41.150503.
    [4A[J3           0.893583       0.9065                    
    [J     total [#######...........................................] 15.00%
    this epoch [..................................................]  0.00%
          1800 iter, 3 epoch / 20 epochs
        245.03 iters/sec. Estimated time to finish: 0:00:41.627412.
    [4A[J     total [#######...........................................] 15.83%
    this epoch [########..........................................] 16.67%
          1900 iter, 3 epoch / 20 epochs
        246.29 iters/sec. Estimated time to finish: 0:00:41.007745.
    [4A[J     total [########..........................................] 16.67%
    this epoch [################..................................] 33.33%
          2000 iter, 3 epoch / 20 epochs
        246.63 iters/sec. Estimated time to finish: 0:00:40.547184.
    [4A[J     total [########..........................................] 17.50%
    this epoch [#########################.........................] 50.00%
          2100 iter, 3 epoch / 20 epochs
        247.22 iters/sec. Estimated time to finish: 0:00:40.045529.
    [4A[J     total [#########.........................................] 18.33%
    this epoch [#################################.................] 66.67%
          2200 iter, 3 epoch / 20 epochs
        248.21 iters/sec. Estimated time to finish: 0:00:39.482367.
    [4A[J     total [#########.........................................] 19.17%
    this epoch [#########################################.........] 83.33%
          2300 iter, 3 epoch / 20 epochs
        248.73 iters/sec. Estimated time to finish: 0:00:38.997955.
    [4A[J4           0.90485        0.9154                    
    [J     total [##########........................................] 20.00%
    this epoch [..................................................]  0.00%
          2400 iter, 4 epoch / 20 epochs
        244.21 iters/sec. Estimated time to finish: 0:00:39.309754.
    [4A[J     total [##########........................................] 20.83%
    this epoch [########..........................................] 16.67%
          2500 iter, 4 epoch / 20 epochs
        244.55 iters/sec. Estimated time to finish: 0:00:38.847329.
    [4A[J     total [##########........................................] 21.67%
    this epoch [################..................................] 33.33%
          2600 iter, 4 epoch / 20 epochs
        245.78 iters/sec. Estimated time to finish: 0:00:38.245938.
    [4A[J     total [###########.......................................] 22.50%
    this epoch [#########################.........................] 50.00%
          2700 iter, 4 epoch / 20 epochs
        246.89 iters/sec. Estimated time to finish: 0:00:37.668330.
    [4A[J     total [###########.......................................] 23.33%
    this epoch [#################################.................] 66.67%
          2800 iter, 4 epoch / 20 epochs
        247.85 iters/sec. Estimated time to finish: 0:00:37.119132.
    [4A[J     total [############......................................] 24.17%
    this epoch [#########################################.........] 83.33%
          2900 iter, 4 epoch / 20 epochs
        248.84 iters/sec. Estimated time to finish: 0:00:36.568961.
    [4A[J5           0.9128         0.9222                    
    [J     total [############......................................] 25.00%
    this epoch [..................................................]  0.00%
          3000 iter, 5 epoch / 20 epochs
        246.32 iters/sec. Estimated time to finish: 0:00:36.537719.
    [4A[J     total [############......................................] 25.83%
    this epoch [########..........................................] 16.67%
          3100 iter, 5 epoch / 20 epochs
        247.27 iters/sec. Estimated time to finish: 0:00:35.993611.
    [4A[J     total [#############.....................................] 26.67%
    this epoch [################..................................] 33.33%
          3200 iter, 5 epoch / 20 epochs
        247.64 iters/sec. Estimated time to finish: 0:00:35.535495.
    [4A[J     total [#############.....................................] 27.50%
    this epoch [#########################.........................] 50.00%
          3300 iter, 5 epoch / 20 epochs
        248.02 iters/sec. Estimated time to finish: 0:00:35.078297.
    [4A[J     total [##############....................................] 28.33%
    this epoch [#################################.................] 66.67%
          3400 iter, 5 epoch / 20 epochs
         248.3 iters/sec. Estimated time to finish: 0:00:34.635942.
    [4A[J     total [##############....................................] 29.17%
    this epoch [#########################################.........] 83.33%
          3500 iter, 5 epoch / 20 epochs
        248.35 iters/sec. Estimated time to finish: 0:00:34.225545.
    [4A[J6           0.9182         0.9251                    
    [J     total [###############...................................] 30.00%
    this epoch [..................................................]  0.00%
          3600 iter, 6 epoch / 20 epochs
        245.49 iters/sec. Estimated time to finish: 0:00:34.217710.
    [4A[J     total [###############...................................] 30.83%
    this epoch [########..........................................] 16.67%
          3700 iter, 6 epoch / 20 epochs
        245.88 iters/sec. Estimated time to finish: 0:00:33.755860.
    [4A[J     total [###############...................................] 31.67%
    this epoch [################..................................] 33.33%
          3800 iter, 6 epoch / 20 epochs
         245.9 iters/sec. Estimated time to finish: 0:00:33.346716.
    [4A[J     total [################..................................] 32.50%
    this epoch [#########################.........................] 50.00%
          3900 iter, 6 epoch / 20 epochs
        245.96 iters/sec. Estimated time to finish: 0:00:32.931534.
    [4A[J     total [################..................................] 33.33%
    this epoch [#################################.................] 66.67%
          4000 iter, 6 epoch / 20 epochs
        245.99 iters/sec. Estimated time to finish: 0:00:32.521949.
    [4A[J     total [#################.................................] 34.17%
    this epoch [#########################################.........] 83.33%
          4100 iter, 6 epoch / 20 epochs
        246.12 iters/sec. Estimated time to finish: 0:00:32.098613.
    [4A[J7           0.923683       0.9281                    
    [J     total [#################.................................] 35.00%
    this epoch [..................................................]  0.00%
          4200 iter, 7 epoch / 20 epochs
        244.37 iters/sec. Estimated time to finish: 0:00:31.918388.
    [4A[J     total [#################.................................] 35.83%
    this epoch [########..........................................] 16.67%
          4300 iter, 7 epoch / 20 epochs
        244.24 iters/sec. Estimated time to finish: 0:00:31.526645.
    [4A[J     total [##################................................] 36.67%
    this epoch [################..................................] 33.33%
          4400 iter, 7 epoch / 20 epochs
         244.7 iters/sec. Estimated time to finish: 0:00:31.058855.
    [4A[J     total [##################................................] 37.50%
    this epoch [#########################.........................] 50.00%
          4500 iter, 7 epoch / 20 epochs
        245.22 iters/sec. Estimated time to finish: 0:00:30.584594.
    [4A[J     total [###################...............................] 38.33%
    this epoch [#################################.................] 66.67%
          4600 iter, 7 epoch / 20 epochs
        245.84 iters/sec. Estimated time to finish: 0:00:30.100470.
    [4A[J     total [###################...............................] 39.17%
    this epoch [#########################################.........] 83.33%
          4700 iter, 7 epoch / 20 epochs
         246.3 iters/sec. Estimated time to finish: 0:00:29.638363.
    [4A[J8           0.927233       0.9312                    
    [J     total [####################..............................] 40.00%
    this epoch [..................................................]  0.00%
          4800 iter, 8 epoch / 20 epochs
        245.02 iters/sec. Estimated time to finish: 0:00:29.385524.
    [4A[J     total [####################..............................] 40.83%
    this epoch [########..........................................] 16.67%
          4900 iter, 8 epoch / 20 epochs
        245.47 iters/sec. Estimated time to finish: 0:00:28.923795.
    [4A[J     total [####################..............................] 41.67%
    this epoch [################..................................] 33.33%
          5000 iter, 8 epoch / 20 epochs
        245.91 iters/sec. Estimated time to finish: 0:00:28.465973.
    [4A[J     total [#####################.............................] 42.50%
    this epoch [#########################.........................] 50.00%
          5100 iter, 8 epoch / 20 epochs
        246.47 iters/sec. Estimated time to finish: 0:00:27.994909.
    [4A[J     total [#####################.............................] 43.33%
    this epoch [#################################.................] 66.67%
          5200 iter, 8 epoch / 20 epochs
        246.95 iters/sec. Estimated time to finish: 0:00:27.535404.
    [4A[J     total [######################............................] 44.17%
    this epoch [#########################################.........] 83.33%
          5300 iter, 8 epoch / 20 epochs
        247.33 iters/sec. Estimated time to finish: 0:00:27.089584.
    [4A[J9           0.931317       0.9341                    
    [J     total [######################............................] 45.00%
    this epoch [..................................................]  0.00%
          5400 iter, 9 epoch / 20 epochs
        245.58 iters/sec. Estimated time to finish: 0:00:26.874639.
    [4A[J     total [######################............................] 45.83%
    this epoch [########..........................................] 16.67%
          5500 iter, 9 epoch / 20 epochs
        245.87 iters/sec. Estimated time to finish: 0:00:26.437190.
    [4A[J     total [#######################...........................] 46.67%
    this epoch [################..................................] 33.33%
          5600 iter, 9 epoch / 20 epochs
        246.33 iters/sec. Estimated time to finish: 0:00:25.981189.
    [4A[J     total [#######################...........................] 47.50%
    this epoch [#########################.........................] 50.00%
          5700 iter, 9 epoch / 20 epochs
        246.78 iters/sec. Estimated time to finish: 0:00:25.528408.
    [4A[J     total [########################..........................] 48.33%
    this epoch [#################################.................] 66.67%
          5800 iter, 9 epoch / 20 epochs
         247.2 iters/sec. Estimated time to finish: 0:00:25.080847.
    [4A[J     total [########################..........................] 49.17%
    this epoch [#########################################.........] 83.33%
          5900 iter, 9 epoch / 20 epochs
        247.69 iters/sec. Estimated time to finish: 0:00:24.627826.
    [4A[J10          0.934733       0.9369                    
    [J     total [#########################.........................] 50.00%
    this epoch [..................................................]  0.00%
          6000 iter, 10 epoch / 20 epochs
        246.59 iters/sec. Estimated time to finish: 0:00:24.332159.
    [4A[J     total [#########################.........................] 50.83%
    this epoch [########..........................................] 16.67%
          6100 iter, 10 epoch / 20 epochs
           247 iters/sec. Estimated time to finish: 0:00:23.886641.
    [4A[J     total [#########################.........................] 51.67%
    this epoch [################..................................] 33.33%
          6200 iter, 10 epoch / 20 epochs
        247.36 iters/sec. Estimated time to finish: 0:00:23.448076.
    [4A[J     total [##########################........................] 52.50%
    this epoch [#########################.........................] 50.00%
          6300 iter, 10 epoch / 20 epochs
        247.73 iters/sec. Estimated time to finish: 0:00:23.008541.
    [4A[J     total [##########################........................] 53.33%
    this epoch [#################################.................] 66.67%
          6400 iter, 10 epoch / 20 epochs
        248.16 iters/sec. Estimated time to finish: 0:00:22.566452.
    [4A[J     total [###########################.......................] 54.17%
    this epoch [#########################################.........] 83.33%
          6500 iter, 10 epoch / 20 epochs
        248.61 iters/sec. Estimated time to finish: 0:00:22.123234.
    [4A[J11          0.937883       0.9414                    
    [J     total [###########################.......................] 55.00%
    this epoch [..................................................]  0.00%
          6600 iter, 11 epoch / 20 epochs
        247.52 iters/sec. Estimated time to finish: 0:00:21.816101.
    [4A[J     total [###########################.......................] 55.83%
    this epoch [########..........................................] 16.67%
          6700 iter, 11 epoch / 20 epochs
        247.67 iters/sec. Estimated time to finish: 0:00:21.399559.
    [4A[J     total [############################......................] 56.67%
    this epoch [################..................................] 33.33%
          6800 iter, 11 epoch / 20 epochs
        247.88 iters/sec. Estimated time to finish: 0:00:20.977519.
    [4A[J     total [############################......................] 57.50%
    this epoch [#########################.........................] 50.00%
          6900 iter, 11 epoch / 20 epochs
        248.13 iters/sec. Estimated time to finish: 0:00:20.553526.
    [4A[J     total [#############################.....................] 58.33%
    this epoch [#################################.................] 66.67%
          7000 iter, 11 epoch / 20 epochs
        248.28 iters/sec. Estimated time to finish: 0:00:20.138771.
    [4A[J     total [#############################.....................] 59.17%
    this epoch [#########################################.........] 83.33%
          7100 iter, 11 epoch / 20 epochs
        248.42 iters/sec. Estimated time to finish: 0:00:19.724508.
    [4A[J12          0.940583       0.9438                    
    [J     total [##############################....................] 60.00%
    this epoch [..................................................]  0.00%
          7200 iter, 12 epoch / 20 epochs
        247.45 iters/sec. Estimated time to finish: 0:00:19.398094.
    [4A[J     total [##############################....................] 60.83%
    this epoch [########..........................................] 16.67%
          7300 iter, 12 epoch / 20 epochs
        247.79 iters/sec. Estimated time to finish: 0:00:18.967364.
    [4A[J     total [##############################....................] 61.67%
    this epoch [################..................................] 33.33%
          7400 iter, 12 epoch / 20 epochs
         248.1 iters/sec. Estimated time to finish: 0:00:18.540794.
    [4A[J     total [###############################...................] 62.50%
    this epoch [#########################.........................] 50.00%
          7500 iter, 12 epoch / 20 epochs
        248.46 iters/sec. Estimated time to finish: 0:00:18.111734.
    [4A[J     total [###############################...................] 63.33%
    this epoch [#################################.................] 66.67%
          7600 iter, 12 epoch / 20 epochs
        248.77 iters/sec. Estimated time to finish: 0:00:17.687175.
    [4A[J     total [################################..................] 64.17%
    this epoch [#########################################.........] 83.33%
          7700 iter, 12 epoch / 20 epochs
        249.07 iters/sec. Estimated time to finish: 0:00:17.264007.
    [4A[J13          0.942633       0.9451                    
    [J     total [################################..................] 65.00%
    this epoch [..................................................]  0.00%
          7800 iter, 13 epoch / 20 epochs
        248.22 iters/sec. Estimated time to finish: 0:00:16.920387.
    [4A[J     total [################################..................] 65.83%
    this epoch [########..........................................] 16.67%
          7900 iter, 13 epoch / 20 epochs
        248.52 iters/sec. Estimated time to finish: 0:00:16.497482.
    [4A[J     total [#################################.................] 66.67%
    this epoch [################..................................] 33.33%
          8000 iter, 13 epoch / 20 epochs
        248.86 iters/sec. Estimated time to finish: 0:00:16.073042.
    [4A[J     total [#################################.................] 67.50%
    this epoch [#########################.........................] 50.00%
          8100 iter, 13 epoch / 20 epochs
         249.2 iters/sec. Estimated time to finish: 0:00:15.649976.
    [4A[J     total [##################################................] 68.33%
    this epoch [#################################.................] 66.67%
          8200 iter, 13 epoch / 20 epochs
        249.47 iters/sec. Estimated time to finish: 0:00:15.232395.
    [4A[J     total [##################################................] 69.17%
    this epoch [#########################################.........] 83.33%
          8300 iter, 13 epoch / 20 epochs
        249.72 iters/sec. Estimated time to finish: 0:00:14.816816.
    [4A[J14          0.945083       0.9465                    
    [J     total [###################################...............] 70.00%
    this epoch [..................................................]  0.00%
          8400 iter, 14 epoch / 20 epochs
        248.89 iters/sec. Estimated time to finish: 0:00:14.463988.
    [4A[J     total [###################################...............] 70.83%
    this epoch [########..........................................] 16.67%
          8500 iter, 14 epoch / 20 epochs
        249.19 iters/sec. Estimated time to finish: 0:00:14.045501.
    [4A[J     total [###################################...............] 71.67%
    this epoch [################..................................] 33.33%
          8600 iter, 14 epoch / 20 epochs
        249.44 iters/sec. Estimated time to finish: 0:00:13.630462.
    [4A[J     total [####################################..............] 72.50%
    this epoch [#########################.........................] 50.00%
          8700 iter, 14 epoch / 20 epochs
        249.64 iters/sec. Estimated time to finish: 0:00:13.219213.
    [4A[J     total [####################################..............] 73.33%
    this epoch [#################################.................] 66.67%
          8800 iter, 14 epoch / 20 epochs
        249.92 iters/sec. Estimated time to finish: 0:00:12.804288.
    [4A[J     total [#####################################.............] 74.17%
    this epoch [#########################################.........] 83.33%
          8900 iter, 14 epoch / 20 epochs
        250.18 iters/sec. Estimated time to finish: 0:00:12.390956.
    [4A[J15          0.947233       0.9495                    
    [J     total [#####################################.............] 75.00%
    this epoch [..................................................]  0.00%
          9000 iter, 15 epoch / 20 epochs
         249.4 iters/sec. Estimated time to finish: 0:00:12.028884.
    [4A[J     total [#####################################.............] 75.83%
    this epoch [########..........................................] 16.67%
          9100 iter, 15 epoch / 20 epochs
        249.64 iters/sec. Estimated time to finish: 0:00:11.616690.
    [4A[J     total [######################################............] 76.67%
    this epoch [################..................................] 33.33%
          9200 iter, 15 epoch / 20 epochs
        249.92 iters/sec. Estimated time to finish: 0:00:11.203418.
    [4A[J     total [######################################............] 77.50%
    this epoch [#########################.........................] 50.00%
          9300 iter, 15 epoch / 20 epochs
        250.17 iters/sec. Estimated time to finish: 0:00:10.792487.
    [4A[J     total [#######################################...........] 78.33%
    this epoch [#################################.................] 66.67%
          9400 iter, 15 epoch / 20 epochs
        250.43 iters/sec. Estimated time to finish: 0:00:10.382150.
    [4A[J     total [#######################################...........] 79.17%
    this epoch [#########################################.........] 83.33%
          9500 iter, 15 epoch / 20 epochs
        250.59 iters/sec. Estimated time to finish: 0:00:09.976316.
    [4A[J16          0.949033       0.9496                    
    [J     total [########################################..........] 80.00%
    this epoch [..................................................]  0.00%
          9600 iter, 16 epoch / 20 epochs
        249.87 iters/sec. Estimated time to finish: 0:00:09.605143.
    [4A[J     total [########################################..........] 80.83%
    this epoch [########..........................................] 16.67%
          9700 iter, 16 epoch / 20 epochs
        250.05 iters/sec. Estimated time to finish: 0:00:09.197988.
    [4A[J     total [########################################..........] 81.67%
    this epoch [################..................................] 33.33%
          9800 iter, 16 epoch / 20 epochs
        250.32 iters/sec. Estimated time to finish: 0:00:08.788854.
    [4A[J     total [#########################################.........] 82.50%
    this epoch [#########################.........................] 50.00%
          9900 iter, 16 epoch / 20 epochs
        250.58 iters/sec. Estimated time to finish: 0:00:08.380646.
    [4A[J     total [#########################################.........] 83.33%
    this epoch [#################################.................] 66.67%
         10000 iter, 16 epoch / 20 epochs
        250.77 iters/sec. Estimated time to finish: 0:00:07.975449.
    [4A[J     total [##########################################........] 84.17%
    this epoch [#########################################.........] 83.33%
         10100 iter, 16 epoch / 20 epochs
        251.01 iters/sec. Estimated time to finish: 0:00:07.569486.
    [4A[J17          0.9507         0.9526                    
    [J     total [##########################################........] 85.00%
    this epoch [..................................................]  0.00%
         10200 iter, 17 epoch / 20 epochs
        250.13 iters/sec. Estimated time to finish: 0:00:07.196375.
    [4A[J     total [##########################################........] 85.83%
    this epoch [########..........................................] 16.67%
         10300 iter, 17 epoch / 20 epochs
        250.15 iters/sec. Estimated time to finish: 0:00:06.795972.
    [4A[J     total [###########################################.......] 86.67%
    this epoch [################..................................] 33.33%
         10400 iter, 17 epoch / 20 epochs
        250.12 iters/sec. Estimated time to finish: 0:00:06.397005.
    [4A[J     total [###########################################.......] 87.50%
    this epoch [#########################.........................] 50.00%
         10500 iter, 17 epoch / 20 epochs
        250.15 iters/sec. Estimated time to finish: 0:00:05.996337.
    [4A[J     total [############################################......] 88.33%
    this epoch [#################################.................] 66.67%
         10600 iter, 17 epoch / 20 epochs
        251.26 iters/sec. Estimated time to finish: 0:00:05.571862.
    [4A[J     total [############################################......] 89.17%
    this epoch [#########################################.........] 83.33%
         10700 iter, 17 epoch / 20 epochs
        251.44 iters/sec. Estimated time to finish: 0:00:05.170228.
    [4A[J18          0.952383       0.9532                    
    [J     total [#############################################.....] 90.00%
    this epoch [..................................................]  0.00%
         10800 iter, 18 epoch / 20 epochs
        250.63 iters/sec. Estimated time to finish: 0:00:04.787898.
    [4A[J     total [#############################################.....] 90.83%
    this epoch [########..........................................] 16.67%
         10900 iter, 18 epoch / 20 epochs
        250.76 iters/sec. Estimated time to finish: 0:00:04.386683.
    [4A[J     total [#############################################.....] 91.67%
    this epoch [################..................................] 33.33%
         11000 iter, 18 epoch / 20 epochs
         250.8 iters/sec. Estimated time to finish: 0:00:03.987294.
    [4A[J     total [##############################################....] 92.50%
    this epoch [#########################.........................] 50.00%
         11100 iter, 18 epoch / 20 epochs
        250.85 iters/sec. Estimated time to finish: 0:00:03.587843.
    [4A[J     total [##############################################....] 93.33%
    this epoch [#################################.................] 66.67%
         11200 iter, 18 epoch / 20 epochs
        251.83 iters/sec. Estimated time to finish: 0:00:03.176797.
    [4A[J     total [###############################################...] 94.17%
    this epoch [#########################################.........] 83.33%
         11300 iter, 18 epoch / 20 epochs
           252 iters/sec. Estimated time to finish: 0:00:02.777783.
    [4A[J19          0.953817       0.953                     
    [J     total [###############################################...] 95.00%
    this epoch [..................................................]  0.00%
         11400 iter, 19 epoch / 20 epochs
        251.32 iters/sec. Estimated time to finish: 0:00:02.387425.
    [4A[J     total [###############################################...] 95.83%
    this epoch [########..........................................] 16.67%
         11500 iter, 19 epoch / 20 epochs
        251.59 iters/sec. Estimated time to finish: 0:00:01.987384.
    [4A[J     total [################################################..] 96.67%
    this epoch [################..................................] 33.33%
         11600 iter, 19 epoch / 20 epochs
        251.86 iters/sec. Estimated time to finish: 0:00:01.588182.
    [4A[J     total [################################################..] 97.50%
    this epoch [#########################.........................] 50.00%
         11700 iter, 19 epoch / 20 epochs
        252.12 iters/sec. Estimated time to finish: 0:00:01.189929.
    [4A[J     total [#################################################.] 98.33%
    this epoch [#################################.................] 66.67%
         11800 iter, 19 epoch / 20 epochs
        253.16 iters/sec. Estimated time to finish: 0:00:00.790023.
    [4A[J     total [#################################################.] 99.17%
    this epoch [#########################################.........] 83.33%
         11900 iter, 19 epoch / 20 epochs
         253.1 iters/sec. Estimated time to finish: 0:00:00.395094.
    [4A[J20          0.95535        0.9551                    
    [J     total [##################################################] 100.00%
    this epoch [..................................................]  0.00%
         12000 iter, 20 epoch / 20 epochs
        252.37 iters/sec. Estimated time to finish: 0:00:00.
    [4A[J

这些扩展执行以下任务：

* Evaluator
在每个epoch 结束时基于测试数据集评估当前模型。它会自动切换到测试模式，因此我们不必为在训练/测试模式（例如，dropout（），BatchNormalization）中表现不同的模式采取任何特殊的功能。

* LogReport
汇总要报告的数值并将其发送到输出目录中的日志文件。

* PrintReport
在LogReport中打印选定的项目。

* ProgressBar
显示进度条。

在chainer.training.extensions模块中实现了许多扩展。其中最重要的一个就是snapshot（），它将训练过程的快照（即Trainer对象）保存到输出目录中的一个文件中。

examples / mnist目录中的示例代码还包含GPU支持，尽管其基本部分与本教程中的代码相同。我们将在后面的章节中回顾如何使用GPU。













