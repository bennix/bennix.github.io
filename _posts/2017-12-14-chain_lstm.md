---
ilayout: post
title: Chainer 递归网络及其计算图
date: 2017-12-14
categories: blog
tags: [Chainer,递归网络及其计算图]
descrption: Chainer 递归网络及其计算图
---



# 递归网络及其计算图

在本节中，您将学习如何编写

* 全后向传播的递归神经网络
* 截断后向传播的递归神经网络
* 占用更少的内存来评估网络

阅读本节后，您将能够：

* 处理可变长度的输入序列
* 前向计算时截断网络上游
* 使用非向后传播模式来防止网络构建

## 递归神经网络

递归神经网络是带有循环的神经网络。它们通常用于从序列输入/输出学习。 给定一个输入流$x_1，x_2,\dots，x_t，\dots$和初始状态$h_0$，一个递归神经网络迭代地更新它的状态$h_t = f(x_t, h_{t-1})$，在某个或每个时间点$t$，它输出$y_t=g(h_t)$。
如果我们沿着时间轴扩展这个过程，除了在网络中重复使用相同的参数之外，它看起来像一个普通的前馈网络。

这里，我们学习如何编写一个简单的一层递归神经网络。任务是语言建模：给定一个有限的单词序列，我们要在而不偷看后续的单词的前提下预测每个位置的下一个单词。 假设有1000种不同的词类型，并且我们使用100维的实数矢量来表示每个词（又称为词嵌入）



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

我们从定义递归神经网络语言模型（RNNLM）作为一个`Chain`开始。我们可以使用 `chainer.links.LSTM` 连接来实现全连接的有状态`LSTM`层。这个连接看起来像一个普通的全连接层。在构建时，将输入和输出大小传递给构造函数：



```python
l = L.LSTM(100, 50)
```

然后，调用这个实例`l(x)`执行LSTM层的一步:


```python
l.reset_state()
x = Variable(np.random.randn(10, 100).astype(np.float32))
y = l(x)
```

在正向计算之前，不要忘记重置LSTM层的内部状态！每个递归层保持其内部状态（即先前调用的输出）。在递归层的第一个应用程序中，您必须重置内部状态。然后，下一个输入可以馈入到LSTM实例：



```python
x2 = Variable(np.random.randn(10, 100).astype(np.float32))
y2 = l(x2)
```

基于这个LSTM连接，让我们把递归网络写成一个新的连接：


```python
class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(1000, 100)  # word embedding
            self.mid = L.LSTM(100, 50)  # the first LSTM layer
            self.out = L.Linear(50, 1000)  # the feed-forward output layer

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y

rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)
```

`EmbedID`是一个词嵌入的连接。它将输入整数转换成相应的固定维嵌入向量。最后一个全连接`out`代表前馈输出层。

RNN连接实现了步进计算。它本身并不处理序列，但是我们可以使用它来处理序列，只需将序列中的项目按顺序直接提供给chain即可。

假设我们有一个单词变量列表x_list。然后，我们可以通过简单的for循环来计算单词序列的损失值。



```python
def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        loss += model(cur_word, next_word)
    return loss
```

当然，累计损失是一个具有完整计算历史的Variable对象。所以我们可以调用它的`backward（）`方法根据模型参数计算总损失的梯度：



```python
# Suppose we have a list of word variables x_list.
rnn.reset_state()
model.cleargrads()
loss = compute_loss(x_list)
loss.backward()
optimizer.update()
```

或者等价地，我们可以使用compute_loss作为损失函数：


```python
rnn.reset_state()
optimizer.update(compute_loss, x_list)
```

## 使用解链截断计算图

从很长的序列学习也是递归网络的典型用例。假设输入和状态序列太长，无法放入内存。在这种情况下，我们经常会将反向传播截断到一个很短的时间范围内。这种技术被称为截断式反向传播。这是启发式的，它使梯度有偏。然而，如果时间范围足够长的话，这种技术在实践中效果很好。

如何在Chainer中实现截断式反向传播？ Chainer有一个聪明的机制来实现截断，称为后向解链。它在`Variable.unchain_backward()`方法中实现。向后解链从`Variable`对象开始，并从Variable中向后剔除计算历史。剔除的变量被自动处理（如果它们没有被任何其他用户对象显式引用）。因此，它们不再是计算历史的一部分，也不再涉及反向传播。

我们来写一个截断式反向传播的例子。这里我们使用与前一小节中相同的网络。假设我们得到一个非常长的序列，并且我们想要每30个时间步骤反向传播被截断。我们可以使用上面定义的模型编写截断反向传播：



```python
loss = 0
count = 0
seqlen = len(x_list[1:])

rnn.reset_state()
for cur_word, next_word in zip(x_list, x_list[1:]):
    loss += model(cur_word, next_word)
    count += 1
    if count % 30 == 0 or count == seqlen:
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
```

状态在 model()中更新，损失累积后存储到损失变量。在每30个步骤中，在累积损失的时候反向传播会发生。然后调用`unchain_backward（）`方法，从累计损失中删除计算历史。注意，模型的最后状态不会丢失，因为RNN实例拥有对它的引用。

截断的反向传播的实现很简单，由于没有复杂的技巧，我们可以将这种方法推广到不同的情况。例如，我们可以很容易地扩展上面的代码，以便在反向计时和截断长度之间使用不同的调度机制。


## 不存储计算历史的网络计算

在计算递归网络时，通常不需要存储计算历史。但是如果需要的话，解除连接使我们能够在有限的内存中遍历无限长度的序列。

作为替代，Chainer提供了不存储计算历史的前向计算的评估模式。这是通过调用`no_backprop_mode（）`上下文来实现的:


```python
with chainer.no_backprop_mode():
    x_list = [Variable(...) for _ in range(100)]  # list of 100 words
    loss = compute_loss(x_list)
```

注意，我们不能在这里调用`loss.backward（）`来计算梯度，因为在no-backprop上下文中创建的变量不会记住计算历史记录。

No-backprop 上下文对于在评估前馈网络时以减少内存占用情况也很有用。

我们可以使用`no_backprop_mode（）`来组合一个固定的特征提取器网络和一个可训练的预测器网络。例如，假设我们想要训练一个前向网络predictor_func，它位于另一个固定的预训练网络fixed_func之上。我们想要训练predictor_func而不存储fixed_func的计算历史。这是简单的通过下面的代码片断（假设`x_data`和`y_data`分别指示输入数据和标签）：


```python
with chainer.no_backprop_mode():
    x = Variable(x_data)
    feat = fixed_func(x)
y = predictor_func(feat)
y.backward()
```

首先，输入变量`x`处于`no-backprop`模式，所以`fixed_func`不记忆计算历史。然后predictor_func以反向传播模式执行，即记忆计算的历史。由于计算的历史只记忆在变量`feat`和`y`之间，所以反向计算停在变量`feat`上。


## 结合训练器

上面的代码是用简单的函数/变量API编写的。当我们编写训练循环时，最好使用Trainer，因为我们可以通过扩展轻松添加功能。

在实施训练器之前，让我们先弄清楚训练的设置。我们这里使用Penn Tree Bank数据集作为句子组。每个句子都表示为一个单词序列。我们将所有句子连接成一个长单词序列，每个句子由一个特殊的词<eos>分隔，代表 `“End of Sequence”`。这个数据集很容易通过`chainer.datasets.get_ptb_words()` 获得。这个函数返回训练，验证和测试数据集，每个数据集都被表示为一个整数的长整数。每个整数代表一个单词ID。

我们的任务是从长词序列中学习循环递归神经网络语言模型。我们用不同的地点的句子来形成小批量数据。这意味着我们保持 `B` 指向序列中不同位置的索引，在每次迭代时从这些索引读取，并且在读取之后递增所有索引。当然，当一个索引到达整个序列的末尾时，我们将索引变回0。




为了实现这个训练过程，我们必须定制训练器的以下组件：

* 迭代器。内置的迭代器不支持从不同位置读取数据，并将它们聚合到一个小批量中。
* 更新函数。 缺省的更新函数不支持截断的BPTT。

当我们编写专用于数据集的数据集迭代器时，数据集实现可以是任意的;即使接口不固定。另一方面，迭代器必须支持Iterator接口。
应实现的关键的方法和属性是 `batch_size`, `epoch`, `epoch_detail`, `is_new_epoch`, `iteration`, `__next__`, 以及`serialize`。 以下是 `examples/ptb` 目录中官方示例的代码。


```python
from __future__ import division

class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.offsets = [i * len(dataset) // batch_size for i in range(batch_size)]
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

train_iter = ParallelSequentialIterator(train, 20)
val_iter = ParallelSequentialIterator(val, 1, repeat=False)
```

虽然代码稍长，但想法很简单。首先，这个迭代器创建指向整个序列中等间隔位置的偏移量。小批量的第i个例子指的是第i个偏移量的序列。迭代器返回当前单词和下一个单词的元组列表。。每个小批量都由标准更新程序中的`concat_examples`函数转换为整型数组的元组（参见上一个教程）。


通过时间的反向传播可以如下实现。


```python
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len):
        super(BPTTUpdater, self).__init__(train_iter, optimizer)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

updater = BPTTUpdater(train_iter, optimizer, bprop_len)  # instantiation
```

在这种情况下，我们更新每`bprop_len`个连续词的参数。`unchain_backward`的调用减少了累计到`LSTM`连接的计算历史。其余的设置训练器的代码几乎和前一个教程中给出的一样。

在本节中，我们演示了如何在`Chainer`中编写循环递归网络，以及如何管理计算历史的一些基本技术（又名计算图）。 `example/ptb`目录中的例子实现了来自Penn Treebank语料库的`LSTM`语言模型的截断反向学习。在下一节中，我们将回顾如何在`Chainer`中使用GPU。

