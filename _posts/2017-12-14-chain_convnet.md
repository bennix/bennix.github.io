---
layout: post
title: Chainer 用于视觉识别任务的卷积网络
date: 2017-12-14
categories: blog
tags: [Chainer,用于视觉识别任务的卷积网络]
description: Chainer 用于视觉识别任务的卷积网络
---


# 用于视觉识别任务的卷积网络

在本节中，您将学习如何编写

* 从Chain继承的具有模型类的小型卷积网络
* 一个用 ChainList 连接的含有构造块的大的卷积网络。

阅读本节后，您将能够：

* 用 Chainer 写你自己的卷积网络


卷积网络（ConvNet）主要由卷积层组成。这种类型的网络通常用于各种视觉识别任务，例如，将手写数字或自然图像分类到给定的对象类别中，从图像中检测对象，并用对象类别标记图像的所有像素（语义分割）等等。



在这样的任务中，典型的ConvNet需要一组形状为（N，C，H，W）的图像，其中$ N $表示小批量图像的数量，$ C $表示这些图像的通道数，$ H $和$ W $分别表示这些图像的高度和宽度。然后，它通常输出一个固定大小的向量作为目标对象类的成员概率。它也可以输出一组特征映射，这些特征映射对于像素标记任务等具有与输入图像相对应的大小。



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

## LeNet5

在这里，我们首先定义Chainer中的LeNet5 [LeCun98]。这是一个5层ConvNet模型，由3个卷积层和2个完全连接的层组成。这是在1998年提出的用于手写数字图像分类。在Chainer中，模型可以写成如下：



```python
class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))
```

编写网络的典型方法是创建一个从Chain类继承的新类。当以这种方式定义模型时，通常将所有具有可训练参数的网络层通过分配Link的对象作为属性注册到模型中。

模型类在前向和反向计算之前被实例化。`__call __（）`通常在模型类中定义，用来简单地通过调用模型对象来提供输入图像和标签矢量。 该方法执行模型的前向计算。`Chainer`使用功能强大的`autograd`系统来处理用`Function`和`Link`（实际上是一个`Link`调用其内部的相应函数）编写的任何计算图，这样您就不需要在模型中明确写入用于反向计算的代码。 只准备数据，然后把它交给模型。工作过程就是从前向计算求出 `Variable` 的输出而通过 `backward()` 计算 `autograd`. 在上面的模型中，`__call __（）`在末尾有一个`if`语句，通过`Chainer`的运行模式（即是否是训练模式）来切换它的行为。 `Chainer`将运行模式作为全局变量`chainer.config.train`提供。 当它处于训练模式时，`__call __（）`返回最后一层的输出值，以便稍后计算损失，否则通过计算`softmax（）`来返回预测结果。

如果您不想多次写入conv1和其他图层，也可以像这样编写模型:


```python
class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        net = [('conv1', L.Convolution2D(1, 6, 5, 1))]
        net += [('_sigm1', F.Sigmoid())]
        net += [('_mpool1', F.MaxPooling2D(2, 2))]
        net += [('conv2', L.Convolution2D(6, 16, 5, 1))]
        net += [('_sigm2', F.Sigmoid())]
        net += [('_mpool2', F.MaxPooling2D(2, 2))]
        net += [('conv3', L.Convolution2D(16, 120, 4, 1))]
        net += [('_sigm3', F.Sigmoid())]
        net += [('_mpool3', F.MaxPooling2D(2, 2))]
        net += [('fc4', L.Linear(None, 84))]
        net += [('_sigm4', F.Sigmoid())]
        net += [('fc5', L.Linear(84, 10))]
        net += [('_sigm5', F.Sigmoid())]
        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
        self.forward = net

    def __call__(self, x):
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)
        if chainer.config.train:
            return x
        return F.softmax(x)
```

此代码在调用其超类的构造函数之后创建所有`Link`和`Function`的列表。然后当元素的名字不以_字符开始时，列表的元素被注册到这个模型中作为可训练层。这个操作可以用许多其他方式自由地替换，因为这些名字只是设计用来仅仅从列表网络中方便地选择`Link`。`Function` 没有任何可训练的参数，所以我们不能将它注册到模型中，但是我们要使用 `Function` 来构造一个前向路径。列表网络以引用`__call __（）`被存储为一个前向网络。在`__call __（）`中，它依次从`self.forward`中检索网络中的所有网络层，而不管它是什么类型的对象（`Link` 或 `Function`），并将输入变量或上一层的中间输出提供给当前层。`__call __（）`的最后部分与前一种方式相同用来进行训练/推理模式切换。


## 计算损失的方法

当用标号向量t训练模型时，应该使用模型的输出来计算损失。有几种计算损失的方法：



```python
model = LeNet5()

# Input data and label
x = np.random.rand(32, 1, 28, 28).astype(np.float32)
t = np.random.randint(0, 10, size=(32,)).astype(np.int32)

# Forward computation
y = model(x)

# Loss calculation
loss = F.softmax_cross_entropy(y, t)
```

这是从模型的输出中计算损失值的主要方法。另一方面，通过用从`Chain`继承的类包装模型对象（`Chain`或`ChainList`对象），计算损失可以被包括在模型本身中。输出`Chain`应该采用上面定义的模型，并用`init_scope（）`注册。`Chain` 实际上是从 `Link` 继承的，所以 `Chain` 本身也可以被注册为可连接到另一个`Chain`的可训练链接。实际上，`Classifier` 类包装模型并且添加损失计算到模型已经存在。实际上，已经有一个`Classifier`类可以用来包装模型，并且包含了损失计算。它可以像这样使用：



```python
model = L.Classifier(LeNet5())

# Foward & Loss calculation
loss = model(x, t)
```

该类将模型对象作为输入参数，并将其作为训练参数注册到 `predictor`属性。如上所示，返回的对象可以被调用为一个函数，在这个函数中我们传入`x`和`t`作为输入参数，并返回所产生的损失值（记做一个变量）。

查看分类器的详细实现：chainer.links.Classifier 并通过查看源代码来检查实现。

从上面的例子中，我们可以看到，Chainer提供了以多种不同方式编写我们神经网络的灵活性。这种灵活性旨在使用户能够直观地设计新的和复杂的模型。

## VGG16

接下来，让我们在Chainer中写一些更大的模型。当你写一个由多个构造块网络组成的大型网络时，ChainList是有用的。首先，让我们来看看如何编写一个VGG16 [Simonyan14]模型。


```python
class VGG16(chainer.ChainList):
    def __init__(self):
        super(VGG16, self).__init__(
            VGGBlock(64),
            VGGBlock(128),
            VGGBlock(256, 3),
            VGGBlock(512, 3),
            VGGBlock(512, 3, True))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        if chainer.config.train:
            return x
        return F.softmax(x)


class VGGBlock(chainer.Chain):
    def __init__(self, n_channels, n_convs=2, fc=False):
        w = chainer.initializers.HeNormal()
        super(VGGBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)
            if n_convs == 3:
                self.conv3 = L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w)
            if fc:
                self.fc4 = L.Linear(None, 4096, initialW=w)
                self.fc5 = L.Linear(4096, 4096, initialW=w)
                self.fc6 = L.Linear(4096, 1000, initialW=w)

        self.n_convs = n_convs
        self.fc = fc

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        if self.n_convs == 3:
            h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, 2)
        if self.fc:
            h = F.dropout(F.relu(self.fc4(h)))
            h = F.dropout(F.relu(self.fc5(h)))
            h = self.fc6(h)
        return h
```

这就是VGG16的实现。VGG16是在ILSVRC 2014上荣获分类+定位任务第一名的模型，从此成为应对许多不同任务的标准模型之一的预训练模型。
它有16层，所以它被称为“VGG-16”，但我们可以不用单独写所有的层。由于这个模型由几个具有相同体系结构的构造块组成，我们可以通过重用构造块定义来构建整个网络。网络的每个部分由2或3个卷积层和激活函数（`relu（）`）组成，以及`max_pooling_2d（）`操作层所组成。在上面的示例代码中，该块被写为`VGGBlock`。而整个网络只是依次调用这个块。


## ResNet152

ResNet如何？ ResNet`[He16]`在第二年的ILSVRC进来。这是一个比VGG16更深的模型，有152层。这听起来非常费力，但可以像VGG16一样实施。换句话说，这很容易。实现ResNet-152的一个可能的方法是：



```python
class ResNet152(chainer.Chain):
    def __init__(self, n_blocks=[3, 8, 36, 3]):
        w = chainer.initializers.HeNormal()
        super(ResNet152, self).__init__(
            conv1=L.Convolution2D(
                None, 64, 7, 2, 3, initialW=w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=ResBlock(n_blocks[0], 64, 64, 256, 1),
            res3=ResBlock(n_blocks[1], 256, 128, 512),
            res4=ResBlock(n_blocks[2], 512, 256, 1024),
            res5=ResBlock(n_blocks[3], 1024, 512, 2048),
            fc6=L.Linear(2048, 1000))

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2, 2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, h.shape[2:], stride=1)
        h = self.fc6(h)
        if chainer.config.train:
            return h
        return F.softmax(h)


class ResBlock(chainer.ChainList):
    def __init__(self, n_layers, n_in, n_mid, n_out, stride=2):
        w = chainer.initializers.HeNormal()
        super(ResBlock, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
        for _ in range(n_layers - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, stride=1, proj=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1x1a = L.Convolution2D(
                n_in, n_mid, 1, stride, 0, initialW=w, nobias=True)
            self.conv3x3b = L.Convolution2D(
                n_mid, n_mid, 3, 1, 1, initialW=w, nobias=True)
            self.conv1x1c = L.Convolution2D(
                n_mid, n_out, 1, 1, 0, initialW=w, nobias=True)
            self.bn_a = L.BatchNormalization(n_mid)
            self.bn_b = L.BatchNormalization(n_mid)
            self.bn_c = L.BatchNormalization(n_out)
            if proj:
                self.conv1x1r = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, initialW=w, nobias=True)
                self.bn_r = L.BatchNormalization(n_out)
        self.proj = proj

    def __call__(self, x):
        h = F.relu(self.bn_a(self.conv1x1a(x)))
        h = F.relu(self.bn_b(self.conv3x3b(h)))
        h = self.bn_c(self.conv1x1c(h))
        if self.proj:
            x = self.bn_r(self.conv1x1r(x))
        return F.relu(h + x)
```

在`BottleNeck`类中，根据提供给初始化程序的`proj`参数的值，它将有条件地计算卷积层`conv1x1r`，这将扩展输入`x`的通道数目，使其等于`conv1x1c`的输出通道数，然后是最后的`ReLU`层之前的批量标准化层。以这种方式编写构建块可以提高一个类的可重用性。它不仅通过标志切换`__class __（）`中的行为，还切换参数注册。在这种情况下，当`proj`为`False`时，`BottleNeck`不具有`conv1x1r`和`bn_r`层，因此与注册两者的情况相比，内存使用效率会更高，如果`proj`为`False`，则会忽略它们。

我们能够轻松地使用嵌套`Chain`和`ChainList`作为顺序部分来编写复杂和非常深的模型。

## 使用预训练的模型

上面介绍了编写模型的各种方法。事实证明，VGG16和ResNet作为多种任务的一般特征提取器非常有用，包括但不限于图像分类。因此，Chainer通过一个简单的API为您提供预训练的VGG16和ResNet-50/101/152模型。您可以如下使用这些模型：



```python
from chainer.links import VGG16Layers

model = VGG16Layers()
```

    Downloading from http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel...
    Now loading caffemodel (usually it may take few minutes)


当VGG16Layers被实例化时，预先训练的参数将自动从作者的服务器下载。所以，你可以立即开始使用VGG16与预先训练的权重作为一个很好的图像特征提取。在`chainer.links.VGG16Layers`看到这个模型的细节。

在`ResNet`模型的情况下，有三种不同的层数。我们有`chainer.links.ResNet50`，`chainer.links.ResNet101`和`chainer.links.ResNet152`模型，具有简单的参数加载功能。ResNet的预训练的参数不可直接下载，因此您需要首先从作者的网页下载权值，然后将其放置在目录`$CHAINER_DATSET_ROOT/pfnet/chainer/models`或您最喜爱的地方。准备工作完成后，用法与VGG16相同：



```python
from chainer.links import ResNet152Layers

model = ResNet152Layers()
```

请查看chainer.links.ResNet50的使用细节以及如何为ResNet准备预先训练的权重。

## 参考文献

`[LeCun98]`   Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324, 1998.

`[Simonyan14]`    Simonyan, K. and Zisserman, A., Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556, 2014.

`[He16]`  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
