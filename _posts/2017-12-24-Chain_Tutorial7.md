---
ilayout: post
title: Chainer 入门教程（7）数据集模块介绍
date: 2017-12-24
categories: blog
tags: [Chainer,入门教程（7), 数据集模块介绍]
descrption: Chainer 入门教程（7）数据集模块介绍
---

# 数据集模块介绍


```python
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import chainer.dataset
import chainer.datasets
```

## 内建的数据模块

一些数据集格式已经在chainer.datasets中实现，例如TupleDataset


```python
from chainer.datasets import TupleDataset

x = np.arange(10)
t = x * x

data = TupleDataset(x, t)

print('data type: {}, len: {}'.format(type(data), len(data)))
```

    data type: <class 'chainer.datasets.tuple_dataset.TupleDataset'>, len: 10


第`i`个数据可以通过`data[i]`访问，是一个元组($x_i$, $t_i$, ...)


```python
# get forth data -> x=3, t=9
data[3]
```




    (3, 9)



### 切片访问

当通过切片索引访问TupleDataset时，例如`data[i:j]`, 返回一个元组列表 $[(x_i, t_i), ..., (x_{j-1}, t_{j-1})]$



```python
# Get 1st, 2nd, 3rd data at the same time.
examples = data[0:4]

print(examples)
print('examples type: {}, len: {}'
      .format(type(examples), len(examples)))
```

    [(0, 0), (1, 1), (2, 4), (3, 9)]
    examples type: <class 'list'>, len: 4


要将示例转换为小批量格式，可以在chainer.dataset中使用concat_examples函数。返回的数值格式是 
`([x_array], [t array], ...)`



```python
from chainer.dataset import concat_examples

data_minibatch = concat_examples(examples)

#print(data_minibatch)
#print('data_minibatch type: {}, len: {}'
#      .format(type(data_minibatch), len(data_minibatch)))

x_minibatch, t_minibatch = data_minibatch
# Now it is array format, which has shape
print('x_minibatch = {}, type: {}, shape: {}'.format(x_minibatch, type(x_minibatch), x_minibatch.shape))
print('t_minibatch = {}, type: {}, shape: {}'.format(t_minibatch, type(t_minibatch), t_minibatch.shape))
```

    x_minibatch = [0 1 2 3], type: <class 'numpy.ndarray'>, shape: (4,)
    t_minibatch = [0 1 4 9], type: <class 'numpy.ndarray'>, shape: (4,)


## DictDataset


```python
from chainer.datasets import DictDataset

x = np.arange(10)
t = x * x

# To construct `DictDataset`, you can specify each key-value pair by passing "key=value" in kwargs.
data = DictDataset(x=x, t=t)

print('data type: {}, len: {}'.format(type(data), len(data)))
```

    data type: <class 'chainer.datasets.dict_dataset.DictDataset'>, len: 10



```python
# Get 3rd data at the same time.
example = data[2]
          
print(example)
print('examples type: {}, len: {}'
      .format(type(example), len(example)))

# You can access each value via key
print('x: {}, t: {}'.format(example['x'], example['t']))
```

    {'x': 2, 't': 4}
    examples type: <class 'dict'>, len: 2
    x: 2, t: 4


## ImageDataset

这是图像数据集的实用工具类。如果数据集的数量变得非常大（例如ImageNet数据集），则不像CIFAR-10或CIFAR-100那样将所有图像加载到内存中。在这种情况下，可以使用ImageDataset类在每次创建小批量时从外存储器（例如硬盘）中打开图像。

>ImageDataset 将只下载图像，如果您需要另一个标签信息（例如，如果您正在处理图像分类任务），请使用LabeledImageDataset。

您需要创建一个文本文件，例如名叫`images.dat`其中包含要使用ImageDataset的图像路径列表。有关路径文本文件的外观，请参阅如下

```
cute-animal-degu-octodon-71487.jpeg
guinea-pig-pet-nager-rodent-47365.jpeg
kittens-cat-cat-puppy-rush-45170.jpeg
kitty-cat-kitten-pet-45201.jpeg
pexels-photo-96938.jpeg
pexels-photo-126407.jpeg
pexels-photo-206931.jpeg
pexels-photo-208845.jpeg
pexels-photo-209079.jpeg
rat-pets-eat-51340.jpeg
```


```python
import os

from chainer.datasets import ImageDataset

# print('Current direcotory: ', os.path.abspath(os.curdir))

filepath = './data/images.dat'
image_dataset = ImageDataset(filepath, root='./data/images')

print('image_dataset type: {}, len: {}'.format(type(image_dataset), len(image_dataset)))
```

    image_dataset type: <class 'chainer.datasets.image_dataset.ImageDataset'>, len: 10


我们已经创建了上面的`image_dataset`，但是图像还没有扩展到内存中。
每次通过索引访问时，图像数据都会从存储器加载到内存中，以便高效地使用内存。


```python
# Access i-th image by image_dataset[i].
# image data is loaded here. for only 0-th image.
img = image_dataset[0]

# img is numpy array, already aligned as (channels, height, width), 
# which is the standard shape format to feed into convolutional layer.
print('img', type(img), img.shape)
```

    img <class 'numpy.ndarray'> (3, 426, 640)


## LabeledImageDataset

这是图像数据集的应用工具类。它与ImageDataset类似，允许在运行时将图像文件从存储器加载到内存中。不同之处在于它包含了标签信息，通常用于图像分类任务。您需要创建一个文本文件，其中包含要使用LabeledImageDataset的图像路径和标签列表。
具体参见如下：

```
cute-animal-degu-octodon-71487.jpeg 0
guinea-pig-pet-nager-rodent-47365.jpeg 0
kittens-cat-cat-puppy-rush-45170.jpeg 1
kitty-cat-kitten-pet-45201.jpeg 1
pexels-photo-96938.jpeg 1
pexels-photo-126407.jpeg 1
pexels-photo-206931.jpeg 0
pexels-photo-208845.jpeg 1
pexels-photo-209079.jpeg 0
rat-pets-eat-51340.jpeg 0
```


```python
import os

from chainer.datasets import LabeledImageDataset

# print('Current direcotory: ', os.path.abspath(os.curdir))

filepath = './data/images_labels.dat'
labeled_image_dataset = LabeledImageDataset(filepath, root='./data/images')

print('labeled_image_dataset type: {}, len: {}'.format(type(labeled_image_dataset), len(labeled_image_dataset)))
```

    labeled_image_dataset type: <class 'chainer.datasets.image_dataset.LabeledImageDataset'>, len: 10


我们已经创建了上面的labeled_image_dataset，但是图像还没有扩展到内存中。 每次通过索引访问时，图像数据都会从存储器加载到内存中，以便高效地使用内存。


```python
# Access i-th image and label by image_dataset[i].
# image data is loaded here. for only 0-th image.
img, label = labeled_image_dataset[0]

print('img', type(img), img.shape)
print('label', type(label), label)
```

    img <class 'numpy.ndarray'> (3, 426, 640)
    label <class 'numpy.ndarray'> 0


# 使用DatasetMixin从您自己的数据创建数据集类

在前面的章节中，我们学习了如何使用MNIST手写数字数据集来训练深度神经网络。但是，MNIST数据集由chainer实用程序库准备，您现在可能想知道如何使用自己的数据进行回归/分类任务时准备相应的数据集。

`Chainer`提供了`DatasetMixin`类来让你定义你自己的数据集类

## 准备数据

在本次任务中，我们尝试一个非常简单的回归任务。数据集可以由下面代码生成


```python
import os
import numpy as np
import pandas as pd

DATA_DIR = 'data'


def black_box_fn(x_data):
    return np.sin(x_data) + np.random.normal(0, 0.1, x_data.shape)
```


```python
os.mkdir(DATA_DIR)

x = np.arange(-5, 5, 0.01)
t = black_box_fn(x)
df = pd.DataFrame({'x': x, 't': t}, columns={'x', 't'})
df.to_csv(os.path.join(DATA_DIR, 'my_data.csv'), index=False)

```

以上代码创建一个名为`data/my_data.csv`的非常简单的csv文件，列名称为`x`和`t`。 `x`表示输入值，`t`表示预测的目标值。

我采用简单的sin函数和一点点高斯噪声从`x`生成`t`。 （你可以尝试修改black_box_fn函数来改变函数来估计。

我们的任务是获得这个`black_box_fn`的回归模型。

## 将MyDataset定义为DatasetMixin的子类

现在你有了自己的数据，我们通过继承chainer提供的DatasetMixin类来定义数据集类。

实现

我们通常实现以下3个函数

* `__init__(self, *args)`
编写初始化代码。

* `__len__(self)`
训练器模块（迭代器）访问此属性来计算每个epoch中训练的进度。

* `get_examples(self, i)`
返回第i个数据


```python
import numpy as np
import pandas as pd
 
import chainer
 
 
class MyDataset(chainer.dataset.DatasetMixin):
 
    def __init__(self, filepath, debug=False):
        self.debug = debug
        # Load the data in initialization
        df = pd.read_csv(filepath)
        self.data = df.values.astype(np.float32)
        if self.debug:
            print('[DEBUG] data: \n{}'.format(self.data))
 
    def __len__(self):
        """return length of this dataset"""
        return len(self.data)
 
    def get_example(self, i):
        """Return i-th data"""
        x, t = self.data[i]
        return [x], [t]

```

最重要的部分是重载函数，`get_example（self，i）`这个函数实现用来返回第i个数据。

我们不需要关心小批量数据的连接问题，迭代器会处理这些东西。你只需要准备一个数据集来返回第i个数据。

上面的代码工作如下，
1. 在初始化代码的`__init__`函数中加载准备好的数据`data/my_data.csv`（设置为`filepath`），并将扩展数组（严格来说，`pandas.DataFrame`类）设置为`self.data`。

2. 返回第i个数据xi和ti作为`get_example(self，i)`中大小为1的向量。


## 它是如何工作的

这个想法很简单。您可以使用`MyDataset()`实例化数据集，然后通过`dataset[i]`访问第i个数据。

也可以通过切片或一维矢量进行访问 `dataset[i：j]`从而返回`[dataset[i]，dataset[i + 1]，...，dataset[j-1]]`。




```python
dataset = MyDataset('data/my_data.csv', debug=True)

print('Access by index dataset[1] = ', dataset[1])
print('Access by slice dataset[:3] = ', dataset[:3])
print('Access by list dataset[[3, 5]] = ', dataset[[3, 5]])
index = np.arange(3)
print('Access by numpy array dataset[[0, 1, 2]] = ', dataset[index])
# Randomly take 3 data
index = np.random.permutation(len(dataset))[:3]
print('dataset[{}] = {}'.format(index, dataset[index]))
```

    [DEBUG] data: 
    [[ 0.95193064 -5.        ]
     [ 0.97486413 -4.98999977]
     [ 1.05177033 -4.98000002]
     ..., 
     [-1.08878708  4.96999979]
     [-0.98387295  4.98000002]
     [-0.89990532  4.98999977]]
    Access by index dataset[1] =  ([0.97486413], [-4.9899998])
    Access by slice dataset[:3] =  [([0.95193064], [-5.0]), ([0.97486413], [-4.9899998]), ([1.0517703], [-4.98])]
    Access by list dataset[[3, 5]] =  [([1.0441649], [-4.9699998]), ([0.87154579], [-4.9499998])]
    Access by numpy array dataset[[0, 1, 2]] =  [([0.95193064], [-5.0]), ([0.97486413], [-4.9899998]), ([1.0517703], [-4.98])]
    dataset[[834 666 533]] = [([-0.241432], [3.3399999]), ([1.1102532], [1.66]), ([0.29236495], [0.33000001])]


## 数据集灵活性 - 来自存储的动态加载，预处理，数据增强

DatasetMixin类的好处是它的灵活性。基本上你可以在`get_example`函数中实现任何东西，每当我们用`data[i]`访问数据的时候，都会调用`get_example`。

1. 数据增强

这意味着我们可以编写动态的预处理。特别是在图像处理领域，数据增强是众所周知的重要的技术，以避免过度拟合，并获得高的验证分数。



```python
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label
```

2. 从存储动态加载

如果您处理的数据量非常大，并且所有数据都不能立即在内存中扩展，那么最好的做法是每次必要时（在创建小批量时）加载数据。

我们可以用`DatasetMixin`类轻松实现这个过程。简单地说，你可以在`get_example`函数中写入加载代码，从存储中加载第`i`个数据！


## TransformDataset

可以使用变换数据集从现有数据集创建/修改数据集。新的（修改的）数据集可以通过`TransformDataset(original_data，transform_function)`创建。让我们看一个具体的例子，通过添加一个小的噪音，从原始的元组数据集创建新的数据集。



```python
from chainer.datasets import TransformDataset

x = np.arange(10)
t = x * x - x

original_dataset = TupleDataset(x, t)

def transform_function(in_data):
    x_i, t_i = in_data
    new_t_i = t_i + np.random.normal(0, 0.1)
    return x_i, new_t_i

transformed_dataset = TransformDataset(original_dataset, transform_function)

```


```python
original_dataset[:3]
```




    [(0, 0), (1, 0), (2, 2)]




```python
# Now Gaussian noise is added (in transform_function) to the original_dataset.
transformed_dataset[:3]
```




    [(0, -0.10313827057174003), (1, 0.13332423623441678), (2, 2.0453149576361631)]


