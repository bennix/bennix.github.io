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


我们经常使用均方误差作为损失函数


```python
from chainer import reporter
class MyMLP(chainer.Chain):
 
    def __init__(self, n_units):
        super(MyMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units)  # n_units -> n_units
            self.l4 = L.Linear(1)    # n_units -> n_out
 
    def __call__(self, *args):
        # Calculate loss
        h = self.forward(*args)
        t = args[1]
        self.loss = F.mean_squared_error(h, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss
 
    def forward(self, *args):
        # Common code for both loss (__call__) and predict
        x = args[0]
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        return h
```

在这种情况下，MyMLP模型将在前向计算中计算y（预测目标），并且在模型的`__call__`函数处计算损失。


## 验证/测试的数据分离

当您下载公开可用的机器学习数据集时，通常将其从开始分离为训练数据和测试数据（有时是验证数据）。

但是，我们的自定义数据集尚未分离。我们可以使用chainer的函数来轻松地分割现有的数据集，其中包括以下功能

* chainer.datasets.split_dataset(dataset, split_at, order=None)
* chainer.datasets.split_dataset_random(dataset, first_size, seed=None)
* chainer.datasets.get_cross_validation_datasets(dataset, n_fold, order=None)
* chainer.datasets.get_cross_validation_datasets_random(dataset, n_fold, seed=None)

有关详细信息，请参阅SubDataset。

这些是有用的分开训练数据和测试数据，例如可以如下使用，


```python
 # Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=13)
```

在这里，我们将数据加载为数据集（它是`DatasetMixin`的子类），使用`chainer.datasets.split_dataset_random`函数将这个数据集分成70%的训练数据和30%的测试数据。

我们也可以指定种子参数来修正随机置换顺序，这对再现实验或者用相同的训练/测试数据集预测代码是有用的。

## 训练代码


```python
from __future__ import print_function
import argparse
 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
import numpy as np
from chainer import reporter
from chainer.dataset import concat_examples
 

```


```python
parser = argparse.ArgumentParser(description='Train custom dataset')
parser.add_argument('--batchsize', '-b', type=int, default=10,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=50,
                    help='Number of units')
args = parser.parse_args(['-g','0'])
```


```python
print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')
```

    GPU: 0
    # unit: 50
    # Minibatch-size: 10
    # epoch: 20
    



```python
# Set up a neural network to train
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
model = MyMLP(args.unit)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

# Setup an optimizer
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)
```


```python
# Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=13)
```


```python
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
#trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

# Plot graph for loss for each epoch
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
else:
    print('Warning: PlotReport is not available in your environment')
# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)

# Run the training
trainer.run()
serializers.save_npz('{}/mymlp.model'.format(args.out), model)
```

    epoch       main/loss   validation/main/loss  elapsed_time
    [J1           8.7217      13.2216               0.264993      
    [J     total [###...............................................]  7.14%
    this epoch [#####################.............................] 42.86%
           100 iter, 1 epoch / 20 epochs
           inf iters/sec. Estimated time to finish: 0:00:00.
    [4A[J2           8.7564      8.27661               0.62847       
    [J     total [#######...........................................] 14.29%
    this epoch [##########################################........] 85.71%
           200 iter, 2 epoch / 20 epochs
        214.07 iters/sec. Estimated time to finish: 0:00:05.605751.
    [4A[J3           8.47132     8.20647               0.99818       
    [J4           8.19539     8.48856               1.37226       
    [J     total [##########........................................] 21.43%
    this epoch [##############....................................] 28.57%
           300 iter, 4 epoch / 20 epochs
         186.1 iters/sec. Estimated time to finish: 0:00:05.910877.
    [4A[J5           8.26764     8.48402               1.73545       
    [J     total [##############....................................] 28.57%
    this epoch [###################################...............] 71.43%
           400 iter, 5 epoch / 20 epochs
        192.58 iters/sec. Estimated time to finish: 0:00:05.192770.
    [4A[J6           8.35916     7.82453               2.1203        
    [J7           8.22192     8.26731               2.47891       
    [J     total [#################.................................] 35.71%
    this epoch [#######...........................................] 14.29%
           500 iter, 7 epoch / 20 epochs
           186 iters/sec. Estimated time to finish: 0:00:04.838621.
    [4A[J8           8.21255     7.90139               2.84666       
    [J     total [#####################.............................] 42.86%
    this epoch [############################......................] 57.14%
           600 iter, 8 epoch / 20 epochs
        185.53 iters/sec. Estimated time to finish: 0:00:04.311946.
    [4A[J9           8.1826      7.86489               3.29141       
    [J10          8.20058     8.18055               3.6595        
    [J     total [#########################.........................] 50.00%
    this epoch [..................................................]  0.00%
           700 iter, 10 epoch / 20 epochs
        182.82 iters/sec. Estimated time to finish: 0:00:03.828946.
    [4A[J11          8.23385     7.83185               4.02586       
    [J     total [############################......................] 57.14%
    this epoch [#####################.............................] 42.86%
           800 iter, 11 epoch / 20 epochs
        185.26 iters/sec. Estimated time to finish: 0:00:03.238664.
    [4A[J12          8.13546     8.0219                4.40651       
    [J     total [################################..................] 64.29%
    this epoch [##########################################........] 85.71%
           900 iter, 12 epoch / 20 epochs
        188.43 iters/sec. Estimated time to finish: 0:00:02.653515.
    [4A[J13          8.1298      7.78307               4.77653       
    [J14          8.26764     7.91379               5.1378        
    [J     total [###################################...............] 71.43%
    this epoch [##############....................................] 28.57%
          1000 iter, 14 epoch / 20 epochs
        185.62 iters/sec. Estimated time to finish: 0:00:02.154961.
    [4A[J15          8.23635     7.92182               5.50792       
    [J     total [#######################################...........] 78.57%
    this epoch [###################################...............] 71.43%
          1100 iter, 15 epoch / 20 epochs
        188.09 iters/sec. Estimated time to finish: 0:00:01.594948.
    [4A[J16          8.27431     7.98348               5.8799        
    [J17          8.16515     7.83324               6.24324       
    [J     total [##########################################........] 85.71%
    this epoch [#######...........................................] 14.29%
          1200 iter, 17 epoch / 20 epochs
        185.75 iters/sec. Estimated time to finish: 0:00:01.076714.
    [4A[J18          8.30931     8.15014               6.6156        
    [J     total [##############################################....] 92.86%
    this epoch [############################......................] 57.14%
          1300 iter, 18 epoch / 20 epochs
        187.82 iters/sec. Estimated time to finish: 0:00:00.532415.
    [4A[J19          8.15276     7.89404               6.98574       
    [J20          8.04605     9.16781               7.36121       
    [J     total [##################################################] 100.00%
    this epoch [..................................................]  0.00%
          1400 iter, 20 epoch / 20 epochs
        186.08 iters/sec. Estimated time to finish: 0:00:00.
    [4A[J


```python
'{}/mymlp.model'.format(args.out)
```




    'result/mymlp.model'



如果我们修改一下MLP的实现，给它加入预测的功能


```python
from chainer.dataset import concat_examples


class MyMLP(chainer.Chain):

    def __init__(self, n_units):
        super(MyMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units)  # n_units -> n_units
            self.l4 = L.Linear(1)    # n_units -> n_out

    def __call__(self, *args):
        # Calculate loss
        h = self.forward(*args)
        t = args[1]
        self.loss = F.mean_squared_error(h, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def forward(self, *args):
        # Common code for both loss (__call__) and predict
        x = args[0]
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

    def predict(self, *args):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self.forward(*args)

    def predict2(self, *args, batchsize=32):
        data = args[0]
        x_list = []
        y_list = []
        t_list = []
        for i in range(0, len(data), batchsize):
            x, t = concat_examples(data[i:i + batchsize])
            y = self.predict(x)
            y_list.append(y.data)
            x_list.append(x)
            t_list.append(t)

        x_array = np.concatenate(x_list)[:, 0]
        y_array = np.concatenate(y_list)[:, 0]
        t_array = np.concatenate(t_list)[:, 0]
        return x_array, y_array, t_array

```

## 预测代码配置

预测阶段与训练阶段相比有一定差异，

- 函数行为

培训阶段和验证/预测阶段的某些功能的预期行为是不同的。例如，F.dropout有望在训练阶段让某个神经单元断线，而最好不要在验证/预测阶段出现断线。这些类型的函数行为是由chainer.config.train配置来处理的。

- 反向传播是没有必要的

当启用反向传播时，模型需要构建需要额外内存的计算图。然而，在验证/预测阶段不需要反向传播，我们可以省略构建计算图来减少内存使用量。

这可以通过`chainer.config.enable_backprop`控制，而`chainer.no_backprop_mode()`函数也是一种方便的方法。


有一个方便的函数concat_examples，用于从数据集中准备小批量。

```
chainer.dataset.concat_examples(batch, device=None, padding=None)
``` 

![](https://bennix.github.io/imgs/concat_examples-700x301.png)

concat_examples 将数据集列表转换为可以输入到神经网络中的每个特征（这里是x和y）的小批量。

通常，当我们通过切片索引访问数据集时，例如`dataset[i：j]`，它会返回一个连续的数据列表。 `concat_examples`分隔数据的每个元素并连接它以生成小批量。


我们再执行一下上面的训练代码：


```python
# Set up a neural network to train
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
model = MyMLP(args.unit)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

# Setup an optimizer
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)
# Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=13)
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
#trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

# Plot graph for loss for each epoch
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
else:
    print('Warning: PlotReport is not available in your environment')
# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)

# Run the training
trainer.run()
serializers.save_npz('{}/mymlp.model'.format(args.out), model)

```

    epoch       main/loss   validation/main/loss  elapsed_time
    [J1           9.02448     8.18393               0.262726      
    [J     total [###...............................................]  7.14%
    this epoch [#####################.............................] 42.86%
           100 iter, 1 epoch / 20 epochs
           inf iters/sec. Estimated time to finish: 0:00:00.
    [4A[J2           8.52984     8.22332               0.629526      
    [J     total [#######...........................................] 14.29%
    this epoch [##########################################........] 85.71%
           200 iter, 2 epoch / 20 epochs
        191.54 iters/sec. Estimated time to finish: 0:00:06.265068.
    [4A[J3           8.3094      8.28372               1.05295       
    [J4           8.25953     7.86636               1.41657       
    [J     total [##########........................................] 21.43%
    this epoch [##############....................................] 28.57%
           300 iter, 4 epoch / 20 epochs
        178.25 iters/sec. Estimated time to finish: 0:00:06.170950.
    [4A[J5           8.0706      7.86111               1.78539       
    [J     total [##############....................................] 28.57%
    this epoch [###################################...............] 71.43%
           400 iter, 5 epoch / 20 epochs
        187.66 iters/sec. Estimated time to finish: 0:00:05.328856.
    [4A[J6           8.09598     7.84718               2.16374       
    [J7           8.25873     7.952                 2.52728       
    [J     total [#################.................................] 35.71%
    this epoch [#######...........................................] 14.29%
           500 iter, 7 epoch / 20 epochs
        181.48 iters/sec. Estimated time to finish: 0:00:04.959360.
    [4A[J8           8.09947     7.87801               2.89913       
    [J     total [#####################.............................] 42.86%
    this epoch [############################......................] 57.14%
           600 iter, 8 epoch / 20 epochs
        187.11 iters/sec. Estimated time to finish: 0:00:04.275658.
    [4A[J9           8.30052     8.12619               3.26968       
    [J10          8.21021     7.86035               3.64373       
    [J     total [#########################.........................] 50.00%
    this epoch [..................................................]  0.00%
           700 iter, 10 epoch / 20 epochs
        184.13 iters/sec. Estimated time to finish: 0:00:03.801619.
    [4A[J11          8.15902     7.88363               4.00784       
    [J     total [############################......................] 57.14%
    this epoch [#####################.............................] 42.86%
           800 iter, 11 epoch / 20 epochs
        186.79 iters/sec. Estimated time to finish: 0:00:03.212135.
    [4A[J12          8.09043     7.81935               4.3803        
    [J     total [################################..................] 64.29%
    this epoch [##########################################........] 85.71%
           900 iter, 12 epoch / 20 epochs
        189.39 iters/sec. Estimated time to finish: 0:00:02.640071.
    [4A[J13          8.23572     7.82124               4.7561        
    [J14          8.10109     7.97537               5.12721       
    [J     total [###################################...............] 71.43%
    this epoch [##############....................................] 28.57%
          1000 iter, 14 epoch / 20 epochs
        186.26 iters/sec. Estimated time to finish: 0:00:02.147575.
    [4A[J15          8.24532     7.8214                5.49437       
    [J     total [#######################################...........] 78.57%
    this epoch [###################################...............] 71.43%
          1100 iter, 15 epoch / 20 epochs
        188.63 iters/sec. Estimated time to finish: 0:00:01.590383.
    [4A[J16          8.07317     7.82089               5.90988       
    [J17          8.19283     7.81849               6.28176       
    [J     total [##########################################........] 85.71%
    this epoch [#######...........................................] 14.29%
          1200 iter, 17 epoch / 20 epochs
        184.39 iters/sec. Estimated time to finish: 0:00:01.084668.
    [4A[J18          8.11496     7.83497               6.66296       
    [J     total [##############################################....] 92.86%
    this epoch [############################......................] 57.14%
          1300 iter, 18 epoch / 20 epochs
        186.32 iters/sec. Estimated time to finish: 0:00:00.536701.
    [4A[J19          8.2032      8.00207               7.03671       
    [J20          8.16395     7.82686               7.41704       
    [J     total [##################################################] 100.00%
    this epoch [..................................................]  0.00%
          1400 iter, 20 epoch / 20 epochs
        184.55 iters/sec. Estimated time to finish: 0:00:00.
    [4A[J
