---
ilayout: post
title: Chainer 入门教程（5）重构基于MNIST的训练
date: 2017-12-21
categories: blog
tags: [Chainer,入门教程（5), 重构基于MNIST的训练]
descrption: Chainer 入门教程（5）重构基于MNIST的训练
---
# 重构基于MNIST的训练

上一节，我们学习了MNIST训练代码的最小实现。现在，让我们重构代码。

首先，我们先定义多层感知机


```python
import chainer
import chainer.functions as F
import chainer.links as L


# Network definition Chainer v2
# 1. `init_scope()` is used to initialize links for IDE friendly design.
# 2. input size of Linear layer can be omitted
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # input size of each layer will be inferred when omitted
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

在命令行调用Python程序的时候，有的时候需要对于命令行参数进行解析。argparse用于提供可配置的脚本代码。执行代码时用户可以对于运行的脚本进行设置。这是在训练代码之前增加的命令行解析部分。


```python
import argparse
```


```python
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result/2',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=50,
                    help='Number of units')
```




    _StoreAction(option_strings=['--unit', '-u'], dest='unit', nargs=None, const=None, default=50, type=<class 'int'>, choices=None, help='Number of units', metavar=None)




```python
args = parser.parse_args(['-g','0'])
```


```python
args
```




    Namespace(batchsize=100, epoch=20, gpu=0, initmodel='', out='result/2', resume='', unit=50)



在命令行的时候，你可能需要键入
```
python <程序名>.py -g 0
```
达到和上面一样的效果

我们试着列出一些参数配置：


```python
print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')
```

    GPU: 0
    # unit: 50
    # Minibatch-size: 100
    # epoch: 20
    


import time
import os

import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import computational_graph
from chainer import serializers

## 建立一个神经网络来训练

分类器“包装”预测器输出y来计算y和实际目标t之间的损失。
```
classifier_model = L.Classifier(model)
```
而
```
optimizer.update(classifier_model, x, t)
```
它在内部调用`classifier_model(x，t)`，计算损失并通过反向传播更新内部参数。


```python
model = MLP(args.unit, 10)
classifier_model = L.Classifier(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    classifier_model.to_gpu()  # Copy the model to the GPU
xp = np if args.gpu < 0 else cuda.cupy
```

## 设定优化器
    


```python
optimizer = chainer.optimizers.Adam()
optimizer.setup(classifier_model)
```

## 载入 MNIST 数据集


```python
train, test = chainer.datasets.get_mnist()

batchsize = args.batchsize
n_epoch = args.epoch
N = len(train)       # training data size
N_test = len(test)  # test data size
```

## 是初始化还是继续训练


```python
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, classifier_model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

if not os.path.exists(args.out):
    os.makedirs(args.out)
```


```python
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
        t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(classifier_model, x, t)

        if epoch == 1 and i == 0:
            with open('{}/graph.dot'.format(args.out), 'w') as o:
                g = computational_graph.build_computational_graph(
                    (classifier_model.loss,))
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(classifier_model.loss.data) * len(t.data)
        sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        index = np.asarray(list(range(i, i + batchsize)))
        x = chainer.Variable(xp.asarray(test[index][0]))
        t = chainer.Variable(xp.asarray(test[index][1]))
        with chainer.no_backprop_mode():
            # When back propagation is not necessary,
            # we can omit constructing graph path for better performance.
            # `no_backprop_mode()` is introduced from chainer v2,
            # while `volatile` flag was used in chainer v1.
            loss = classifier_model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
```

    epoch 1
    graph generated
    train mean loss=0.40927788959195216, accuracy=0.8873833338481685, throughput=16387.236906039903 images/sec
    test  mean loss=0.20570483731105924, accuracy=0.9385000044107437
    epoch 2
    train mean loss=0.17687243939066927, accuracy=0.9487666701277097, throughput=35086.18244998239 images/sec
    test  mean loss=0.14263299162266777, accuracy=0.9556000012159348
    epoch 3
    train mean loss=0.133713704533875, accuracy=0.9605500031510988, throughput=34922.40306002655 images/sec
    test  mean loss=0.12005061563337222, accuracy=0.9645000034570694
    epoch 4
    train mean loss=0.10752507843542844, accuracy=0.9683333398898443, throughput=35508.567916431384 images/sec
    test  mean loss=0.10576737503346521, accuracy=0.966400004029274
    epoch 5
    train mean loss=0.09149456447921693, accuracy=0.9722500093777975, throughput=35379.674104332385 images/sec
    test  mean loss=0.09852059424272738, accuracy=0.9703000050783157
    epoch 6
    train mean loss=0.07771000775353362, accuracy=0.9767500099539757, throughput=35471.036130503766 images/sec
    test  mean loss=0.0970954722361057, accuracy=0.9697000050544738
    epoch 7
    train mean loss=0.06814443566370755, accuracy=0.9790333433945974, throughput=35405.85116523151 images/sec
    test  mean loss=0.09697691924637183, accuracy=0.9701000052690506
    epoch 8
    train mean loss=0.06006682687050973, accuracy=0.9819833445549011, throughput=35492.58255404526 images/sec
    test  mean loss=0.09864984683401418, accuracy=0.9715000063180923
    epoch 9
    train mean loss=0.053054940653188776, accuracy=0.9839500105381012, throughput=35159.76597062436 images/sec
    test  mean loss=0.09695819896340253, accuracy=0.9709000051021576
    epoch 10
    train mean loss=0.04789321313224112, accuracy=0.9848833438754082, throughput=35115.23881696846 images/sec
    test  mean loss=0.09009438048000447, accuracy=0.9732000052928924
    epoch 11
    train mean loss=0.04270464556563335, accuracy=0.9869000096122423, throughput=34519.5195215791 images/sec
    test  mean loss=0.0951644035075151, accuracy=0.9730000048875809
    epoch 12
    train mean loss=0.03886480388231575, accuracy=0.9876500088969866, throughput=34869.18146081638 images/sec
    test  mean loss=0.10312971735525935, accuracy=0.9725000059604645
    epoch 13
    train mean loss=0.03562387730111368, accuracy=0.9888500086466472, throughput=34386.94451155057 images/sec
    test  mean loss=0.09578683844978514, accuracy=0.9747000056505203
    epoch 14
    train mean loss=0.03129998438099089, accuracy=0.990000008046627, throughput=29898.38919905711 images/sec
    test  mean loss=0.09477038305081806, accuracy=0.9758000046014785
    epoch 15
    train mean loss=0.028575848049173753, accuracy=0.9909500076373419, throughput=30329.13475673733 images/sec
    test  mean loss=0.10486655341068399, accuracy=0.9723000019788742
    epoch 16
    train mean loss=0.027250811755657196, accuracy=0.9914333405097325, throughput=28889.929471389576 images/sec
    test  mean loss=0.10126175733705167, accuracy=0.9752000063657761
    epoch 17
    train mean loss=0.025719110272087467, accuracy=0.9915500070651372, throughput=33927.91958625067 images/sec
    test  mean loss=0.11167271406433428, accuracy=0.9723000037670135
    epoch 18
    train mean loss=0.02003041707192703, accuracy=0.9938000057140987, throughput=31177.527938994397 images/sec
    test  mean loss=0.10606268948264187, accuracy=0.9752000063657761
    epoch 19
    train mean loss=0.021392025589351153, accuracy=0.9933666728933652, throughput=31570.574954546257 images/sec
    test  mean loss=0.11467492570336617, accuracy=0.9716000038385392
    epoch 20
    train mean loss=0.017389228352403734, accuracy=0.9946500050028165, throughput=35198.87165889165 images/sec
    test  mean loss=0.11811088476270924, accuracy=0.9731000065803528


## 保存模型和优化器    


```python
print('save the model')
serializers.save_npz('{}/classifier.model'.format(args.out), classifier_model)
serializers.save_npz('{}/mlp.model'.format(args.out), model)
print('save the optimizer')
serializers.save_npz('{}/mlp.state'.format(args.out), optimizer)
```

    save the model
    save the optimizer


# 训练标志框架

在chainer v2中，引入了全局标志chainer.config.train。在同一个模型中，训练阶段的损失函数和推断截断的预测代码同时存在，并且他们的行为由**训练标志**控制。




```python
# Network definition
class MLP(chainer.Chain):
 
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out
 
        # Define train flag
        self.train = True
 
    def __call__(self, x, t=None):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        if self.train:
            # return loss in training phase
            #y = self.predictor(x)
            self.loss = F.softmax_cross_entropy(y, t)
            self.accuracy = F.accuracy(y, t)
            return self.loss
        else:
            # return y in predict/inference phase
            return y
```

默认情况下，self.train = True，并且该模型将计算损失，以便优化器可以更新其内部参数。

为了进行预测，我们可以设置训练标志为False，



```python
model.train = False
y = model(x)
```

## 对比

Predictor – Classifier框架有一个优点，即分类器模块可以是独立的，并且是可重用的。但是，如果损失计算复杂，则很难应用这个框架。

在训练标志框架中，训练损失计算和预测计算可以是独立的。您可以实施任何损失计算，即使损失计算与预测计算有很大不同。

基本上，如果损失函数是典型的，则可以使用Predictor - Classifier框架。否则使用训练标志框架。

