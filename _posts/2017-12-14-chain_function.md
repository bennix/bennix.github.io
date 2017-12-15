---
ilayout: post
title: Chainer 定义你自己的函数
date: 2017-12-15
categories: blog
tags: [Chainer,定义你自己的函数]
descrption: Chainer 定义你自己的函数
---

# 定义你自己的函数

在本节中，您将了解以下内容：

* 如何在变量上定义一个函数
* 编写使用GPU函数的有用工具
* 如何测试函数定义

阅读本节后，您将能够：

* 写你自己的函数
* 在函数定义中定义简单的内核


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

## 可微函数

Chainer在函数模块中提供了的一系列函数。它涵盖了深度学习中的典型用例，因此可以实施许多现有的方法。另一方面，深度学习正在迅速发展，我们无法涵盖所有未知的架构。所以了解如何定义自己的函数是很重要的。

首先，假设我们要定义一个元素级别的函数 $f(x,y,z)=x∗y+z$。 虽然可以使用 * 和 + 函数的组合来实现这个公式，但将其定义为单个函数可能会减少内存消耗，所以它不仅仅是一个玩具的例子。在这里我们称这个函数为MulAdd。

让我们开始在CPU上定义MulAdd。任何函数都必须继承Function类。函数的框架如下所示：


```python
class MulAdd(Function):
    def forward_cpu(self, inputs):
        # do forward computation on CPU
        return some_tuple

    def backward_cpu(self, inputs, grad_outputs):
        # do backward computation on CPU
        return some_tuple
```

我们必须实现`forward_cpu()`和`backward_cpu()`方法。这些函数的非自变量是数组的元组，这些函数必须返回数组的元组。


>小心: 即使只有一个数组返回，也要返回一个数组的元组。

MulAdd很简单，实现如下


```python
class MulAdd(Function):
    def forward_cpu(self, inputs):
        x, y, z = inputs
        w = x * y + z
        return w,

    def backward_cpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y * gw
        gy = x * gw
        gz = gw
        return gx, gy, gz
```

根据上面的警告，forward_cpu方法返回单个元素的元组。请注意，出现在CPU函数中的所有数组都是numpy.ndarray。 forward函数很简单：它将输入元组解包，计算输出并将其打包成一个元组。backward函数有点复杂。回想一下乘法的差分规则。这个例子只是实现规则。看看返回值，函数只是以相同的顺序打包每个输入的梯度并返回它们。

通过定义前向和反向的核心计算，Function类提供了一个连接逻辑（即存储计算历史等）。




>假设我们实现了一个（向前）函数y = f（x）将 $x \in \mathbb{R}^n$ 作为输入向量并且生成一个向量 $y \in \mathbb{R}^m$。 那么要计算的反向函数记做

> $\lambda_i = \sum_{j=1}^m \frac{\partial y_j}{\partial x_i} \, \gamma_j \,\, \text{for}\, i = 1 \dots n$

>其中 $\gamma$ 是梯度输出。 注意，结果向量 $\lambda$ 必须和forward方法的参数具有相同的形状。

现在我们来定义相应的GPU方法。你可以很容易地预测到我们要编写的方法名为forward_gpu（）和backward_gpu（）：


```python
class MulAdd(Function):
    def forward_cpu(self, inputs):
        ...

    def backward_cpu(self, inputs, grad_outputs):
        ...

    def forward_gpu(self, inputs):
        x, y, z = inputs
        w = x * y + z
        return w,

    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y * gw
        gy = x * gw
        gz = gw
        return gx, gy, gz
```

在GPU方法中，数组的类型是cupy.ndarray。我们使用为这个类定义的算术运算符。这些操作符实施基本的元素级别的运算。

您可能会发现GPU方法的定义与CPU方法的定义完全相同。在这种情况下，我们可以将它们缩减为forward（）和backward（）方法Y


```python
class MulAdd(Function):
    def forward(self, inputs):
        x, y, z = inputs
        w = x * y + z
        return w,

    def backward(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y * gw
        gy = x * gw
        gz = gw
        return gx, gy, gz
```

由于cup.ndarray类实现了numpy.ndarray的许多方法，因此大多数情况下我们可以编写这些统一的方法。

MulAdd函数的用法如下：


```python
x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
w = MulAdd()(x, y, z)
```

它看起来有点难看：我们必须在将MulAdd应用于变量之前显式实例化MulAdd。我们还必须小心，MulAdd的一个实例不能多次使用，因为它作为计算图中的一个节点。在Chainer中，我们经常定义一个很小的封装Python函数来隐藏实例：


```python
def muladd(x, y, z):
    return MulAdd()(x, y, z)

w = muladd(x, y, z)
```

## 面向NumPy / CuPy函数的统一的前向/反向函数


CuPy还实现了许多与NumPy兼容的功能。我们可以用它们写出统一的前向/反向方法。考虑我们想写一个可反向传播函数f（x，y）= exp（x）+ exp（y）
。我们把它命名为ExpAdd。它可以直接写成如下


```python
class ExpAdd(Function):
    def forward_cpu(self, inputs):
        x, y = inputs
        z = np.exp(x) + np.exp(y)
        return z,

    def backward_cpu(self, inputs, grad_outputs):
        x, y = inputs
        gz, = grad_outputs

        gx = gz * np.exp(x)
        gy = gz * np.exp(y)
        return gx, gy

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, y = inputs
        z = cupy.exp(x) + cupy.exp(y)
        return z,

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, y = inputs
        gz, = grad_outputs

        gx = gz * cupy.exp(x)
        gy = gz * cupy.exp(y)
        return gx, gy

def expadd(x, y):
    return ExpAdd()(x, y)

```

>这里我们用cuda.cupy代替直接访问cupy。这是因为如果没有安装CUDA，cupy模块将无法导入。为了保持在非CUDA环境下的有效性，我们不得不延迟访问cupy模块。请注意，即使未安装CUDA，chainer.cuda模块也可以导入。当然，这样的环境中的模块几乎是无用的，但是如果解释器没有运行访问CUDA专用函数的代码，代码仍然是有效的。

CPU和GPU的实现几乎是相同的，只是在GPU方法中numpy被cupy所取代。我们可以使用`cuda.get_array_module()`函数来统一这些函数。这个函数接受任意数量的数组，并为它们返回一个合适的模块。看下面的代码


```python
class ExpAdd(Function):
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        z = xp.exp(x) + xp.exp(y)
        return z,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        gz, = grad_outputs

        gx = gz * xp.exp(x)
        gy = gz * xp.exp(y)
        return gx, gy

def expadd(x, y):
    return ExpAdd()(x, y)
```

请注意，即使未在环境中安装CUDA，此代码也能正常工作。如果没有找到CUDA，get_array_module函数总是返回numpy。我们经常使用名称xp作为可变模块名称，类似于NumPy的缩写np和CuPy的cp。

## 编写一个元素级内核函数

让我们回到`MulAdd`示例。

上面所示的`MulAdd`的GPU实现已经在GPU内核上快速并行化了。但是，它在每次前向和反向计算中调用两个内核。这可能会损害性能，因为中间临时数组是由可能不同的GPU内核读取和写入的，这消耗了很多带宽。我们可以通过定义我们自己的内核来减少调用次数。这也减少了内存消耗。

大多数函数只需要像`MulAdd`这样的元素操作。 `CuPy`提供了一个有用的工具来定义元素内核，`cupy.elementwise.ElementwiseKernel`类和Chainer包装`cuda.elementwise()`函数。我们的MulAdd实现可以改进如下：



```python
class MulAdd(Function):
    def forward_cpu(self, inputs):
        ...

    def backward_cpu(self, inputs, grad_outputs):
        ...

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, y, z = inputs
        w = cuda.elementwise(
            'float32 x, float32 y, float32 z',
            'float32 w',
            'w = x * y + z',
            'muladd_fwd')(x, y, z)
        return w,

    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx, gy = cuda.elementwise(
            'float32 x, float32 y, float32 gw',
            'float32 gx, float32 gy',
            '''
               gx = y * gw;
               gy = x * gw;
            ''',
            'muladd_bwd')(x, y, gw)

        gz = gw
        return gx, gy, gz
```

`cuda.elementwise()`函数接受内核函数的基本实现，并返回一个内核调用函数（实际上，它返回可调用的`ElementwiseKernel`对象）。在典型的用法中，我们将四个参数传递给这个函数，如下所示：

* 输入参数列表。这是一个以逗号分隔的字符串，每个字符串由一个类型说明和一个参数名组成。
* 以与输入参数列表相同的格式输出参数列表。
* 并行循环体。我们可以使用输入/输出参数名称作为这些数组的一个元素。
* 内核函数的名称，显示在调试器和分析器中。

由于`cuda.elementwise()`提供了两种缓存机制，所以上面的代码不会在每次前向/反向计算时编译。

第一个是二进制缓存：`cuda.elementwise()`函数在$(HOME)/.cupy/kernel_cache目录中缓存编译的二进制文件并计算该CUDA代码的散列值，如果给定的代码与散列值相匹配，则重用它。这个缓存机制实际上是在CuPy中实现的。

第二个是上传缓存：给定一个编译的二进制代码，我们必须上传到当前的GPU才能执行它。 `cuda.elementwise()`函数记忆参数和当前设备，如果它被相同设备的相同参数调用，它将重用先前上传的内核代码。

上面的MulAdd代码只适用于float32数组。 ElementwiseKernel也支持type-variadic内核定义。为了定义可变内核函数，可以通过将单个字符作为类型说明符来使用类型占位符：


```python
class MulAdd(Function):
    def forward_cpu(self, inputs):
        ...

    def backward_cpu(self, inputs, grad_outputs):
        ...

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, y, z = inputs
        w = cuda.elementwise(
            'T x, T y, T z',
            'T w',
            'w = x * y + z',
            'muladd_fwd')(x, y, z)
        return w,

    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx, gy = cuda.elementwise(
            'T x, T y, T gw',
            'T gx, T gy',
            '''
               gx = y * gw;
               gy = x * gw;
            ''',
            'muladd_bwd')(x, y, gw)

        gz = gw
        return gx, gy, gz
```

类型占位符T表示CuPy支持的任意数据类型。

CuPy中用户定义的内核有更多的功能。有关更多详细信息，请参阅用户定义内核的CuPy文档。



## 为训练/测试模式写一个函数

我们有时候想让函数在训练和测试模式中表现不同。 Chainer中的训练/测试模式由chainer.config配置。这是一个本地线程配置对象，用户可以将它的train属性替换为True或False。您可以参考配置Chainer来了解如何配置此标志以及其他配置项目。

在这里，我们只是展示如何使用这个标志来做一个支持的培训/测试模式的函数。您将需要检查布尔标志chainer.config.train的值，并适当地分支。

例如，考虑以下简单的 dropout 函数:


```python
def dropout(x):
    xp = cuda.get_array_module(x.data)
    mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
    return x * mask
```

此功能适用于每个元素的dropout并将存活下来的元素乘以2。即使在测试模式下，上面的实现也会使用dropout，但这不是一个理想的行为。我们可以修复它如下：



```python
def dropout(x):
    if not chainer.config.train:
        return x

    xp = cuda.get_array_module(x.data)
    mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
    return x * mask
```

该功能现在支持测试模式。请注意，您通常不必实现自己的`dropout`函数，因为正式提供了`dropout()`。


## 包装函数的连接

有些函数是与参数结合使用的。在这种情况下，写一个包装函数的小连接是很有用的。我们已经看到如何定义一个包装其他连接的Chain（通过继承Chain类）。在这里，我们研究如何定义一个不包含任何其他连接的连接。

作为第一个例子，假设我们要在输入数组和参数数组之间实现元素乘积函数。它可以定义如下：


```python
class EltwiseParamProduct(Link):
    def __init__(self, shape):
        super(EltwiseParamProduct, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(initializers.Normal(scale=1.), shape)

    def __call__(self, x):
        return self.W * x
```

再举一个例子，假设我们要定义一个简单的线性层。它已被定义为线性，所以这是一个示范的例子。线性层分为两部分：一个函数及其包装连接。首先，我们必须定义一个基于Variable的函数：



```python
class LinearFunction(Function):
    def forward(self, inputs):
        x, W, b = inputs
        return x.dot(W.T) + b,

    def backward(self, inputs, grad_outputs):
        x, W, b = inputs
        gy, = grad_outputs

        gx = gy.dot(W)
        gW = gy.T.dot(x)
        gb = gy.sum(axis=0)
        return gx, gW, gb

def linear(x, W, b):
    return LinearFunction()(x, W, b)
```

这个函数有三个参数：输入，权重和偏差。它可以作为模型定义的一部分，但由于用户必须直接管理权重和偏差参数，所以不方便。为了使一个方便的模块，让我们把它包装成一个连接：



```python
class Linear(Link):
    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(
                initializers.Normal(1. / math.sqrt(in_size)),
                (out_size, in_size))
            self.b = chainer.Parameter(0, (out_size,))

    def __call__(self, x):
        return linear(x, self.W, self.b)
```

这个连接隐藏了线性层的参数。

>实现函数的高级技巧：如果要在前向和反向计算之间保留一些信息（例如缓存某些数组），可以将其存储为属性。注意在整个前后计算过程中可能会增加内存消耗。如果要在有限内存的GPU上训练超大型网络，建议不要在前向和反向之间缓存数组。有一个例外：缓存输出数组不会改变内存消耗，因为它们也由输出变量对象保存。

>警告：你不应该假定一个一对一的前向和反向的调用匹配。一些用户可能在一次前向调用后不止一次地调用反向。

## 测试函数 

为了从实现错误中分离学习失败的原因，测试函数实现是非常重要的。 `Chainer`提供简单的实用程序来帮助编写单元测试。它们在`gradient_check`模块中定义。

最重要的测试工具是`numeric_grad()`函数。该函数使用有限差分计算给定函数的数值梯度。它可以使用如下



```python
x  = np.random.randn(4, 3).astype(np.float32)
gy = np.ones((4, 3), dtype=np.float32)
f  = lambda: (x * x,)
gx = gradient_check.numerical_grad(f, (x,), (gy,))
```

`f`是返回从输入数组计算得到的数组的元组的闭包。 `numeric_grad()`的第二个和第三个参数分别是输入数组和输出梯度数组的元组。上面的代码计算`sum(f(x))`的数值梯度，其中`sum`表示所有元素的总和。总和可以通过改变`gy`来加权。 `numeric_grad()`函数也接受附加的`eps`参数，它表示有限差分的量化宽度。


> `numerical_grad()` 函数接受CPU和GPU数组。请注意，我们不能混合CPU和GPU数组。

另一个实用工具是`chainer.testing.assert_allclose()`函数。这与`numpy.testing.assert_allclose()`函数类似。不同之处在于`Chainer`的版本接受CPU和GPU数组作为输入。我们可以混合使用`chainer.testing.assert_allclose()`。可选参数的默认值也不同。

以下是梯度检查实用工具的典型用法。这是`functions.relu()`函数的测试示例


```python
import unittest

from chainer import testing

class TestReLU(unittest.TestCase):
    def test_backward_cpu(self):
        x = Variable(np.random.randn(3, 2).astype(np.float32))
        y = F.relu(x)
        y.grad = np.random.randn(3, 2).astype(np.float32)
        y.backward()

        def f():
            return F.relu(x).data,

        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))
        testing.assert_allclose(gx, x.grad)
```

测试代码的前四行是ReLU函数的简单前向和反向计算。接下来的两行使用相同的前向函数来计算数字梯度，而没有后向程序。最后，我们将这两个结果进行元素比较。请注意，上面的测试代码可以很容易地修改，以通过将CPU数组替换为GPU数组来测试GPU版本。

在大多数情况下，我们并不像上面那样编写代码，因为Chainer提供了一个实用程序函数 chainer.gradient_check.check_backward() 来执行上面的过程。


```python
import unittest

from chainer import gradient_check

class TestReLU(unittest.TestCase):
    def test_backward_cpu(self):

        def f(x):
            return F.relu(x)

        x = np.random.randn(3, 2).astype(np.float32)
        y_grad = np.random.randn(3, 2).astype(np.float32)

        gradient_check.check_backward(f, x, y_grad, atol=1e-4, rtol=1e-4)
```

你可以在 tests/chainer_tests/function_tests 目录下找到很多功能测试的例子。
