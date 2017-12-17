---
ilayout: post
title: Chainer 类型检查
date: 2017-12-17
categories: blog
tags: [Chainer,类型检查]
descrption: Chainer 类型检查
---

# 类型检查

在本节中，您将了解以下内容：

* 类型检查的基本用法
* 类型信息的细节
* 类型检查的内部机制
* 更复杂的情况
* 函数调用
* 典型的检查例子

阅读本节后，您将能够：

* 编写一个代码来检查你自己函数的输入参数类型

## 类型检查的基本用法

当您使用无效类型的数组调用某个函数时，您有时不会收到错误，但会通过广播获得意外的结果。当您使用非法类型的CUDA数组时，会导致内存损坏，并且会出现严重错误。这些错误很难修复。 Chainer可以检查每个函数的先决条件，并有助于防止这些问题。这些条件可以帮助用户理解函数的设定。
 
每个函数的实现都有一个类型检查方法check_type_forward（）。该函数在Function类的forward（）方法之前被调用。您可以重写此方法来检查参数的类型和形状的条件。


check_type_forward() 有一个 in_types 参数:

```
def check_type_forward(self, in_types):
  ...
```

`in_types`是`TypeInfoTuple`的一个实例，它是元组的一个子类。要获取有关第一个参数的类型信息，请使用`in_types [0]`。如果函数获取多个参数，我们建议使用新的变量来提高可读性：


```python
x_type, y_type = in_types
```

在这种情况下，x_type表示第一个参数的类型，y_type表示第二个参数。

我们用一个例子来描述in_types的用法。当你想检查x_type的维数是否等于2时，写下这个代码：


```python
utils.type_check.expect(x_type.ndim == 2)
```

当这个条件成立时，没有任何反应。否则，这段代码会抛出一个异常，并且用户得到这样的消息:

```
Traceback (most recent call last):
...
chainer.utils.type_check.InvalidType: Expect: in_types[0].ndim == 2
Actual: 3 != 2
```

这个错误消息意味着“第一个参数的ndim预期为2，但实际上是3”。


## 类型信息的细节


您可以访问x_type的三方面的信息。

* .shape 是一个int的元组。每个值都是每个维度的大小。
* .ndim 是表示维数的int值。请注意，ndim == len（shape）
* .dtype 是表示数据类型的 numpy.dtype 。

你可以检查所有成员。例如，第一维的大小必须是正值，你可以这样写：



```python
utils.type_check.expect(x_type.shape[0] > 0)
```

您也可以使用.dtype检查数据类型：


```python
utils.type_check.expect(x_type.dtype == np.float64)
```

而一个错误是这样的：

```
Traceback (most recent call last):
...
chainer.utils.type_check.InvalidType: Expect: in_types[0].dtype == <class 'numpy.float64'>
Actual: float32 != <class 'numpy.float64'>
```
你同样可以检查dtype的kind。下面代码检查type是否为浮点型


```python
utils.type_check.expect(x_type.dtype.kind == 'f')
```

你可以在变量之间进行比较。例如，下面的代码检查第一个参数和第二个参数是否具有相同的长度：



```python
utils.type_check.expect(x_type.shape[1] == y_type.shape[1])
```

## 类型检查的内部机制

它如何显示`in_types[0].ndim == 2`的错误信息？如果`x_type`是一个包含ndim成员变量的对象，我们不能显示这样的错误信息，因为Python解释器将此方程作为布尔值进行计算。


其实`x_type`是一个`Expr`对象，本身并没有一个`ndim`成员变量。Expr代表一个语法树。`x_type.ndim`使`Expr`对象表示为`（getattr，x_type，'ndim'）`。 `x_type.ndim == 2`使对象执行像`（eq，（getattr，x_type，'ndim'），2）`这样的操作。 `type_check.expect（）`获取Expr对象并对其进行估值运算。如果为真，则不会导致错误，也不会显示任何内容。否则，此方法显示可读的错误消息。

如果要估值`Expr`对象，请调用`eval（）`方法：


```python
actual_type = x_type.eval()
```

`actual_type`是`TypeInfo`的一个实例，而`x_type`是`Expr`的一个实例。以同样的方式，
`x_type.shape[0].eval()`返回一个`int`值。


## 更强大的方法

`Expr`类更强大。它支持所有的数学运算符，如`+`和`*`。你可以写出一个条件，即`x_type`的第一个维度是`y_type`的第一维度乘以四：


```python
utils.type_check.expect(x_type.shape[0] == y_type.shape[0] * 4)
```
和`y_type.shape [0] == 1`时，用户可以得到下面的错误信息：

```
Traceback (most recent call last):
...
chainer.utils.type_check.InvalidType: Expect: in_types[0].shape[0] == in_types[1].shape[0] * 4
Actual: 3 != 4
```
要比较函数的成员变量，请用Variable包装一个值以显示可读的错误消息：


```python
x_type.shape[0] == utils.type_check.Variable(self.in_size, "in_size")
```

这段代码可以检查下面的等价条件：


```python
x_type.shape[0] == self.in_size
```

但是，后一种情况不知道这个值的意思。当这个条件不满意时，后面的代码显示不可读的错误信息：

```
chainer.utils.type_check.InvalidType: Expect: in_types[0].shape[0] == 4  # what does '4' mean?
Actual: 3 != 4
```

请注意，`utils.type_check.Variable`的第二个参数仅用于可读性。

前者显示这个消息：

```
chainer.utils.type_check.InvalidType: Expect: in_types[0].shape[0] == in_size  # OK, `in_size` is a value that is given to the constructor
Actual: 3 != 4  # You can also check actual value here
```

## 调用函数

如何检查所有shape值的总和？ Expr也支持函数调用：



```python
sum = utils.type_check.Variable(np.sum, 'sum')
utils.type_check.expect(sum(x_type.shape) == 10)
```

为什么我们需要用`utils.type_check.Variable`包装函数`numpy.sum`？`x_type.shape`不是一个元组，而是`Expr`的一个对象，就像我们之前看到的那样。 因此，`numpy.sum（x_type.shape）`会失效。

上面的例子产生这样的错误信息：

```
Traceback (most recent call last):
...
chainer.utils.type_check.InvalidType: Expect: sum(in_types[0].shape) == 10
Actual: 7 != 10
```

## 更加复杂的例子

在不能用这些操作符的情况下如何写一个更复杂的条件？您可以使用eval（）方法评估Expr并获取其结果值。然后检查情况并手动显示警告消息：



```python
x_shape = x_type.shape.eval()  # get actual shape (int tuple)
if not more_complicated_condition(x_shape):
    expect_msg = 'Shape is expected to be ...'
    actual_msg = 'Shape is ...'
    raise utils.type_check.InvalidType(expect_msg, actual_msg)
```

请写一个可读的错误信息。此代码会生成以下错误消息：
```
Traceback (most recent call last):
...
chainer.utils.type_check.InvalidType: Expect: Shape is expected to be ...
Actual: Shape is ...
```

## 典型的检查例子

我们给出一个典型的函数检查类型。

首先检查参数的数量：


```python
utils.type_check.expect(in_types.size() == 2)
```

n_types.size（）返回表示参数个数的Expr对象。你可以用同样的方法检查它。

然后，获取每种类型：
```
x_type, y_type = in_types
```

检查in_types.size（）之前不要获取每个值。当参数个数不合法时，type_check.expect可能会输出无用的错误消息。例如，当in_types的大小为0时，此代码不起作用：


```python
utils.type_check.expect(
  in_types.size() == 2,
  in_types[0].ndim == 3,
)

```

之后，检查每种类型：



```python
utils.type_check.expect(
  x_type.dtype == np.float32,
  x_type.ndim == 3,
  x_type.shape[1] == 2,
)
```

上述示例即使在`x_type.ndim == 0`的情况下也能正常工作，因为所有条件都将被懒惰地评估。
