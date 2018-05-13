# SVM学习笔记（二）----手写数字识别

# 引言 {#引言}

[上一篇博客](http://blog.csdn.net/chunxiao2008/article/details/50266025)整理了一下SVM分类算法的基本理论问题，它分类的基本思想是利用最大间隔进行分类，处理非线性问题是通过核函数将特征向量映射到高维空间，从而变成线性可分的，但是运算却是在低维空间运行的。考虑到数据中可能存在噪音，还引入了松弛变量。  
理论是抽象的，问题是具体的。站在岸上学不会游泳，光看着梨子不可能知道梨子的滋味。本篇博客就是用SVM分类算法解决一个经典的机器学习问题–手写数字识别。体会一下SVM算法的具体过程，理理它的一般性的思路。

# 问题的提出 {#问题的提出}

人类视觉系统是世界上众多的奇迹之一。看看下面的手写数字序列：  
![](http://7xp3us.com1.z0.glb.clouddn.com/xbnumsb.png "手写数字")  
大多数人毫不费力就能够认出这些数字为504192。如果尝试让计算机程序来识别诸如上面的数字，就会明显感受到视觉模式识别的困难。关于我们识别形状——–“9顶上有一个圈，右下方则是一条竖线”这样的简单直觉，实际上算法很难轻易表达出来。  
![](http://7xp3us.com1.z0.glb.clouddn.com/xbnumsball.png "大量手写数字")  
SVM分类算法以另一个角度来考虑问题。其思路是获取大量的手写数字，常称作训练样本，然后开发出一个可以从这些训练样本中进行学习的系统。换言之，SVM使用样本来自动推断出识别手写数字的规则。随着样本数量的增加，算法可以学到更多关于手写数字的知识，这样就能够提升自身的准确性。  
本文采用的数据集就是著名的“MNIST数据集”。这个数据集有60000个训练样本数据集和10000个测试用例。直接调用scikit-learn库中的SVM，使用默认的参数，1000张手写数字图片，判断准确的图片就高达9435张。

# SVM的算法过程 {#svm的算法过程}

通常，对于分类问题。我们会将数据集分成三部分，训练集、测试集、交叉验证集。用训练集训练生成模型，用测试集和交叉验证集进行验证模型的准确性。  
加载数据的代码如下：

```
"""
mnist_loader
~~~~~~~~~~~~
一个加载模式识别图片数据的库。
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """
    返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    训练数据包含50，000张图片，测试数据和验证数据都只包含10,000张图片
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
```

SVM算法进行训练和预测的代码如下：

```
"""
mnist_svm
~~~~~~~~~
使用SVM分类器，从MNIST数据集中进行手写数字识别的分类程序
"""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm
import time

def svm_baseline():
    print time.strftime('%Y-%m-%d %H:%M:%S') 
    training_data, validation_data, test_data = mnist_loader.load_data()
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC()
    # clf = svm.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(training_data[0], training_data[1])
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "%s of %s test values correct." % (num_correct, len(test_data[1]))
    print time.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    svm_baseline()
```

以上代码没有用验证集进行验证。这是因为本例中，用测试集和验证集要判断的是一个东西，没有必要刻意用验证集再来验证一遍。事实上，我的确用验证集也试了一下，和测试集的结果基本一样。呵呵

直接运行代码，结果如下：

```
2016-01-02 14:01:46
9435 of 10000 test values correct.
2016-01-02 14:12:37
```

在我的ubuntu上，运行11分钟左右就可以完成训练，并预测测试集的结果。  
需要说明的是，svm.SVC\(\)函数的几个重要参数。直接用help命令查看一下文档，这里我稍微翻译了一下：  
C : 浮点型，可选 \(默认=1.0\)。误差项的惩罚参数C  
kernel : 字符型, 可选 \(默认=’rbf’\)。指定核函数类型。只能是’linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 或者自定义的。如果没有指定，默认使用’rbf’。如果使用自定义的核函数，需要预先计算核矩阵。  
degree : 整形, 可选 \(默认=3\)。用多项式核函数\(‘poly’\)时，多项式核函数的参数d，用其他核函数，这个参数可忽略  
gamma : 浮点型, 可选 \(默认=0.0\)。’rbf’, ‘poly’ and ‘sigmoid’核函数的系数。如果gamma是0，实际将使用特征维度的倒数值进行运算。也就是说，如果特征是100个维度，实际的gamma是1/100。  
coef0 : 浮点型, 可选 \(默认=0.0\)。核函数的独立项，’poly’ 和’sigmoid’核时才有意义。  
可以适当调整一下SVM分类算法，看看不同参数的结果。当我的参数选择为C=100.0, kernel=’rbf’, gamma=0.03时，预测的准确度就已经高达98.5%了。

# SVM参数的调优初探 {#svm参数的调优初探}

SVM分类算法需要调整的参数就只有几个。那么这些参数如何选取，有没有一些经验性的规律呢？

* 核函数选择

![](http://7xp3us.com1.z0.glb.clouddn.com/xbplot_iris_001.png "核函数比较")  
如上图，线性核函数的分类边界是线性的，非线性核函数分类边界是很复杂的非线性边界。所以当能直观地观察数据时，大致可以判断分类边界，从而有倾向性地选择核函数。

* 参数gamma和C的选择

机器学习大牛Andrew Ng说，关于SVM分类算法，他一直用的是高斯核函数，其它核函数他基本就没用过。可见，这个核函数应用最广。  
gamma参数，当使用高斯核进行映射时，如果选得很小的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果gamma选得很大，则可以将任意的数据映射为线性可分——这样容易导致非常严重的过拟合问题。  
C参数是寻找 margin 最大的超平面”和“保证数据点偏差量最小”）之间的权重。C越大，模型允许的偏差越小。  
下图是一个简单的二分类情况下，不同的gamma和C对分类结果的影响。  
![](http://7xp3us.com1.z0.glb.clouddn.com/xbplot_rbf_parameters_001.png "gamma和C")  
相同的C，gamma越大，分类边界离样本越近。相同的gamma，C越大，分类越严格。  
下图是不同C和gamma下分类器交叉验证准确率的热力图  
![](http://7xp3us.com1.z0.glb.clouddn.com/xbplot_rbf_parameters_002.png "gamma和C")  
由图可知，模型对gamma参数是很敏感的。如果gamma太大，无论C取多大都不能阻止过拟合。当gamma很小，分类边界很像线性的。取中间值时，好的模型的gamma和C大致分布在对角线位置。还应该注意到，当gamma取中间值时，C取值可以是很大的。  
在实际项目中，这几个参数按一定的步长，多试几次，一般就能得到比较好的分类效果了。

# 小结 {#小结}

回顾一下整个问题。我们进行了如下操作。对数据集分成了三部分，训练集、测试集和交叉验证集。用SVM分类模型进行训练，依据测试集和验证集的预测结果来优化参数。依靠sklearn这个强大的机器学习库，我们也能解决手写识别这么高大上的问题了。事实上，我们只用了几行简单代码，就让测试集的预测准确率高达98.5%。  
SVM算法也没有想象的那么高不可攀嘛，呵呵！  
事实上，就算是一般性的机器学习问题，我们也是有一些一般性的思路的，如下：



![](/assets/xbprocess.jpg)

