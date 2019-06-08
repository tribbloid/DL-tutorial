# %%

import mxnet as mx

predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
print(predicts[0].shape)
labels = [mx.nd.array([0, 1, 1])]
print(labels[0].shape)
ce = mx.metric.CrossEntropy()
ce.update(labels, predicts)
print(ce.get())

# %%

import numpy

import mxnet as mx

loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
input =  mx.ndarray.random_normal(shape=(3, 5))
target = mx.ndarray.random_normal(0, 5, shape=(3), dtype=numpy.double)
output = loss(input, target)
print(output)