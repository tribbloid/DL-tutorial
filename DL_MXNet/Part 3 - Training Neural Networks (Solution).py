# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# ## Training Neural Networks
#
# The network we built in the previous part isn't so smart, it doesn't know anything about our handwritten digits. Neural networks with non-linear activations work like universal function approximators. There is some function that maps your input to the output. For example, images of handwritten digits to class probabilities. The power of neural networks is that we can train them to approximate this function, and basically any function given enough data and compute time.
#
# <img src="assets/function_approx.png" width=500px>
#
# At first the network is naive, it doesn't know the function mapping the inputs to the outputs. We train the network by showing it examples of real data, then adjusting the network parameters such that it approximates this function.
#
# To find these parameters, we need to know how poorly the network is predicting the real outputs. For this we calculate a **loss function** (also called the cost), a measure of our prediction error. For example, the mean squared loss is often used in regression and binary classification problems
#
# $$
# \large \ell = \frac{1}{2n}\sum_i^n{\left(y_i - \hat{y}_i\right)^2}
# $$
#
# where $n$ is the number of training examples, $y_i$ are the true labels, and $\hat{y}_i$ are the predicted labels.
#
# By minimizing this loss with respect to the network parameters, we can find configurations where the loss is at a minimum and the network is able to predict the correct labels with high accuracy. We find this minimum using a process called **gradient descent**. The gradient is the slope of the loss function and points in the direction of fastest change. To get to the minimum in the least amount of time, we then want to follow the gradient (downwards). You can think of this like descending a mountain by following the steepest slope to the base.
#
# <img src='assets/gradient_descent.png' width=350px>

# %% [markdown]
# ## Backpropagation
#
# For single layer networks, gradient descent is straightforward to implement. However, it's more complicated for deeper, multilayer neural networks like the one we've built. Complicated enough that it took about 30 years before researchers figured out how to train multilayer networks.
#
# Training multilayer networks is done through **backpropagation** which is really just an application of the chain rule from calculus. It's easiest to understand if we convert a two layer network into a graph representation.
#
# <img src='assets/backprop_diagram.png' width=550px>
#
# In the forward pass through the network, our data and operations go from bottom to top here. We pass the input $x$ through a linear transformation $L_1$ with weights $W_1$ and biases $b_1$. The output then goes through the sigmoid operation $S$ and another linear transformation $L_2$. Finally we calculate the loss $\ell$. We use the loss as a measure of how bad the network's predictions are. The goal then is to adjust the weights and biases to minimize the loss.
#
# To train the weights with gradient descent, we propagate the gradient of the loss backwards through the network. Each operation has some gradient between the inputs and outputs. As we send the gradients backwards, we multiply the incoming gradient with the gradient for the operation. Mathematically, this is really just calculating the gradient of the loss with respect to the weights using the chain rule.
#
# $$
# \large \frac{\partial \ell}{\partial W_1} = \frac{\partial L_1}{\partial W_1} \frac{\partial S}{\partial L_1} \frac{\partial L_2}{\partial S} \frac{\partial \ell}{\partial L_2}
# $$
#
# **Note:** I'm glossing over a few details here that require some knowledge of vector calculus, but they aren't necessary to understand what's going on.
#
# We update our weights using this gradient with some learning rate $\alpha$.
#
# $$
# \large W^\prime_1 = W_1 - \alpha \frac{\partial \ell}{\partial W_1}
# $$
#
# The learning rate $\alpha$ is set such that the weight update steps are small enough that the iterative method settles in a minimum.

# %% [markdown]
# ## Losses in PyTorch
#
# Let's start by seeing how we calculate the loss with PyTorch. Through the `nn` module, PyTorch provides losses such as the cross-entropy loss (`nn.CrossEntropyLoss`). You'll usually see the loss assigned to `criterion`. As noted in the last part, with a classification problem such as MNIST, we're using the softmax function to predict class probabilities. With a softmax output, you want to use cross-entropy as the loss. To actually calculate the loss, you first define the criterion then pass in the output of your network and the correct labels.
#
# Something really important to note here. Looking at [the documentation for `nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss),
#
# > This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class.
# >
# > The input is expected to contain scores for each class.
#
# This means we need to pass in the raw output of our network into the loss, not the output of the softmax function. This raw output is usually called the *logits* or *scores*. We use the logits because softmax gives you probabilities which will often be very close to zero or one but floating-point numbers can't accurately represent values near zero or one ([read more here](https://docs.python.org/3/tutorial/floatingpoint.html)). It's usually best to avoid doing calculations with probabilities, typically we use log-probabilities.

# %% [markdown]
# # In[4]:

# %%
import mxnet as mx
import mxnet.gluon as gl
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms, MNIST

# %%

# construct and initialize network.
from mxnet.ndarray import NDArray

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# %%
toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=(0.5), std=(0.5))

transform = transforms.Compose([
    toTensor,
    normalize
])

# %%
# Download and load the training data
trainset: MNIST = MNIST('~/.mxnet/MNIST_data/', train=True).transform_first(transform)
trainloader: DataLoader = DataLoader(trainset, batch_size=64, shuffle=True)


# %%


# %%
# Build a feed-forward network
def getModel() -> nn.Sequential:
    model = nn.Sequential()
    # with model.name_scope():
    model.add(
        nn.Dense(128),
        nn.Activation('relu'),
        nn.Dense(64),
        nn.Activation('relu'),
        nn.Dense(10)
    )

    # model.initialize()
    model.initialize(ctx=ctx)
    return model


# %%

model = getModel()

# %%
# Define the loss
criterion = gl.loss.SoftmaxCrossEntropyLoss()


# %%
# Get our data

def getBatch():
    ii, ll = next(trainloader.__iter__())

    # print(f"image shape = {ii.shape}")
    # print(f"image context = {ii.context}")

    # Flatten images
    ii = ii.as_in_context(ctx)
    ll = ll.as_in_context(ctx)

    return ii, ll


images, labels = getBatch()

# %%
# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels

# %%
loss = criterion(logits, labels)

# %%
print(loss)

# %% [markdown]
# In my experience it's more convenient to build the model with a log-softmax output using `nn.LogSoftmax` or `F.log_softmax` ([documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax)). Then you can get the actual probabilites by taking the exponential `torch.exp(output)`. With a log-softmax output, you want to use the negative log likelihood loss, `nn.NLLLoss` ([documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss)).
#
# >**Exercise:** Build a model that returns the log-softmax as the output and calculate the loss using the negative log likelihood loss.

# %%


# %% [markdown]
# # Solution

# %%

# # Build a feed-forward network
# model = nn.Sequential()
# # with model.name_scope():
# model.add(
#     nn.Dense(128, activation='relu'),
#     # nn.Activation('relu'),
#     nn.Dense(64, activation='relu'),
#     nn.Dense(10)
# )
#
# # %%
# model.initialize()
#
# # %%
# # Define the loss
# criterion = gl.loss.SoftmaxCrossEntropyLoss()

# %%
# Get our data
images, labels = getBatch()

# %%
# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

# %%
print(loss)

# %% [markdown]
# ## Autograd
#
# Now that we know how to calculate a loss, how do we use it to perform backpropagation? Torch provides a module, `autograd`, for automatically calculating the gradients of tensors. We can use it to calculate the gradients of all our parameters with respect to the loss. Autograd works by keeping track of operations performed on tensors, then going backwards through those operations, calculating gradients along the way. To make sure PyTorch keeps track of operations on a tensor and calculates the gradients, you need to set `requires_grad = True` on a tensor. You can do this at creation with the `requires_grad` keyword, or at any time with `x.requires_grad_(True)`.
#
# You can turn off gradients for a block of code with the `torch.no_grad()` content:
# ```python
# x = torch.zeros(1, requires_grad=True)
# >>> with torch.no_grad():
# ...     y = x * 2
# >>> y.requires_grad
# False
# ```
#
# Also, you can turn on or off gradients altogether with `torch.set_grad_enabled(True|False)`.
#
# The gradients are computed with respect to some variable `z` with `z.backward()`. This does a backward pass through the operations that created `z`.

# %%


# %%
x = mx.random.randn(2, 2)
x.attach_grad()
print(x)

# %%


# %%
with autograd.record():
    y = x ** 2
    print(y)

# %% [markdown]
# Below we can see the operation that created `y`, a power operation `PowBackward0`.

# %%


# %% [markdown]
# # grad_fn shows the function that generated this variable
# print(y.grad_fn)

# %% [markdown]
# The autgrad module keeps track of these operations and knows how to calculate the gradient for each one. In this way, it's able to calculate the gradients for a chain of operations, with respect to any one tensor. Let's reduce the tensor `y` to a scalar value, the mean.

# %%


# %%
with autograd.record():
    z = y.mean()
    print(z)

# %% [markdown]
# You can check the gradients for `x` and `y` but they are empty currently.

# %%


# %%
print(x.grad)

# %% [markdown]
# To calculate the gradients, you need to run the `.backward` method on a Variable, `z` for example. This will calculate the gradient for `z` with respect to `x`
#
# $$
# \frac{\partial z}{\partial x} = \frac{\partial}{\partial x}\left[\frac{1}{n}\sum_i^n x_i^2\right] = \frac{x}{2}
# $$

# %%


# %%
z.backward()
print(x.grad)
print(x / 2)

# %% [markdown]
# These gradients calculations are particularly useful for neural networks. For training we need the gradients of the weights with respect to the cost. With PyTorch, we run data forward through the network to calculate the loss, then, go backwards to calculate the gradients with respect to the loss. Once we have the gradients we can make a gradient descent step.

# %% [markdown]
# ## Loss and Autograd together
#
# When we create a network with PyTorch, all of the parameters are initialized with `requires_grad = True`. This means that when we calculate the loss and call `loss.backward()`, the gradients for the parameters are calculated. These gradients are used to update the weights with gradient descent. Below you can see an example of calculating the gradients using a backwards pass.

# %%


# %%
# # Build a feed-forward network
# model = nn.Sequential()
# # with model.name_scope():
# model.add(
# #     nn.Dense(128, activation='relu'),
#     nn.Dense(64, activation='relu'),
#     # nn.Activation('relu'),
#     nn.Dense(64, activation='relu'),
#     nn.Dense(10)
# )
# model.initialize()

# %%

images, labels = getBatch()

# %%
with autograd.record():
    logits = model(images)
    loss = criterion(logits, labels)

# %%
print('Before backward pass: \n', model[0].weight.grad())

# %%
loss.backward()

# %%
print('After backward pass: \n', model[0].weight.grad())

# %% [markdown]
# ## Training the network!
#
# There's one last piece we need to start training, an optimizer that we'll use to update the weights with the gradients. We get these from PyTorch's [`optim` package](https://pytorch.org/docs/stable/optim.html). For example we can use stochastic gradient descent with `optim.SGD`. You can see how to define an optimizer below.


# %%
# Optimizers require the parameters to optimize and a learning rate
optimizer = gl.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})

# %% [markdown]
# Now we know how to use all the individual parts so it's time to see how they work together. Let's consider just one learning step before looping through all the data. The general process with PyTorch:
#
# * Make a forward pass through the network
# * Use the network output to calculate the loss
# * Perform a backward pass through the network with `loss.backward()` to calculate the gradients
# * Take a step with the optimizer to update the weights
#
# Below I'll go through one training step and print out the weights and gradients so you can see how it changes. Note that I have a line of code `optimizer.zero_grad()`. When you do multiple backwards passes with the same parameters, the gradients are accumulated. This means that you need to zero the gradients on each training pass or you'll retain gradients from previous training batches.


# %%
print('Initial weights - ', model[0].weight.data())

# %%
images, labels = getBatch()

# %% [markdown]
# Clear the gradients, do this because gradients are accumulated
# optimizer.zero_grad()

# %%
# Forward pass, then backward pass, then update weights
with autograd.record():
    output = model.forward(images)
    loss = criterion(output, labels)

# %%
loss.backward()
print('Gradient -', model[0].weight.grad())

# %%
# Take an update step and few the new weights
optimizer.step(64)
print('Updated weights - ', model[0].weight.data())

# %% [markdown]
# ### Training for real
#
# Now we'll put this algorithm into a loop so we can go through all the images. Some nomenclature, one pass through the entire dataset is called an *epoch*. So here we're going to loop through `trainloader` to get our training batches. For each batch, we'll doing a training pass where we calculate the loss, do a backwards pass, and update the weights.
#
# > **Exercise: ** Implement the training pass for our network. If you implemented it correctly, you should see the training loss drop with each epoch.


# %%
# model = nn.Sequential()
# # with model.name_scope():
# model.add(
#     nn.Dense(128, activation='relu'),
#     nn.Dense(64, activation='relu'),
#     nn.Dense(10)
# )
# model.initialize()

# %%
# criterion = gl.loss.SoftmaxCrossEntropyLoss()
optimizer = gl.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})

# %%
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.as_in_context(ctx)
        labels = labels.as_in_context(ctx)

        with autograd.record():
            output = model.forward(images)
            loss = criterion(output, labels)

        loss.backward()
        optimizer.step(images.shape[0] / 2)

        running_loss += loss.mean().asscalar()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# %% [markdown]
# With the network trained, we can check out it's predictions.

# %%
# %matplotlib inline
import helper

images, labels = getBatch()

# print(images[0].shape)

img = images[0]

# Turn off gradients to speed up this part
logits = model.forward(img)

# print(img_raw.shape)

# Output of the network are logits, need to take softmax for probabilities
ps = mx.ndarray.softmax(logits, axis=1)

helper.view_classify(img, ps)

# %%


# %% [markdown]
# Now our network is brilliant. It can accurately predict the digits in our images. Next up you'll write the code for training a neural network on a more complex dataset.
