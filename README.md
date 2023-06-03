# Building a Neural Network From Scratch
The data and code can be found [here](https://drive.google.com/drive/folders/1XjMR1pgWWjI3V_7D_EfF3j-_Q2UV0Ccw?usp=sharing)
### By: _Lucas Thomazi_
### The inspiration for this project came from [this](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) playlist from [3Blue1Brown](https://www.youtube.com/@3blue1brown) where he explains what are neural networks and how do they work. It is only meant to be used as a learning tool but feel free to play around with it! ;)

### **Goal**: The 'hello world' of machine learning. 
> ### Using the [MNIST database](https://www.wikiwand.com/en/MNIST_database) to recognize hand written digits. The MNIST database is a large dataset of handwritten digits used widely in machine learning and image processing for benchmarking classification algorithms. It contains 60,000 training images and 10,000 testing images, all grayscale and 28x28 pixels.

## P1. The Math Behind The 'Magic'

Each image is a 28x28 pixel training image where each pixel has a value between 0 and 255 (gray scale) where 0 is black and 255 is white. 

We can represent the data as a matrix where each row is an example image of 784 items long because each item represents a pixel in that image (28 * 28 = 784). But first we need to transpose this example. So instead of each row being an example, each column will be an example still 784 items long.

### P1.1 The NN
I'll build a quite simple NN with one hidden layers. The input layer will be 784 nodes. Each of the pixels maps to a node. The hidden layer will have 10 units and the output layer will have 10 units too. Each of the nodes of will correspond to a digit that can be predicted (0,1...8,9). 

There is three parts to training this network:

1. Forward Propagation
* Take an image and run it through the network. From this network you compute what your output is going to be. 

* We have our variable A<sup>[0]</sup> which is just our input layer.

* Z<sup>[1]</sup> is the unactivated hidden layer. To get Z<sup>[1]</sup> we need a weight and a bias

> Z<sup>[1]</sup> = w<sup>[1]</sup> * A<sup>[0]</sup> + b<sup>[1]</sup> 
>> These are Matrixes so * represents the dot product

* Then we need to apply an activation function. If we didn't apply the activation function, each node would just be a linear combination of the nodes before and we would just have a glorified linear regression algorithm. To solve that we apply an [activation function](https://en.wikipedia.org/wiki/Activation_function) (ReLU, Sigmoid, Tanh, etc.). I'll use [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (Rectified Linear Unit).

* Then A<sup>[1]</sup>, the activated hidden layer, would just be ReLU(Z<sup>[1]</sup>) 

* To get to A<sup>[2]</sup>, the output layer, we have another Z value.
> Z<sup>[2]</sup> = w<sup>[2]</sup> * A<sup>[1]</sup> + b<sup>[2]</sup> 
>> where: 
>> * w<sup>[2]</sup> is the matrix with the weights between the hidden and output layer
>> * A<sup>[1]</sup> is the activated hidden layer
>> * b<sup>[2]</sup> is another constant bias term.

* Finally I'll apply a Softmax activation function to get A<sup>[2]</sup>

* Then A<sup>[2]</sup>, the output layer, would just be [Softmax](https://en.wikipedia.org/wiki/Softmax_function) (Z<sup>[2]</sup>). The Softmax activation function gives you a probability which is what I want. The output matrix would look like [0.02, 0.9, 0.05, 0.12, ...] and will have 10 values where 0.02 would be the probability of the input being a zero, 0.9 would be the probability of the output being a one, and so on.

2. Backwards Propagation 
This is necessary because it 'tunes' the network by telling it how far off the predictions were.

For the output layer:
* dZ<sup>[2]</sup> = A<sup>[2]</sup> - Y (output)
> Represents the error of the output layer. If Y = 4, we can't subtract 4 from it, so we need to one hot encode it. 

* dw<sup>[2]</sup> = 1/m * dZ<sup>[2]</sup> A<sup>[1]T</sup>
> The derivative of the loss function with respect to the weights in the output layer. Here, 'm' represents an arbitrary number of images.

* db<sup>[2]</sup> = 1/m * sum(dZ<sup>[2]</sup>)
> The average of the absolute error aka how much the output was off by.

For the hidden layer:
* Z<sup>[1]</sup> = w<sup>[2]T</sup> dZ<sup>[2]</sup> .* g'(Z<sup>[2]</sup>)
> How much the hidden layer was off by. Taking the error from the output layer and applying the weights to it in reverse to get to the errors in the first layer. Here g' is undoing the activtion function to get the proper error for the first layer.

* dw<sup>[1]</sup> = 1/m * dz<sup>[1]</sup> X<sup>[T]</sup>
> Same as before

* db<sup>[2]</sup> = 1/m * sum(dZ<sup>[1]</sup>)
> Same as before

2. Update Parameters

* w<sup>[1]</sup> = w<sup>[1]</sup> - alpha*dw<sup>[1]</sup>
* b<sup>[1]</sup> = b<sup>[1]</sup> - alpha*db<sup>[1]</sup>
* w<sup>[2]</sup> = w<sup>[2]</sup> - alpha*dw<sup>[2]</sup>
* b<sup>[2]</sup> = b<sup>[2]</sup> - alpha*db<sup>[2]</sup>

Where alpha is the learning rate 

3. Repeat!