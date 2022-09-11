import mnist_loader
import network
import network2
import network3
import network4
import pickle
# import theano
# import theano.tensor as T
# from theano.tensor.nnet import conv
# from theano.tensor.nnet import softmax
# from theano.tensor import shared_randomstreams
# from theano.tensor.signal.pool import pool_2d
import adam_optim 
from adam_optim import AdamOptim
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

# training_data = list(training_data)
# test_data = list(test_data)

# net=network.Network([784,30,10])
# net.SGD( training_data, 30, 10, 3.0, test_data=test_data)

# training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

# training_data = list(training_data)
# test_data = list(test_data)

# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
#     monitor_evaluation_accuracy=True)

# # read data:
# training_data, validation_data, test_data = network3.load_data_shared()
# # mini-batch size:
# mini_batch_size = 10

# net = Network([
#     FullyConnectedLayer(n_in=784, n_out=30),
#     SoftmaxLayer(n_in=30, n_out=10)], mini_batch_size)
# net.SGD(training_data, 5, mini_batch_size, 0.1, validation_data, test_data)

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network4.Network([784,30,10])
net.ADAM( training_data, 30, 10, 3.0, test_data=test_data)



