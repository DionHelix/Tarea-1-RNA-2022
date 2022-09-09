import mnist_loader
import network
import network2
import network3
import pickle
import adam_optim 
from adam_optim import AdamOptim
from network3 import Network, FullyConnectedLayer, SoftmaxLayer 

training_data, validation_data, test_data = network3.load_data_shared()

mini_batch_size = 10

net = Network([
    FullyConnectedLayer(n_in=784, n_out=30),
    SoftmaxLayer(n_in=30, n_out=10)], mini_batch_size)
net.SGD(training_data, 5, mini_batch_size, 3.0, validation_data, test_data)