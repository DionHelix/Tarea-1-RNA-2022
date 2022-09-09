import mnist_loader
import network
import network2
import network3
import pickle
import adam_optim 
from adam_optim import AdamOptim
from network3 import Network, FullyConnectedLayer, SoftmaxLayer 

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network.Network([784,30,10])
net.SGD( training_data, 50, 10, 3.0, test_data=test_data)