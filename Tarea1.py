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

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 3.0, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)