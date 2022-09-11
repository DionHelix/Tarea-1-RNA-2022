import mnist_loader
import network4
from network4 import AdamOptim

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network4.Network([784,30,10], AdamOptim(eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8))
net.ADAM( training_data, 30, 10, 3.0, test_data=test_data)