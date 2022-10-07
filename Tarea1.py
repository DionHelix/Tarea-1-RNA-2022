import mnist_loader
import network4


training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network4.Network([784,30,10])
net.RMSPROP( training_data, 30, 10, 3.0, test_data=test_data)