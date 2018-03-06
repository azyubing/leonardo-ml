import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 784 输入神经元， 100 隐藏神经元， 10 输出神经元
net = network.Network([784, 100, 10])
# python 3.x 需要 list()后，再索引查找元素
print('Make sure list training_data before find element in it.')
print('training data is: ', training_data)
print('test data is: ', test_data)
# 训练次数=30, learning rate = 3.0, mini_batch_size=10
net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))


