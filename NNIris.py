import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split


# 定义激励函数
def logistic(x):
    return 1/(1+np.exp(-x))

# 定义激励函数的导数
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:

    # 构造方法，其中layers类似于(5,10,2)
    def __init__(self, layers):
        self.weights = []
        for i in range(1, len(layers)-1):
            self.weights.append((2 * np.random.random((layers[i-1]+1, layers[i]+1))-1) * 0.25)   # 正负0.25之间
            self.weights.append((2 * np.random.random((layers[i]+1, layers[i+1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):  # 步长0.2, 训练10000次
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # 加入偏置值
        X = temp
        y = np.array(y)


        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):  # 正向计算
                a.append(logistic(np.dot(a[l],self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * logistic_derivative(a[-1])]

            # BP
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * logistic_derivative(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict(self, x):
         x = np.array(x)
         temp = np.ones(x.shape[0] + 1)
         temp[0:-1] = x
         a = temp
         for l in range(0, len(self.weights)):
             a = logistic(np.dot(a, self.weights[l]))

         return a



# import pylab as pl

iris = load_iris()
print(iris.data.shape)

'''
pl.gray()
pl.matshow(digits.images[500])
pl.show()
'''

X = iris.data
y = iris.target
X -= X.min()   # 把输入值映射到0-1
X /= X.max()


nn = NeuralNetwork([4,100,3])
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(len(X_train), ' ', len(X_test))
labels_train = LabelBinarizer().fit_transform(y_train)    # 映射到0 - 1之间
print('labels train', labels_train[0])
labels_test = LabelBinarizer().fit_transform(y_test)

print('Start fitting')
nn.fit(X_train, labels_train, epochs=3000)    # 训练模型
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])   # 预测测试集
    predictions.append(np.argmax(o))


correct = 0
for i in range(len(y_test)):
    if(predictions[i] == y_test[i]):
        correct +=1

print('准确率: ', correct/float(len(y_test))*100, '%')

