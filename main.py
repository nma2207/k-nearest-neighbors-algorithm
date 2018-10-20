import numpy as np
import knn_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def one_hot_encoding(t):
    k = np.max(t)
    res = np.zeros(shape = [t.size, k+1])
    for i in range(t.size):
        res[i, t[i]] = 1
    return res
def from_one_hot_encoding(t):
    res = np.zeros(shape=[t.shape[0]], dtype=int)
    for i in range(t.shape[0]):
        res[i] = np.argmax(t[i])
    return res

def calcY1(x, sigma):
    y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    e = sigma * np.random.randn(x.size)
    return y+e

def calcY2(x, sigma):
    y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x) + 40
    e = sigma * np.random.randn(x.size)
    return y+e

def generateData(n=100):
    sigma = 15
    linspace = np.linspace(0, 1, n)
    y1 = calcY1(linspace, sigma)
    y2 = calcY2(linspace, sigma)

    x = np.zeros(shape=[2*n,2])
    x[:n,0] = linspace
    x[:n, 1] = y1
    x[n:, 0] = linspace
    x[n:, 1] = y2

    t = np.zeros(shape=[2*n,], dtype=int)
    t[n:] = np.ones(shape=[n], dtype=int)
    t = one_hot_encoding(t)
    trainX, testX, trainY, testY = train_test_split(x, t,
                                                        test_size = 0.2,
                                                        random_state = 42)
    return trainX, testX, trainY, testY



def mainClass():
    trainX, testX, trainY, testY = generateData(n = 5000)
    knnClassifier = knn_model.KnnModel(k = 10)
    knnClassifier.fit(trainX, trainY)
    predict = knnClassifier.predict(testX)
    predict = from_one_hot_encoding(predict)
    plt.figure()
    x = np.linspace(0, 1, 100)
    y1 = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    y2 = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x) + 40

    plt.plot(x, y1, 'b')
    plt.plot(x, y2, 'y')
    for i in range(testX.shape[0]):
        if predict[i]==0:
            plt.plot(np.array([testX[i,0]]), np.array([testX[i,1]]), '.g')
        else:
            plt.plot(np.array([testX[i,0]]), np.array([testX[i,1]]), '.r')


    plt.show()
def main():
    x = np.linspace(0, 1, 324)
    y = calcY1(x, 5)
    plt.figure()
    plt.plot(x, y, '.')

    trainX, testX, trainY, testY = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    random_state = 42)
    trainX = trainX.reshape([trainX.size,1])
    trainY = trainY.reshape([trainY.size,1])
    knnRegression = knn_model.KnnModel(k = 21)
    knnRegression.fit(trainX, trainY)

    testX = np.linspace(0, 1, 50).reshape([50,1])
    predict = knnRegression.predict(testX)
    plt.plot(testX, predict)
    plt.show()





if __name__ == "__main__":
    main()