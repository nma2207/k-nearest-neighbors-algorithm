import numpy as np

class KnnModel:
    def __init__(self, k):
        print("I was born with {0:} neighbours!".format(k))
        self.k = k

    def fit(self, trainX, trainY):
        self.trainX = np.copy(trainX)
        self.trainY = np.copy(trainY)

    def predictToOne(self, testX):
        dif = self.trainX - testX
        squareDif = np.zeros(shape=(self.trainX.shape[0],))

        for i in range(self.trainX.shape[1]):
            squareDif += dif[:,i]**2
        smallestIndex = np.argpartition(squareDif, self.k)[:self.k]
        result = np.zeros(shape=[self.trainY.shape[1]])
        for i in smallestIndex:
            result+=self.trainY[i]
        result /= self.k
        #print(squareDif)
        return result

    def predict(self, testX):
        res = []
        for i in range(testX.shape[0]):
            result = self.predictToOne(testX[i])
            res.append(result)
        return np.array(res)




