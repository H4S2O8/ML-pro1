import sklearn
from sklearn.datasets import load_iris
import pandas
import numpy as np
from pandas import Series, DataFrame

class LogisticModel:
    # this model only need to judge if the watermelon is good or not, so only need 1 dimension of output.
    output_dim = 1
    def __init__(self, ori_x:np.ndarray, ori_y:np.ndarray):
        """
        :param ori_x: original input matrix X
        :param ori_y: original output matrix Y
        """
        self.x_hat:np.ndarray = np.c_[ori_x,np.ones(ori_x.shape[0])]
        self.beta:np.ndarray = np.random.rand(ori_x.shape[1]+1, LogisticModel.output_dim)
        self.y = ori_y.reshape(-1,1)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        ans = 1 / (1 + np.exp(-x))
        return ans

    def gradient(self):
        p1 = self.sigmoid(np.dot(self.x_hat, self.beta))
        gra = -np.dot(self.x_hat.T, self.y - p1)
        return gra.reshape(-1,1)

    def fit(self, it:int):
        for i in range(it):
            self.beta = self.beta - self.gradient()



    def predict(self, new_x:np.ndarray):
        x_hat = np.c_[new_x,np.ones(new_x.shape[0])]
        ans = np.dot(x_hat, self.beta)
        ans[ans>=0.5] = 1
        ans[ans<0.5] = 0
        return ans


lm = LogisticModel(x,y)
lm.predict(x)
