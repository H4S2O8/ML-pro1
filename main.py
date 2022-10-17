import sklearn
from sklearn.datasets import load_iris
import pandas
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split


class LogisticModel:
    # this model only need to judge if the watermelon is good or not, so only need 1 dimension of output.
    output_dim = 1

    def __init__(self, ori_x: np.ndarray, ori_y: np.ndarray):
        """
        :param ori_x: original input matrix X
        :param ori_y: original output matrix Y
        """
        self.x_hat: np.ndarray = np.c_[ori_x, np.ones(ori_x.shape[0])]
        self.beta: np.ndarray = np.random.rand(ori_x.shape[1] + 1, LogisticModel.output_dim)
        self.y = ori_y.reshape(-1, 1)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        ans = 1 / (1 + np.exp(-x))
        return ans

    def _cost(self):
        Lbeta = -self.y * np.dot(self.x_hat, self.beta) + np.log(1 + np.exp(np.dot(self.x_hat, self.beta)))
        return Lbeta.sum()

    def gradient(self):
        p1 = self.sigmoid(np.dot(self.x_hat, self.beta))
        gra = -np.dot(self.x_hat.T, self.y - p1)
        return gra

    def fit(self, it: int):
        for i in range(it):
            self.beta = self.beta - 0.05 * self.gradient()
            # if i%10 == 0:
            #     print(f"cost={self._cost()}")

    def predict(self, new_x: np.ndarray):
        x_hat = np.c_[new_x, np.ones(new_x.shape[0])]
        ans = np.dot(x_hat, self.beta)
        ans[ans >= 0.5] = 1
        ans[ans < 0.5] = 0
        return ans

    @staticmethod
    def evaluate(ans: np.ndarray, true_y: np.ndarray):
        print("acc:", (true_y - ans)[true_y == ans].size / true_y.size)
        print("square loss:", np.linalg.norm(true_y - ans) / true_y.size)
        if true_y.size <= 2:
            print("log loss:", sklearn.metrics.log_loss(true_y, ans))


# lm = LogisticModel(x, y)
# lm.predict(x)

def vote(vote_table: np.ndarray):
    vote_table = vote_table.reshape(-1, 3)
    ans = np.zeros(vote_table.shape[0])
    for i in range(vote_table.shape[0]):
        if vote_table[i, 0] == 0 and vote_table[i, 2] == 1:
            ans[i] = 1
        elif vote_table[i, 0] == 1 and vote_table[i, 1] == 1:
            ans[i] = 2
        elif vote_table[i, 1] == 0 and vote_table[i, 2] == 0:
            ans[i] = 0
        else:
            ans[i] = 0
    return ans.reshape(-1, 1)


if __name__ == '__main__':
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X1, y1 = X[y != 0], y[y != 0]  # 1 or 2
    y1[y1 == 1] = 0
    y1[y1 == 2] = 1
    y1 = y1.reshape(-1, 1)
    X2, y2 = X[y != 1], y[y != 1]  # 0 or 2
    y2[y2 == 0] = 0
    y2[y2 == 2] = 1
    y2 = y2.reshape(-1, 1)
    X3, y3 = X[y != 2], y[y != 2]  # 0 or 1
    y3[y3 == 0] = 0
    y3[y3 == 1] = 1
    y3 = y3.reshape(-1, 1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)

    lm_iris_1 = LogisticModel(X1_train, y1_train)
    lm_iris_2 = LogisticModel(X2_train, y2_train)
    lm_iris_3 = LogisticModel(X3_train, y3_train)

    lm_iris_1.fit(5000)
    lm_iris_2.fit(5000)
    lm_iris_3.fit(5000)

    # p_iris_1 = lm_iris_1.predict(X1_test, y1_test)
    # p_iris_2 = lm_iris_2.predict(X2_test, y2_test)
    # p_iris_3 = lm_iris_3.predict(X3_test, y3_test)

    y = y.reshape(-1, 1)
    p_iris_all_1 = lm_iris_1.predict(X)
    p_iris_all_2 = lm_iris_2.predict(X)
    p_iris_all_3 = lm_iris_3.predict(X)

    p_iris = np.c_[p_iris_all_1, p_iris_all_2, p_iris_all_3]
    vote_ans = vote(p_iris)
    LogisticModel.evaluate(vote_ans, y)