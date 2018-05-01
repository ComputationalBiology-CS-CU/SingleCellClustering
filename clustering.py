import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data(n, d):
    data = np.zeros((n, d))
    mu, sig = 0, 0.5
    for i in range(d):
        data[:, i] = np.random.normal(mu, sig, n)
    return data


def load_data(dir):
    data = pd.read_csv(os.path.join(dir, 'data.csv'), header=None)
    return data.value


def l2_dist(A, B):
    return np.linalg.norm(A - B, axis=1)


def update_U(X, U, V, alphas, learning_rate):
    U_p = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for k in range(U.shape[1]):
            sum_temp = 0
            for j in range(X.shape[1]):
                sum_temp += X[i][j] * V[j][k] / (np.dot(U[i], np.reshape(V[j], (-1, 1)))) - V[j][k]
            deriv = sum_temp + (alphas[0][k] - 1) / U[i][k] - 1 / alphas[1][k]
            U_p[i][k] = U[i][k] - learning_rate * deriv
    return U_p


def update_V(X, U, V, betas, learning_rate):
    V_p = np.zeros(V.shape)
    for j in range(V.shape[0]):
        for k in range(V.shape[1]):
            sum_temp = 0
            for i in range(X.shape[0]):
                sum_temp += X[i][j] * U[i][k] / (np.dot(U[i], np.reshape(V[j], (-1, 1)))) - U[i][k]
            deriv = sum_temp + (betas[0][k] - 1) / (V[j][k] + 10e-10) - 1 / (betas[1][k] + 10e-10)
            V_p[j][k] = V[j][k] - learning_rate * deriv
    return V_p


# class Cluster(object):
#     def __init__(self, alphas):
#         self.alphas = alphas
#         U = np.zeros((self.n, self.k))
#         for i in range(self.n):
#             for j in range(self.k):
#                 U[i][j] = np.random.gamma(alphas[0][j], alphas[1][j])
#         self.U = U


class EM(object):
    def __init__(self, X, k, c=5, max_iter=100, learning_rate=10E-10):
        self.X = X
        self.k = k
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.c = c
        self.max_iter = max_iter
        self.lr = learning_rate


    def init_parameters(self):
        self.clusters = np.random.randint(0, self.c, size=self.n, dtype=int)
        # for i in range(self.c):
        #     alphas = np.random.random((2, self.k))
        #     cluster = Cluster(alphas)
        #     clusters.append(cluster)
        # self.clusters = clusters
        self.alphas = np.random.random((2, self.k)) * 2
        self.betas = np.random.random((2, self.k)) * 2
        V = np.zeros((self.d, self.k))
        U = np.zeros((self.n, self.k))
        for k in range(self.k):
            for i in range(self.n):
                U[i][k] = np.random.gamma(self.alphas[0][k], self.alphas[1][k])
            for j in range(self.d):
                V[j][k] = np.random.gamma(self.betas[0][k], self.betas[1][k])
        self.V = V
        self.U = U
        return

    def train(self):
        self.init_parameters()
        Us = []
        Vs = []
        errs = []
        for iter in range(self.max_iter):
            # update U, V
            self.V = update_V(self.X, self.U, self.V, self.betas, self.lr)
            self.U = update_U(self.X, self.U, self.V, self.alphas, self.lr)

            # update cluster centroids
            self.centroids = []
            X_pred = np.dot(self.U, self.V.T)
            sums = np.zeros((self.c, self.d))
            counters = np.zeros((self.c, 1))

            for i in range(self.n):
                index = self.clusters[i]
                counters[index] += 1
                sums[index] += X_pred[i]
            self.centroids = sums / counters

            # re-assign cluster
            for i in range(self.n):
                self.clusters[i] = np.argmin(l2_dist(self.centroids, self.X[i]))

            err = np.sum(l2_dist(self.X, X_pred))
            Us.append(self.U)
            Vs.append(self.V)
            errs.append(err)
            # if iter and iter % 10 == 0:
            print("Iter {} with loss {}.".format(iter, err))
        return X_pred, Us, Vs, errs


def plot(err):
    plt.plot(err)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Test")
    plt.legend()
    plt.savefig('error.png')
    plt.show()



if __name__ == '__main__':
    X = generate_data(50, 500)
    # print(X.shape)
    Model = EM(X, k=100)
    X_pred, U, V, err = Model.train()
    plot(err)




