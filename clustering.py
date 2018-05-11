import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np; np.random.seed(0)
import seaborn as sns
sns.set()
from sklearn.decomposition import NMF

np.set_printoptions(threshold=np.nan)


def generate_data(n, d, isPoisson=True):
    data = np.zeros((n, d))
    mu, sig = 0, 0.5
    lmda = 10
    if not isPoisson:
        for i in range(d):
            data[:, i] = np.random.normal(mu, sig, n)
    else:
        for i in range(d):
            data[:, i] = np.random.poisson(lmda, n)

    model = NMF(n_components=100, init='random', random_state=0)
    W = model.fit_transform(data)
    H = model.components_
    print('WWW')
    sns.heatmap(data)
    plt.show()
    np.save('./result/W', W)
    np.save('./result/H', H)
    np.save('./result/X', data)
    return data, W, H


def load_data(dir):
    data = pd.read_csv(os.path.join(dir, 'data.csv'), header=None)
    return data.value


def l2_dist(A, B):
    return np.linalg.norm(A - B, axis=1)


def update_U(X, U, V, labels, label2alpha, learning_rate):
    print('Update U')
    U_p = np.zeros(U.shape)
    sum_dL1 = 0
    dL2 = 0
    for i in range(U.shape[0]):
        # print('aaa', labels[i])
        label = labels[i]
        alphas = label2alpha[label]

        for k in range(U.shape[1]):
            sum_dL1 = 0
            for j in range(X.shape[1]):
                sum_dL1 += X[i][j] * V[j][k] / ((np.dot(U[i], np.reshape(V[j], (-1, 1)))) + 10e-16) - V[j][k]

            dL2 = ((alphas[0][k] - 1) / (U[i][k] + 10e-16)) - (1 / (alphas[1][k] + 10e-16))
            deriv = sum_dL1 + dL2
            U_p[i][k] = U[i][k] - learning_rate * deriv

    return U_p, sum_dL1, dL2


def update_V(X, U, V, betas, learning_rate):
    print('Update V')
    V_p = np.zeros(V.shape)
    sum_dL1 = 0
    dL2 = 0
    for j in range(V.shape[0]):
        for k in range(V.shape[1]):
            sum_dL1 = 0
            for i in range(X.shape[0]):
                sum_dL1 += X[i][j] * U[i][k] / ((np.dot(U[i], np.reshape(V[j], (-1, 1)))) + 10e-16) - U[i][k]
            dL2 = (betas[0][k] - 1) / (V[j][k] + 10e-16) - 1 / (betas[1][k] + 10e-10)
            deriv = sum_dL1 + dL2
            V_p[j][k] = V[j][k] - learning_rate * deriv
    return V_p, sum_dL1, dL2


class EM(object):
    def __init__(self, k, c=5, max_iter=100, learning_rate=1e-30):
        self.X = np.load('./result/X.npy')
        self.k = k
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.c = c
        self.max_iter = max_iter
        self.lr = learning_rate
        self.U = np.load('./result/W.npy')
        self.V = np.load('./result/H.npy').T
        # print('U: ', self.U.shape)
        # print('V: ', self.V.shape)

    def init_parameters(self):
        labels = np.random.randint(0, self.c, size=self.n, dtype=int)
        self.labels_prob = np.ones((self.n, self.c)) / self.c
        label2alpha = {}
        for curr_label in range(self.c):
            curr_alpha = np.random.random((2, self.k))
            label2alpha[curr_label] = curr_alpha

        self.label2alpha = label2alpha
        self.betas = np.random.random((2, self.k)) * 2
        U = np.zeros((self.n, self.k))
        V = np.zeros((self.d, self.k))

        # for k in range(self.k):
        #     for i in range(self.n):
        #         alphas = label2alpha[labels[i]]
        #         U[i][k] = np.random.gamma(alphas[0][k], alphas[1][k])
        #     for j in range(self.d):
        #         V[j][k] = np.random.gamma(self.betas[0][k], self.betas[1][k])
        # self.V = V
        # self.U = U
        self.labels = labels
        return

    def train(self):
        self.init_parameters()
        Us = []
        Vs = []
        errs = []
        for iter in range(self.max_iter):
            err = np.sum(l2_dist(self.X, np.dot(self.U, self.V.T)))
            Us.append(self.U)
            Vs.append(self.V)
            errs.append(err)
            # if iter and iter % 10 == 0:
            print("Iter {} with loss {}.".format(iter, err))

            # update U, V
            print('Iteration %s start to precess' % str(iter))
            self.V, L1_V, L2_V = update_V(self.X, self.U, self.V, self.betas, self.lr)
            print('V: ', self.V)
            print('L1_V: ', L1_V)
            print('L2_V: ', L2_V)
            self.U, L1_U, L2_U = update_U(self.X, self.U, self.V, self.labels, self.label2alpha, self.lr)
            print('U: ', self.U)
            print('L1_U: ', L1_U)
            print('L2_U: ', L2_U)
            U_results = [self.U, L1_U, L2_U]
            V_results = [self.V, L1_V, L2_V]
            print('Save result')
            filename_U = './result/U_' + str(iter) + '.pickle'
            filename_V = './result/V_' + str(iter) + '.pickle'
            # with open(filename_U, 'wb') as f:
            #     pickle.dump(U_results, f)
            # with open(filename_V, 'wb') as f:
            #     pickle.dump(V_results, f)



            self.centroids = []

            # update cluster centroids
            print('Update centroids')
            X_pred = np.dot(self.U, self.V.T)
            sums = np.zeros((self.c, self.d))
            counters = np.zeros((self.c, 1))

            for i in range(self.n):
                index = self.labels[i]
                counters[index] += 1
                sums[index] += X_pred[i]
            self.centroids = sums / counters

            # re-assign cluster
            for i in range(self.n):
                self.labels[i] = np.argmin(l2_dist(self.centroids, self.X[i]))
                # self.soft_clusters[i] =

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
    # X, W, H = generate_data(100, 500)
    # print(X.shape)
    Model = EM(k=100)
    X_pred, U, V, err = Model.train()
    # plot(err)




