import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import pickle
import numpy as np
import seaborn as sns
from sklearn.decomposition import NMF
sns.set()
np.set_printoptions(threshold=np.nan)
np.random.seed(0)

# learning_rate = 0.01


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


def update(err, U, V):
    return (err, U, V)


class EM(object):
    def __init__(self, X, k, c=5, max_iter=100, learning_rate=1e-10):
        self.X = X
        self.k = k  # reduced dimension
        self.n = self.X.shape[0]  # sample number
        self.d = self.X.shape[1]  # dimensions
        self.c = c  # cluster number
        self.max_iter = max_iter
        self.lr = learning_rate
        # load U and V generated as W and H using NMF in generate_data()
        self.U = np.load('./result/W.npy')
        self.V = np.load('./result/H.npy').T
        # print('U: ', self.U.shape)
        # print('V: ', self.V.shape)

    def init_parameters(self):
        # randomly assign label to each sample
        self.labels = np.random.randint(0, self.c, size=self.n, dtype=int)
        # used for soft clustering
        self.labels_prob = np.ones((self.n, self.c)) / self.c
        # a map from cluster label to parameter alphas for U's prior
        label2alpha = {}
        for curr_label in range(self.c):
            # randomly initialize each cluster's parameter alphas for U's prior
            curr_alpha = np.random.random((2, self.k))
            label2alpha[curr_label] = curr_alpha
        self.label2alpha = label2alpha
        # randomly initialize parameter betas set for V's prior
        self.betas = np.random.random((2, self.k)) * 2
        return

    def train(self):
        # initialize labels, U's and V's parameters
        self.init_parameters()
        # X = tf.placeholder("float")
        print("Set model variables U and V")
        U_tf = tf.Variable(self.U, name='U')
        V_tf = tf.Variable(self.V, name='V')
        print("Construct prediction of X")
        X_pred = np.dot(self.U, self.V.T)
        print("Construct Likelihood")
        likelihood = tf.reduce_sum(tf.multiply(self.X, tf.log(tf.matmul(U_tf, tf.transpose(V_tf)))) - tf.matmul(U_tf, tf.transpose(V_tf)) - (tf.lgamma(tf.add(self.X, 1))))
        print("Construct Prior of U and V")
        U_prior = 0
        V_prior = 0
        for k in range(self.k):
            for i in range(self.n):
                label = self.labels[i]
                alphas = self.label2alpha[label]
                U_prior += -tf.lgamma(alphas[0][k]) - tf.log(tf.pow(alphas[1][k], alphas[0][k])) + (alphas[0][k] - 1) * tf.log(self.U[i][k]) - self.U[i][k] / alphas[1][k]
            for j in range(self.d):
                beta0 = self.betas[0][k]
                beta1 = self.betas[1][k]
                V_prior += -tf.lgamma(beta0) - tf.log(tf.pow(beta1, beta0)) + (beta0 - 1) * tf.log(self.V[j][k]) - self.V[j][k] / beta1
        print("Construct Regularization of U and V")
        U_Reg = tf.nn.l2_loss(U_tf)
        V_Reg = tf.nn.l2_loss(V_tf)
        print("Construct Cost by sum them all")
        cost = likelihood + U_prior + V_prior + U_Reg + V_Reg
        print("Get the updated cost, U, V")
        res = update(cost, U_tf, V_tf)
        print("Optimize the cost")
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(cost)
        errs = []
        print("Initialize global variables")
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            print("Session run")
            sess.run(init)

            for iter in range(self.max_iter):
                sess.run(optimizer)
                training_cost, self.U, self.V = sess.run(res)
                print("Iteration ", iter, ":", training_cost)
                errs.append(training_cost)

        # # if iter and iter % 10 == 0:
        # print("Iter {} with loss {}.".format(iter, err))

                self.centroids = []
                # update cluster centroids
                print('Update centroids')
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

        return errs


def plot(err):
    plt.plot(err)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Test")
    plt.legend()
    plt.savefig('error.png')
    plt.show()


if __name__ == '__main__':
    # data, W, H = generate_data(100, 500)
    # print(data.shape)
    data = np.load('./result/X.npy')
    Model = EM(X=data, k=100, learning_rate=1e-20)
    err = Model.train()
    # plot(err)




