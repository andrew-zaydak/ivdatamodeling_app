'''Created by: Andrew Zaydak
Email: andrew.zaydak@intrepid-vision.com
Modification date: 3 April 2020
Intrepid Vision LLC
Factorial Hidden Markov Model
'''

import numpy as np
from scipy import linalg

class FHMM:

    def __init__(self, T=None, K=None, M=1):
        self.T = T   # size of data frame
        self.K = K   # number of states
        self.M = M      # number of markov chains

        self.features = None    # number of features
        self.Mu = None  # mean vectors
        self.Cov = None # output covariance matrix
        self.P = None   # state transition probabilities matrix
        self.Pi = None  # priors (starting state probabilities)
        self.tol = 0.0001 # learning tolerance
        self.cyc = 100  # maximum learning cycles
        self.iCov = None
        self.alpha = None
        self.LL_fifo = []

    def col_mult(self, A, v):
        Z=np.zeros(A.shape)
        for i in range(A.shape[1]):
            Z[:, i] = np.multiply(A[:, i], v)
        return Z

    def spoof_data(self, time):
        p = self.Mu.shape[1]
        states = np.zeros([time,1],dtype=np.int)
        states[0] = np.random.choice(range(self.K), p=self.Pi[0,:])
        data = np.zeros([time, p], dtype=np.float)
        data[0,:] = self.Mu[states[0], :]
        for i in range(1,time):
            #print(self.P[0,:])
            #print(self.P[states[i-1],:].reshape([self.K]))
            states[i] = np.random.choice(range(self.K), p = self.P[states[i-1],:].reshape([self.K]))
            data[i,:] = self.Mu[states[i],:]
            #states[i] = np.random.choice(range(self.K), p = self.P[0,:])

        return states, data

    def multivariate_normal(self, X):
        p = X.shape[1]
        k1 = (2 * np.pi)**(-p/2)
        k2 = k1 / np.sqrt(np.linalg.det(self.Cov))
        B = np.zeros([1, self.K], dtype=np.float)
        for i in range(self.K):
            d = self.Mu[i,:] - X
            #np.reshape(d,[1, self.K])
            np.reshape(d,[1, p])
            B[0,i] = k2 * np.exp(-0.5 * np.matmul(d, np.matmul(self.iCov, np.transpose(d))))
        return B

    def rest_forward(self):
        self.LL_fifo = []
        self.alpha = np.zeros([1,self.Cov.shape[0]])

    def forward_step(self, X):

        X = np.reshape(X,[1,X.shape[0]])
        B = self.multivariate_normal(X)

        if len(self.LL_fifo)==0:
            self.alpha = np.multiply(self.Pi,B)
        else:
            self.alpha = np.multiply(np.matmul(self.alpha,self.P),B)

        scale = np.sum(self.alpha)
        self.alpha = self.alpha / scale
        LL = np.log(scale)
        self.LL_fifo.insert(0, LL)
        if len(self.LL_fifo) > self.T:
            self.LL_fifo.pop()

        average_LL = sum(self.LL_fifo)

        return np.reshape(self.alpha,[1, self.K]), average_LL

    def train(self, X, cyc=None, tol=None):
        if cyc is not None:
            self.cyc = cyc
        if tol is not None:
            self.tol = tol
        p = X.shape[1]  # number of features
        N = X.shape[0]  # number of samples
        if N % self.T != 0:
            print('ERROR: Training data should have shape[0] an even multiple of {}'.format(self.T))

        N = N / self.T  # number of training frames
        LL_per_cycle = np.zeros([int(N)])
        # init the model (cov, mu, Pi, P)
        if p == 1:
            #self.Cov = np.zeros([1,1],dtype=np.float)
            self.Cov = np.cov(X,rowvar=False).reshape([1,1])
        else:
            self.Cov = np.diag(np.diag(np.cov(X,rowvar=False))) #nto totally right
        self.Mu = np.matmul(np.random.randn(self.K, p), linalg.sqrtm(self.Cov)) + np.ones([self.K, 1]) * np.mean(X,axis=0)   #There is some issue with this....
        self.Pi = np.random.rand(1, self.K)
        self.Pi = self.Pi / np.sum(self.Pi)
        self.P = np.random.rand(self.K,self.K)
        temp_v = np.sum(self.P,axis=1)
        self.P = self.P / temp_v[:, None]

        LL = np.zeros([self.cyc], dtype=np.float)   # save the LL for each cycle
        cycle_number = -1    # track the number of cyc
        lik = 0.0

        alpha = np.zeros([self.T, self.K], dtype=np.float)    # probability of being in any state k during time t (aka FW model)
        beta = np.zeros([self.T, self.K], dtype=np.float)     # backward probability (backward model)
        gamma = np.zeros([self.T, self.K], dtype=np.float)
        B = np.zeros([self.T, self.K], dtype=np.float)        # emission probability. i.e. probability of makeing an observation given a state
                                                              # for time-series data this is modeled as a gaussian
        k1 = (2 * np.pi)**(-p/2)                              # constant part of gaussian equation
        for number_of_cycles in range(self.cyc):
            cycle_number = cycle_number + 1
            # forward - backward calculations
            Gamma = np.zeros([int(N) * self.T, self.K])
            Gammasum = np.zeros([1, self.K], dtype=np.float)
            Scale = np.zeros([self.T, 1], dtype=np.float)
            Xi = np.zeros([self.T-1, self.K*self.K], dtype=np.float)

            for n in range(int(N)):
                iCov = np.linalg.inv(self.Cov)
                k2 = k1 / np.sqrt(np.linalg.det(self.Cov))
                for i in range(self.T):
                    for l in range(self.K):
                        # compute the gaussian probabilities for each observation given the current model
                        d = self.Mu[l,:] - X[(n)*self.T+i,:]
                        B[i, l] = k2 * np.exp(-0.5 * np.matmul(d, np.matmul(iCov, np.transpose(d))))

                # compute the forward probabilities
                scale = np.zeros([self.T, 1])
                alpha[0, :] = np.multiply(self.Pi, B[0, :])
                scale[0] = np.sum(alpha[0, :])
                alpha[0, :] = alpha[0, :] / scale[0]

                for i in range(1, self.T):
                    alpha[i, :] = np.multiply(np.matmul(alpha[i-1,:], self.P), B[i,:])
                    scale[i] = np.sum(alpha[i, :])
                    alpha[i, :] = alpha[i, :] / scale[i]

                # compute the backward porbabilities
                beta[self.T-1, :] = np.ones([1, self.K]) / scale[self.T-1]
                for i in range(self.T-2, -1, -1):
                    beta[i, :] = np.matmul(np.multiply(beta[i+1, :],B[i+1, :]), np.transpose(self.P)) / scale[i]

                gamma = np.multiply(alpha,beta)
                temp_v = np.sum(gamma, axis=1)
                gamma = gamma / temp_v[:, None]
                gammasum = np.sum(gamma, axis=0)
                xi = np.zeros([self.T-1, self.K * self.K], dtype=np.float)
                for i in range(self.T-1):
                    #t = np.multiply(self.P,np.matmul(np.transpose(alpha[i, :]),np.multiply(beta[i+1, :], B[i+1, :])))       # might need to check for similar but above...
                    # there MUST be a better way to do this
                    temp_B = B[i+1,:].reshape([1,self.K])
                    temp_beta = beta[i+1,:].reshape([1,self.K])
                    temp_alpha = alpha[i,:].reshape([1,self.K])

                    tmp1 = np.multiply(temp_beta, temp_B)
                    tmp2 = np.matmul(np.transpose(temp_alpha), tmp1)
                    t = np.multiply(self.P,tmp2)
                    xi[i, :] = np.transpose(t).flatten() / np.sum(t.flatten()) #might need to transpose...
                #print("n={}".format(n))
                #print("scale={}".format(np.log(scale)))
                LL_per_cycle[int(n)] = np.sum(np.log(scale))
                Scale = Scale + np.log(scale)
                Gamma[n*self.T:(n+1)*self.T,:] = gamma    #mgiht have a lot of zeros
                Gammasum = Gammasum + gammasum
                Xi = Xi + xi

            # expectation max step
            # compute Mu
            self.Mu = np.zeros([self.K, p])
            self.Mu = np.matmul(np.transpose(Gamma),X)
            temp_v = np.sum(np.transpose(Gammasum), axis=1)
            self.Mu = self.Mu / temp_v[:, None]

            # compute transition matrix
            sxi = np.transpose(np.sum(np.transpose(Xi), axis=1))
            sxi = np.reshape(sxi,[self.K,self.K], order='F')    # Fortran style reshape is consistant with Octave.
            temp_v = np.sum(sxi, axis=1)
            self.P = sxi / temp_v[:, None]

            # prior probabilities
            self.Pi = np.zeros([1, self.K])
            for i in range(int(N)):
                self.Pi = self.Pi + Gamma[(i-1)*self.T+1,:] # inexed might be wrong

            self.Pi = self.Pi / N
            # covariance
            self.Cov = np.zeros([p, p])
            for l in range(self.K):   # check indexis
                #d = X - np.matmul(np.ones([self.T * int(N),1]), self.Mu[l, :].reshape([1,self.K]))
                d = X - np.matmul(np.ones([self.T * int(N),1]), self.Mu[l, :].reshape([1,p]))
                self.Cov = self.Cov + np.matmul(np.transpose(self.col_mult(d, Gamma[:, l])),d)

            self.Cov = self.Cov / np.sum(Gammasum)
            self.iCov = np.linalg.inv(self.Cov)
            oldlik = lik
            lik = np.sum(Scale)
            LL[number_of_cycles] = lik
            print("------------------------")
            print("cycle      = {}".format(number_of_cycles))
            print("total LL   = {}".format(lik))
            print("average LL = {}".format(lik/N))
            print("LL std = {}".format(np.std(LL_per_cycle)))

            if number_of_cycles<=2:
                likbase = lik
            elif lik < (oldlik - 1e-6):
                print("Violation")
                return LL[0:number_of_cycles], lik/N, np.std(LL_per_cycle)
            elif (lik-likbase) <(1 + self.tol) * (oldlik-likbase) or np.isnan(lik):
                return LL[0:number_of_cycles], lik/N, np.std(LL_per_cycle)
