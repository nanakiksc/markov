#!/usr/bin/env python
#-*- coding:utf-8 -*-

# FIX g() function, generalize to multiple emissions.
# Generalize also to multiple states.

import numpy as np

class HMM():
    """
    Implementation of the forward, backward, Viterbi and Baum-Welch algorithms.
    """
    def __init__(self, x):
        """
        Define the observations vector (x). Class initialization expects a list
        or tuple with 1 dimension and defines an horizontal vector (array).
        """
        
        self.x = np.array(x)
        self.num_states = None
        self.num_emissions = None
        #self.L = [] # Likelihood vector to be filled by forward().

        assert len(np.shape(self.x)) == 1

    def add_pi(self, pi):
        """
        Define the initial probabilities matrix (pi) of the model. That is, the
        probability of being at state pi[i] at time 0. Expects a list or tuple
        with 1 dimension and defines an horizontal vector (array).
        """

        self.pi = np.array(pi)

        if self.num_states is None:
            self.num_states == np.shape(self.pi)[0]
        else:
            assert np.shape(self.pi) == (self.num_states,)

    def add_Q(self, Q):
        """
        Define the state transition probabilities matrix (pi) of the model.
        Expects an M x M list or tuple and defines a square array.
        """

        self.Q = np.array(Q)

        x, y = np.shape(self.Q)
        assert x == y
        if self.num_states is None:
            self.num_states = x
        else:
            assert x == self.num_states

    def add_e(self, e):
        """
        Define the emission probabilities matrix (e) of the model. Expects an
        num_states x num_emissions list or tuple and defines an M x N array.
        """

        self.e = np.array(e)

        x, y = np.shape(self.e)
        if self.num_states is None:
            self.num_states = x
        else:
            assert x == self.num_states
        if self.num_emissions is None:
            self.num_emissions = y
        else:
            assert y == self.num_emissions

    def check_hmm(self):
        """
        Make some assertions about the sizes and values of the model matrices.
        """
        assert 1

    def g(self, k): # Needs to return column vectors. I don't think so...
        """
        Return the probabilities of emmiting x[k] at step k for each state.
        In the multiple emissions version, it should just query the emission
        matrix e, or do ir directly from the caller function.
        """
        ##### FIX: Generalize to multiple emissions. #####
        if self.x[k] == 0:
            # Probabilities of NOT emitting.
            em = [1-e for e in self.e]
        elif self.x[k] == 1:
            # Probabilities of emitting.
            em = [e for e in self.e]
        return np.array(em)

    def normalize_alpha(self, alpha):
        """
        Helper function for the Forward algorithm.
        Normalize alpha to avoid underflow and fill the likelihood vector L.
        """

        alpha_norm = sum(alpha) # L(k) / L(k+1)
        self.L.append(alpha_norm)
        alpha /= alpha_norm # Normalize alpha.

        return alpha

    def forward(self, max_k=len(self.x)):
        """
        Forward algorithm. Solve the Evaluation problem.
        Each row of alphas stores the probabilities of the model being in the
        different states at step k and producing the observations from 0 to k.
        Here, normalized alphas are used, meaning the probability *given* the
        observations.
        Also return the log likelihood of the model tested, that is, how likely
        it is that the model produced the sequence of observations x.
        """
        self.L = [] # Reinitialize self.L just in case.
        alphas = np.zeros((max_k, 2))

        alpha = self.pi * self.g(0)
        alphas[0] = self.normalize_alpha(alpha)

        for k in xrange(1, max_k):
            alpha = np.dot(alpha, self.Q) * self.g(k)
            alphas[k] = self.normalize_alpha(alpha)

        return alphas, sum(np.log(self.L))
            
    def backward(self, max_k=len(self.x)):
        """
        Complementary algorith used by viterbi() and baum_welsch().
        Create a betas matrix that stores.
        """
        if not self.L:
            self.forward(max_k) # Generate self.L

        betas = np.zeros((max_k, 2))
        max_k -= 1

        beta = np.ones((1, 2))
        betas[max_k] = beta
        
        Qt = np.transpose(self.Q)
        for k in xrange(max_k-1, -1, -1):
            # Backward algorithm definition:
            # beta*k = (lk/lk+1) x Q . beta*k+1 x gk+1
            beta = np.dot(beta, Qt) * self.g(k) / self.L[k] 
            betas[k] = beta

        return betas

    def viterbi(self, max_k=len(self.x)):
        """
        Viterbi algorithm. Solve the Decoding problem.
        The phi matrix is the scalar product of alpha and beta matrices and
        represents the probability of the model being in a particular state y at
        step k (log score is used to avoid underflow issues).
        The M matrix stores the log score for each step k.
        The Y vector stores the states with higher log score for each step k,
        and thus represents the most probable path.
        """

        alphas, __ = self.forward(max_k)
        betas = self.backward(max_k)
        phi = alphas * betas
        #lphi = np.log(phi)
        #le = np.log(self.e)
        lQ = np.log(self.Q)
        num_states = np.ndim(lQ)

        # Create M scores matrix. (There...)
        # Mk+1(y)=max(z<=m){Mk(z)+log(Q(z,y))}+log(phik+1|n(y))

        # Initialize M scores matrix.
        M = np.zeros((max_k, 2))
        M[0] = np.log(self.g(0))#lphi[0]
        #M[0], curr_state = max(((lphi[0][s], s) for s in xrange(num_states)))
 
        # And fill the M scores matrix.
        for k in xrange(1, max_k):
            M[k] = [max((M[k-1][z] + lQ[z][y]
                         for z in xrange(num_states)))
                    for y in xrange(num_states)] + np.log(self.g(k))

        # Create most probable states matrix. (... and back again)
        # yk=argmax(z<=m){Mk(y)+log(Q(z,y(k+1)))}

        # Initialize Y states vector.
        Y = [0] * max_k
        max_k -= 1
        __, Y[max_k] = max(((M[max_k][y], y) for y in xrange(num_states)))

        # And fill the Y states vector.
        # If there are equally probable states, take the one with the lowest
        # index. This is somewhat arbitrary, maybe it is better to randomize the
        # choice.
        for k in xrange(max_k-1, -1, -1):
            __, Y[k] = max(((M[k][y] + lQ[y][Y[k+1]], y)
                            for y in xrange(num_states)))

        # Return the most probable sequence of states.
        return Y

    def baum_welch(self, max_k=len(self.x)):
        """ Solve the Learning problem """
        """start with equiprobable Q and 90-10 e"""
        """randomly initiaize Q and e, from them, take the observations and
        compute alphas and betas, from them compute the phi(k:k+1) matrices
        (there must be n-1 of them), then sum them all matrices and make them
        stochastic by dividing each value by the sum of each row, we now have a
        new Q and start over again. 
        phi(k:k+1|n)(i,j) = alpha*k(i) . Q(i,j) . gk+1(j) . beta*k+1(j)
        
        for e, sum all products of phi and e for each k and divide by n. this is
        a weighted mean. e is a vector length m, the number of states.
        e = sum(phi(k) . g(k)) / n"""

        alphas, __ = self.forward(max_k)
        betas = self.backward(max_k)
        
        phis = np.zeros((2, 2))
        es = np.zeros(2)
        k=56
        print alphas
        print self.Q
        print self.g(k+1)
        print betas
        """
        for k in xrange(0, max_k-1):
            phi_k = np.dot(alphas[k], self.Q) * self.g(k+1) * betas[k+1]
            phis += phi_k

            e = np.dot(phi_k, self.g(k))
            es += e
        """
        #self.Q = phis / sum(phis)
        #self.e = es / max_k

        #print self.Q
        #print self.e

if __name__ == '__main__':
    observations = [int(l) for l in open('simulated_markov_exercise.txt')][:1000]
    fb = HMM(observations)

    print fb.backward(len(observations))
