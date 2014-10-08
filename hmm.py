#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

class HMM():
    """
    Implementation of the forward, backward, Viterbi and Baum-Welch algorithms.
    """
    def __init__(self, x):
        """
        Define the observations vector (x). Class initialization expects a list
        or tuple with 1 dimension and defines an horizontal vector (array).
        Variable names that will be used later are declared but not initialized.
        """

        self.x = np.array(x)
        self.max_k = np.shape(self.x)[0]
        self.num_states = None
        self.num_emissions = None

        self.pi = None
        self.Q = None
        self.e = None

        self.alphas = None
        self.L = None
        self.betas = None
        self.M = None
        self.Y = None

        assert len(np.shape(self.x)) == 1 and self.max_k > 0

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

    def __normalize_alpha(self, alpha, k):
        """
        Helper function for the Forward algorithm.
        Normalize alpha to avoid underflow and fill the likelihood vector (L).
        """

        alpha_norm = sum(alpha) # L(k+1) / L(k)
        self.L[k] = alpha_norm
        alpha /= alpha_norm # Normalize alpha.

        return alpha

    def forward(self):
        """
        Forward algorithm. Solve the Evaluation problem.
        Each row of alphas stores the probabilities of the model being in the
        different states at step k and producing the observations from 0 to k.
        Here, normalized alphas are used, meaning the probability *given* the
        observations.
        Also return the log likelihood of the model, that is, how likely it is
        that the sequence of observations x has been produced by the model.
        """

        self.L = np.zeros(self.max_k) # Initialize the likelihood vector (L).
        alphas = np.zeros((self.max_k, self.num_states)) # "Vertical" matrix.

        alpha = self.pi * self.e[:, self.x[0]]
        alphas[0] = self.__normalize_alpha(alpha, 0)

        eps = np.finfo(np.float32).eps
        for k in xrange(1, self.max_k):
            alpha = np.dot(alphas[k-1], self.Q) * self.e[:, self.x[k]]
            alphas[k] = self.__normalize_alpha(alpha, k)

            assert abs(sum(alphas[k]) - 1.0) < eps

        return alphas, sum(np.log(self.L))
            
    def backward(self):
        """
        Complementary algorithm used by the Viterbi and Baum-Welch algorithms.
        Create a normalized betas matrix with the probabilities of being in each
        state at step k *given* the observations from k+1 to max_k.
        """

        if self.alphas is None:
            self.alphas, __ = self.forward() # Generate self.L

        betas = np.zeros((self.max_k, self.num_states)) # "Vertical" matrix.

        betas[self.max_k-1] = np.ones((1, self.num_states))
        
        Qt = self.Q.T
        eps = np.finfo(np.float32).eps
        for k in xrange(self.max_k-2, -1, -1):
            # Backward algorithm definition:
            # beta*k = (lk/lk+1) x Q . beta*k+1 x ek+1
            betas[k] = \
                np.dot(self.Q, betas[k+1] * self.e[:, self.x[k+1]]) \
                    / self.L[k+1]

            assert sum(betas[k]) > 0 \
                    and abs(sum(self.alphas[k] * betas[k]) - 1.0) < eps

        return betas

    def viterbi(self):
        """
        Viterbi algorithm. Solve the Decoding problem.
        The M matrix stores the log score for each step k.
        The Y matrix stores the states with higher log score for each step k,
        and thus contains the most probable path given the state at step k+1.
        The returned vector y contains the most probable absolute path,
        backtracked from the most probable state at the final step.
        """

        le = np.log(self.e)
        lQ = np.log(self.Q)

        # Create M scores matrix.
        # Mk+1(y)=max(z<=m){Mk(z)+log(Q(z,y))}+log(ek+1|n(y))
        # Create most probable states matrix.
        # Yk=argmax(z<=m){Mk(y)+log(Q(z,y(k+1)))}

        self.M = np.zeros((self.max_k, self.num_states)) # "Vertical" matrices.
        self.Y = np.zeros((self.max_k, self.num_states))
        self.M[0] = np.log(self.pi) + le[:, self.x[0]]

        for k in xrange(1, self.max_k):
            tmp = np.array(
                    [max(
                         zip(self.M[k-1] + lQ[:, s],
                             xrange(self.num_states)))
                     for s in xrange(self.num_states)])
            self.M[k] = tmp[:, 0] + le[:, self.x[k]]
            self.Y[k] = tmp[:, 1]

        # Backtrack through the most probable state sequence.
        y = [0] * self.max_k
        __, y[self.max_k-1] = max(
                                  zip(self.M[self.max_k-1],
                                      xrange(self.num_states))
                                  )
        y[self.max_k-1] = int(y[self.max_k-1])

        for k in xrange(self.max_k-2, -1, -1):
            y[k] = int(self.Y[k+1, y[k+1]])

        # Return the most probable sequence of states.
        return y

    def baum_welch(self, num_states, num_emissions, Qconst=None, econst=None):
        """
        Baum-Welch algorithm. Solve the Learning problem.
        The phi matrix is the scalar product of alpha and beta matrices and
        represents the probability of the model being in a particular state y at
        step k (log score is used to avoid underflow).
        start with equiprobable Q and 90-10 e
        randomly initialize Q and e, from them, take the observations and
        compute alphas and betas, from them compute the phi(k:k+1) matrices
        (there must be n-1 of them), then sum them all matrices and make them
        stochastic by dividing each value by the sum of each row, we now have a
        new Q and start over again. 
        ##phi(k:k+1|n)(i,j) = alpha*k(i) . Q(i,j) . ek+1(j) . beta*k+1(j)##
        phi(k|n)(i) = alpha*k(i) x beta*k(i)
        
        for e, sum all products of phi and e for each k and divide by n. this is
        a weighted mean. e is a vector length m, the number of states.
        e = sum(phi(k) . e(k)) / n"""
        
        # TODO: Assert constriction matrices consist only in 0 and 1 and that
        #       row sum is at least 1 (except for the last state transition)
        #       Check also impossible transition-emission combination for a
        #       given observation sequence.

        if Qconst is not None:
            Qconst = np.array(Qconst)
            assert np.shape(Qconst) == (num_states, num_states)
            #assert (Qconst == 0 or Qconst == 1).all()
            #assert (Qconst.sum(1) > 1).all()
        if econst is not None:
            econst = np.array(econst)
            assert np.shape(econst) == (num_states, num_emissions)
            #assert (econst == 0 or econst == 1).all()
            #assert (econst.sum(1) > 1).all()
        rpi = np.random.rand(num_states)
        rQ = np.random.rand(num_states, num_states)
        re = np.random.rand(num_states, num_emissions)
        self.add_pi(rpi / rpi.sum())
        self.add_Q(rQ / rQ.sum(1)[:, None])
        self.add_e(re / re.sum(1)[:, None])

        em = np.zeros((self.max_k, self.num_emissions))
        for k in xrange(self.max_k):
            em[k, self.x[k]] = 1
        assert em.sum() == self.max_k

        eps = np.finfo(np.float32).eps
        for i in xrange(1000):
            self.alphas, __ = self.forward()
            self.betas = self.backward()
            Qt = self.Q.T
            Q = np.zeros((self.num_states, self.num_states))

            for k in xrange(self.max_k-1):
                Q_k = (self.alphas[k] * Qt).T \
                        * (self.e[:, self.x[k+1]] * self.betas[k+1])
                Q_k /= self.L[k+1]
                assert abs(Q_k.sum() - 1.0) < eps 
                Q += Q_k
            assert abs(Q.sum() - (self.max_k-1)) < eps

            phi = self.alphas * self.betas
            phi_norm = phi.sum(0)
            self.pi = phi[0]
            if Qconst is not None:
                Q *= Qconst
            self.Q = Q / Q.sum(1)[:, None]
            self.e = np.dot(phi.T, em) / phi_norm[:, None]
            if econst is not None:
                e = self.e * econst
                self.e = e / e.sum(1)[:, None]

        assert abs(self.pi.sum() - 1) < eps
        assert (abs(self.Q.sum(1) - 1) < eps).all()
        assert (abs(self.e.sum(1) - 1) < eps).all()

        return self.pi, self.Q, self.e


if __name__ == '__main__':
    observations = [int(l) for l in open('test_case.txt')]
    model = HMM(observations)
    #model.add_pi((0.5, 0.5))
    #model.add_Q(((0.7, 0.3),
    #             (0.4, 0.6)))
    #model.add_e(((0.5, 0.4, 0.1),
    #             (0.1, 0.3, 0.6)))

    #print model.forward()
    #print model.backward()
    #print model.viterbi()
    #pi, Q, e = model.baum_welch(2, 3,
    #        Qconst=((1,1),
    #                (1,1)),
    #        econst=((1,1,1),
    #                (1,1,1))
    #        )
    print pi
    print Q
    print e
