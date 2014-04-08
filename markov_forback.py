#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy

class ForBack():
    """
    Implementation of the forward, backward and Viterbi algorithms.
    """

    def __init__(self, pi, e, Q, x):
        # Probability of being at state pi[i] at time 0.
        self.pi = numpy.array(list(pi))
        # Emission probabilities.
        self.e = e
        # Non-transition probabilities.
        self.p, self.q = 0.9, 0.8
        # Transition matrix.
        self.Q = numpy.array(
                [
                    [self.p, 1-self.p],
                    [1-self.q, self.q]
                    ]
                )
        # Series of observations.
        self.x = list(x)
        self.L = []

    def g(self, k): # Needs to return column vectors.
        ##### FIX: Generalize to multiple emissions. #####
        if self.x[k] == 0:
            # Probabilities of NOT emitting.
            em = [1-e for e in self.e]
        elif self.x[k] == 1:
            # Probabilities of emitting.
            em = [e for e in self.e]
        return numpy.array(em)

    def normalize_alpha(self, alpha):
        alpha_norm = sum(alpha) # L(k) / L(k+1)
        self.L.append(alpha_norm)
        alpha /= alpha_norm # Normalize alpha.

        return alpha

    def forward(self, max_k):
        alphas = numpy.zeros((max_k, 2))

        alpha = self.pi * self.g(0)
        alphas[0] = self.normalize_alpha(alpha)

        for k in xrange(1, max_k):
            alpha = numpy.dot(alpha, self.Q) * self.g(k)
            alphas[k] = self.normalize_alpha(alpha)

        #return sum(numpy.log(self.L)) # Should return -687.977398651
        return alphas
            
    def backward(self, max_k, alphas=None):
        if alphas is None:
            alphas = self.forward(max_k)

        betas = numpy.zeros((max_k, 2))
        max_k -= 1

        beta = numpy.ones((1, 2))
        betas[max_k] = beta
        
        Qt = numpy.transpose(self.Q)
        for k in xrange(max_k-1, -1, -1):
            #beta*k = (lk/lk+1) x Q . beta*k+1 x gk+1
            beta = numpy.dot(beta, Qt) * self.g(k) / self.L[k] 
            betas[k] = beta

        return betas

    def viterbi(self, max_k):
        """
        Viterbi algorithm.
        The phi matrix is the scalar product of alpha and beta matrices and
        represents the probability of the model being in a particular state y at
        step k (use log score to avoid underflow issues).
        The M matrix stores the log score for each step k.
        The Y vector stores the states with higher log score for each step k,
        and thus represents the most probable path.
        """

        alphas = self.forward(max_k)
        betas = self.backward(max_k, alphas)
        phi = alphas * betas
        #lphi = numpy.log(phi)
        #le = numpy.log(self.e)
        lQ = numpy.log(self.Q)
        num_states = numpy.ndim(lQ)

        # Create M scores matrix. (There...)
        # Mk+1(y)=max(z<=m){Mk(z)+log(Q(z,y))}+log(phik+1|n(y))

        # Initialize M scores matrix.
        M = numpy.zeros((max_k, 2))
        M[0] = numpy.log(self.g(0))#lphi[0]
        #M[0], curr_state = max(((lphi[0][s], s) for s in xrange(num_states)))
        
        # And fill the M scores matrix.
        for k in xrange(1, max_k):
            M[k] = [max((M[k-1][z] + lQ[z][y]
                         for z in xrange(num_states)))
                    for y in xrange(num_states)] + numpy.log(self.g(k))

        # Create most probable states matrix. (... and back again)
        # yk=argmax(z<=m){Mk(y)+log(Q(z,y(k+1)))}

        # Initialize Y states vector.
        Y = [0] * max_k
        max_k -= 1
        __, Y[max_k] = max(((M[max_k][y], y) for y in xrange(num_states)))

        # And fill the Y states vector.
        for k in xrange(max_k-1, -1, -1):
            __, Y[k] = max(((M[k][y] + lQ[y][Y[k+1]], y)
                            for y in xrange(num_states)))

        # Return the most probable sequence of states.
        return Y


if __name__ == '__main__':
    observations = [int(l) for l in open('simulated_markov_exercise.txt')]
    fb = ForBack(
            pi=(0.5, 0.5),
            e=(0.001, 0.999),
            Q=None,
            x=observations)
    #for i in xrange(len(x)):
    #    print forward(i)
    #print fb.forward(1000)
    #viterbi_output = fb.viterbi(1000)
    #print viterbi_output
    viterbi_output = fb.viterbi(len(observations))
    comparison = [viterbi_output[i] == observations[i] for i in xrange(len(observations))]
    t, f = 0, 0
    for bol in comparison:
        if bol:
            t += 1
        else:
            f += 1
    print t, f, t+f
