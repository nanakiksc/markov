#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy

class ForBack():
    """
    Implementation of the forward and backward algorithms.
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
        alphas = self.forward(max_k)
        betas = self.backward(max_k, alphas)
        phi = alphas * betas
        lphi = log(phi)
        lQ = numpy.log(self.Q)

        #there
        #Mk+1(y)=argmax(z<=m){Mk(z).Q(z,y)}+log(phik+1|n(y))
        #and back again
        #yk=argmax(z<=m){Mk(y)+log(Q(z,y+1))}
        

if __name__ == '__main__':
    fb = ForBack(
            pi=(0.5, 0.5),
            e=(0.3, 0.7),
            Q=1,
            x=[int(l) for l in open('simulated_markov_exercise.txt')]
            )
    #for i in xrange(len(x)):
    #    print forward(i)
    #print fb.forward(1000)
    print fb.backward(1000)
