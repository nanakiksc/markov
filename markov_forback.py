#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy

class ForBack():
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

    def g(self, k): # Needs to return column vectors.
        # Generalize to multiple emissions.
        if self.x[k] == 0:
            # Probabilities of NOT emitting.
            em = [1-e for e in self.e]
        elif self.x[k] == 1:
            # Probabilities of emitting.
            em = [e for e in self.e]
        return numpy.array(em)

    def forward(self, kay):
        log_list = []

        alpha = self.pi * self.g(0)
        alpha_norm = sum(alpha)
        log_list.append(numpy.log(alpha_norm))
        alpha /= alpha_norm
        for k in xrange(1,kay):
            alpha = numpy.dot(alpha, self.Q) * self.g(k)
            
            # Normalize alpha.
            alpha_norm = sum(alpha)
            log_list.append(numpy.log(alpha_norm))
            alpha /= alpha_norm

        return sum(log_list)
            
    """
        # Without normalizing alpha.
        if k == 0:
            alpha = pi * g(k)
        elif k > 0:
            alpha *= numpy.dot(Q, g(k))
            forward(k-1, alpha)
        return alpha
    """

    def backward(self, k, n):
        #beta*k = (lk/lk+1) x Q . beta*k+1 x gk+1
        pass
        

if __name__ == '__main__':
    fb = ForBack(
            pi=(0.5, 0.5),
            e=(0.3, 0.7),
            Q=1,
            x=[int(l) for l in open('simulated_markov_exercise.txt')]
            )
    #for i in xrange(len(x)):
    #    print forward(i)
    print fb.forward(1000)
