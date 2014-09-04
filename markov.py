
#!/usr/bin/env python
#-*- coding:utf-8 -*-

from random import random
import numpy

class Chain:
    """
    Create a first order discrete Markov chain that stores hidden states with
    different emission and transition probabilities.
    """

    def __init__(self, initial_probs=[]):
        """
        Each state is represented by an index in the self.emissions and
        self.transitions lists. Thus the state 'n' contains a list of its
        emission probabilities at self.emissions[n] and a list of its
        transition probabilities at self.transitions[n]. The probability
        of the chain starting at state 'n' (before the first transition is done)
        is given by self.initial_probs[n]. If self.initial_probs is an empty
        list (as by default), all states are equally probable.
        """

        self.initial_probs = [p for p in initial_probs]
        self.emissions = []
        self.transitionsx = []
        self.num_emissions = 0
        self.num_states = 0
        self.current_state = None

    def add_state(self, emission_probability, transition_probability):
        """
        Each state is added with 2 arguments. The first argument are the
        probabilities of the different emissions and is passed as a list or
        tuple of length equal to the number of different emissions. The second
        element is the row of the transition matrix corresponding to the state
        and is passed as a list or tuple of lenght equal to the number of
        different states.
        """

        ep = [float(p) for p in emission_probability]
        self.emissions.append(ep)

        tp = [float(p) for p in transition_probability]
        self.transitions.append(tp)

        self.num_states += 1

    def check_chain(self):
        """
        Check whether the size and values of the emission and
        transition matrices make sense.
        """

        if self.initial_probs:
            assert sum(self.initial_probs) == 1
            assert len(self.initial_probs) == len(self.transitions)

        for i, state in enumerate(self.emissions):
            assert sum(state) == 1
            assert len(state) == self.num_emissions
        assert i + 1 == self.num_states

        for i, state in enumerate(self.transitions):
            assert sum(state) == 1
            assert len(state) == self.num_states
        assert i + 1 == self.num_states

    def emit(self):
        """
        Make an emission based on the probabilities of the different emissions
        in the current state.
        """

        r = random()
        acc = 0
        for em, pr in enumerate(self.emissions[self.current_state]):
            if acc <= r < acc + pr:
                return em
            acc += pr

    def transit(self, state_probs):
        """
        Update current state by choosing a state 'st', with probability 'pr',
        among all possible states, including the current state itself.
        """

        r = random()
        acc = 0
        for st, pr in enumerate(state_probs):
            if acc <= r < acc + pr:
                self.current_state = st
                break
            acc += pr

    def run(self, steps=-1):
        """
        Check that everything makes sense, initialize the chain and run it
        'steps' times. If 'steps' is omitted or negative, the chain will run
        infinitely.
        """

        self.check_chain()

        # Initalize current state of the Markov Chain.
        if self.initial_probs:
            self.transit(self.initial_probs)
        else:
            self.current_state = int(random() * self.num_states)

        steps = int(steps)
        while steps:
            print self.emit()
            self.transit(self.transitions[self.current_state])
            steps -= 1

# FIX g() function, generalize to multiple emissions.
# Generalize also to multiple states.
# Merge to markov_chain.py: merge the two __init__s, put everything else in
# separate classes, one for the chain and one for the algorithms, so they are
# isolated from one another.

class HMM():
    """
    Implementation of the forward, backward, Viterbi and Baum-Welch algorithms.
    """

    def __init__(self, pi, p, q, Q, e, x):
        # Probability of being at state pi[i] at time 0.
        self.pi = numpy.array(list(pi))
        # Emission probabilities.
        self.e = e
        # Non-transition probabilities.
        self.p, self.q = p, q
        # Transition matrix.
        self.Q = numpy.array(
                [
                    [self.p, 1-self.p],
                    [1-self.q, self.q]
                    ]
                )
        # Series of observations.
        self.x = list(x)
        # Likelihood vector to be filled by forward().
        self.L = []

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
        return numpy.array(em)

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
        alphas = numpy.zeros((max_k, 2))

        alpha = self.pi * self.g(0)
        alphas[0] = self.normalize_alpha(alpha)

        for k in xrange(1, max_k):
            alpha = numpy.dot(alpha, self.Q) * self.g(k)
            alphas[k] = self.normalize_alpha(alpha)

        return alphas, sum(numpy.log(self.L))
            
    def backward(self, max_k=len(self.x)):
        """
        Complementary algorith used by viterbi() and baum_welsch().
        Create a betas matrix that stores.
        """
        if not self.L:
            self.forward(max_k) # Generate self.L

        betas = numpy.zeros((max_k, 2))
        max_k -= 1

        beta = numpy.ones((1, 2))
        betas[max_k] = beta
        
        Qt = numpy.transpose(self.Q)
        for k in xrange(max_k-1, -1, -1):
            # Backward algorithm definition:
            # beta*k = (lk/lk+1) x Q . beta*k+1 x gk+1
            beta = numpy.dot(beta, Qt) * self.g(k) / self.L[k] 
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
        
        phis = numpy.zeros((2, 2))
        es = numpy.zeros(2)
        k=56
        print alphas
        print self.Q
        print self.g(k+1)
        print betas
        """
        for k in xrange(0, max_k-1):
            phi_k = numpy.dot(alphas[k], self.Q) * self.g(k+1) * betas[k+1]
            phis += phi_k

            e = numpy.dot(phi_k, self.g(k))
            es += e
        """
        #self.Q = phis / sum(phis)
        #self.e = es / max_k

        #print self.Q
        #print self.e


if __name__ == '__main__':

    import sys

    # Create the Markov chain with the (optional) initial state probabilities.
    chain = Chain((1,0))

    # Add the (not so) Hidden States.
    chain.add_state((1, 0), (0.9, 0.1)) # State 0
    chain.add_state((0, 1), (0.1, 0.9)) # State 1
    #chain.add_state((0, 0, 1), (0.3, 0.1, 0.6)) # State 2
    
    if len(sys.argv) > 1:
        chain.run(int(sys.argv[1]))
    else:
        chain.run()
if __name__ == '__main__':
    observations = [int(l) for l in open('simulated_markov_exercise.txt')][:1000]
    fb = HMM(
            pi=(0.5, 0.5),
            p=0.5,
            q=0.8,
            Q=None,
            e=(0.1,0.9),
            x=observations)


    print fb.backward(len(observations))
