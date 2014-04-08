#!/usr/bin/env python
#-*- coding:utf-8 -*-

from random import random

class MarkovChain:
    """
    Create a first order Markov chain that stores hidden states with different
    emission probabilities and transition probabilities.
    """

    def __init__(self, initial_probs=[]):
        """
        Each state is represented by an index in the self.emission_probs and
        self.transition_matrix lists. Thus the state 'n' contains a list of its
        emission probabilities at self.emission_probs[n] and a list of its
        transition probabilities at self.transition_matrix[n]. The probability
        of the chain starting at state 'n' (before the first transition is done)
        is given by self.initial_probs[n]. If self.initial_probs is an empty
        list (as by default), all states are equally probable.
        """

        self.initial_probs = [p for p in initial_probs]
        self.emission_probs = []
        self.transition_matrix = []
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
        self.emission_probs.append(ep)

        tp = [float(p) for p in transition_probability]
        self.transition_matrix.append(tp)

        self.num_states += 1

    def check_chain(self):
        """
        Check whether the size and values of the emission probability and
        transition matrices make sense.
        """

        if self.initial_probs:
            assert sum(self.initial_probs) == 1
            assert len(self.initial_probs) == len(self.transition_matrix)

        for i, state in enumerate(self.emission_probs):
            assert sum(state) == 1
        assert i + 1 == self.num_states

        for i, state in enumerate(self.transition_matrix):
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
        for em, pr in enumerate(self.emission_probs[self.current_state]):
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

    def run(self):
        """
        Check that everything makes sense, initialize the chain and run it.
        """

        self.check_chain()

        # Initalize current state of the Markov Chain.
        if self.initial_probs:
            self.transit(self.initial_probs)
        else:
            self.current_state = int(random() * self.num_states)

        while 1:
        #for i in xrange(10**5):
            print self.emit()
            self.transit(self.transition_matrix[self.current_state])


if __name__ == '__main__':

    # Create the Markov chain with the (optional) initial state probabilities.
    chain = MarkovChain((1,0,0))

    # Add the (not so) Hidden States.
    chain.add_state((1, 0, 0), (0, 1, 0)) # State 0
    chain.add_state((0, 1, 0), (0.5, 0.2, 0.3)) # State 1
    chain.add_state((0, 0, 1), (0.3, 0.1, 0.6)) # State 2
    
    chain.run()
