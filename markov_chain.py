#!/usr/bin/env python
#-*- coding:utf-8 -*-

from random import random

class MarkovChain:
    """
    Create a Markov Chain that stores hidden states with different emission
    probabilities and transition probabilities.
    """

    def __init__(self):
        """
        Each state is represented by an index in the self.emission_probs and
        self.transition_matrix lists. Thus the state 'n' contains a list of its
        emission probabilities at self.emission_probs[n] and a list of its
        transition probabilities at self.transition_matrix[n].
        """

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

    def transit(self):
        """
        Update current state by choosing a state 'st', with probability 'pr',
        among all possible states, including the current state itself.
        """
        
        r = random()
        acc = 0
        for st, pr in enumerate(self.transition_matrix[self.current_state]):
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
        self.current_state = int(random() * self.num_states)

        while True:
            print self.emit()
            self.transit()


if __name__ == '__main__':
    
    chain = MarkovChain()

    # Add the (not so) Hidden States.
    chain.add_state((0.2, 0.8), (0.9, 0.1)) # State 0
    chain.add_state((0.95, 0.05), (0.15, 0.85)) # State 1
    
    chain.run()
