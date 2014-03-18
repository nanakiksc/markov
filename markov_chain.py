#!/usr/bin/env python
#-*- coding:utf-8 -*-

#FIX: Only works with 2 states! Generalize to N states!
# Implement transition matrix.
from random import random

class MarkovChain:
    """
    Create a Markov Chain that stores hidden states with different emission
    probabilities and transition probabilities.
    """

    def __init__(self):
        """
        """
        
        self.emission_probs = []
        self.transition_matrix = []
        self.num_states = 0
        self.current_state = None

    def add_state(self, emission_probability, transition_probability):
        """
        Each state is added as a list or tuple of length 2. The first element is
        the emission probability of the state. The second element is the row of
        the transition matrix corresponding to the state.
        """

        ep = float(emission_probability)
        self.emission_probs.append(ep)

        tp = [float(p) for p in transition_probability]
        self.transition_matrix.append(tp)
        
        self.num_states += 1

    def check_transition_matrix(self):
        """
        Check whether the size and values of the transition matrix make sense.
        """

        for i, state in enumerate(self.transition_matrix):
            assert sum(state) == 1
            assert len(state) == self.num_states
        assert i + 1 == self.num_states

    def emit(self):
        """
        Make an emission based on the emission probability of the current state.
        """

        if random() < self.emission_probs[self.current_state]:
            return 1
        else:
            return 0

    def transit(self):
        """
        Update current state by choosing a state 'st', with probability 'pr',
        among all possible states, including the current state itself.
        """
        
        r = random()
        acc = 0
        for st, pr in enumerate(self.transition_matrix[self.current_state]):
            if acc <= r < pr:
                self.current_state = st
            acc += pr

    def run(self):
        """
        Check that everything makes sense, initialize the chain and run it.
        """
        
        chain.check_transition_matrix()

        # Initalize current state of the Markov Chain.
        chain.current_state = int(random() * self.num_states)

        while True:
            print self.emit()
            self.transit()


if __name__ == '__main__':
    
    chain = MarkovChain()

    # Add the (not so) Hidden States.
    chain.add_state(0.8, (0.9, 0.1)) # State 0
    chain.add_state(0.05, (0.15, 0.85)) # State 1
    
    chain.run()
