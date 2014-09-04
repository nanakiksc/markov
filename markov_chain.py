#!/usr/bin/env python
#-*- coding:utf-8 -*-

from random import random

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
        self.num_states = len(self.initial_probs)
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
        ne = len(emission_probability)
        if ne > self.num_emissions:
            self.num_emissions = ne

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
