#!/usr/bin/env python
#-*- coding:utf-8 -*-

#FIX: Only works with 2 states! Generalize to N states!
# Implement transition matrix.
from random import random

class State:
    """
    Create a hidden state with an emission probability 'ep' and a transition
    probability 'tp' to state 'S' '{S1: tp1, ..., Sn: tpn}.' The self.active
    attribute indicates wich state is the Markov Chain currenlty in.
    """

    def __init__(self, name, emission_probability, transition_probability):
        """
        type(ep) and type(tp) must be float.
        """
        
        self.name = name
        self.ep = float(emission_probability)
        self.tp = float(transition_probability)
        self.is_active = False

    def emit(self):
        if random() < self.ep:
            return 1
        else:
            return 0

    def transit(self, state_list):
        if random() < self.tp:
            self.is_active = False
            for state in state_list:
                state.is_active = True

    def update(self, state_list):
        if self.is_active:
            print self.emit()
            self.transit(state_list)


if __name__ == '__main__':

    # Create the (not so) Hidden States.
    s1 = State('s1', 0.9, 0.2)
    s2 = State('s2', 0.2, 0.1)

    state_list = [s1, s2]

    # Initalize the Markov Chain.
    s1.is_active = True

    while True:
        for state in state_list:
            state.update(state_list)
