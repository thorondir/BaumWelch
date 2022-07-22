from mimetypes import init
from os import statvfs_result
from turtle import shape
from typing import final
import graphviz

class DFA:
  def __init__(self, name, states, init_state, final_states, transitions):
    self.name = name
    self.states = states
    self.init_state = init_state
    self.final_states = final_states
    self.transitions = transitions

  def viz(self):
    dot = graphviz.Digraph(self.name)
    c,s = self.check()
    if not c:   
        print(s)
        return
    for f_state in self.final_states: 
        dot.node(str(f_state), shape='doublecircle')
    for state in self.states.difference(self.final_states): 
        dot.node(str(state), shape='circle')
    dot.node("", shape='point')
    dot.edge("", str(self.init_state))
    for t in self.transitions:
        dot.edge(str(t[1]), str(t[2]), label=str(t[0]))
    dot.view()  
  
  def check(self):
      if not(self.init_state in self.states):
          return False, "Initial state is not in states"
      if not(self.final_states.issubset(self.states)):
          return False, "Final states are not in states"
      for transition in self.transitions:
          if not(transition[1] in self.states) or not(transition[2] in self.states):
              return False, "Transition " + transition + " doesn't connect states"
      return True, ""