from mimetypes import init
from models import DFA
import jajapy

def dfa_discovery(modelname, epsilon):
    model = jajapy.loadHMM(modelname)
    states = set(range(1,model.nb_states+1))
    transitions = set()
    for i in range(model.nb_states):
        for j in range(model.nb_states):
            if model.matrix[i][j] > epsilon:
                transitions.add((str(model.matrix[i][j]),i+1,j+1))

    min_outgoing = sum(model.matrix[0]) - model.matrix[0][0]
    outgoings=[min_outgoing]
    final = 0
    for i in range(1,model.nb_states):
        outgoing = sum(model.matrix[i]) - model.matrix[i][i]
        outgoings.append(outgoing)
        if outgoing < min_outgoing:
            min_outgoing = outgoing
            final = i

    print(final,outgoings)

    dfa = DFA(modelname, states, 1, set([final+1]), transitions)
    return dfa

def print_log(log):
    for trace in log:
        for event in trace:
           print(event['concept:name'])
