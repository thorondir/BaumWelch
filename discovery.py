from mimetypes import init
from models import DFA

def dfa_discovery(log):
    dfa = DFA("Test", {1,2,3}, 1, {1,3}, {("a",1,2), ("b",2,2), ("c",1,3)})
    return dfa

def print_log(log):
    for trace in log:
        for event in trace:
           print(event['concept:name'])

    