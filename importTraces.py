# imports a list of traces from a json file

import json

traces = []

with open('results_oneset/sepsisTestSet.json', 'r') as f:
    traces = json.load(f)
