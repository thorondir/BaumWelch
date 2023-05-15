import jajapy, pm4py, random, pandas

from importTraces import *
from importModels import *

training_set = traces[:int((2*len(traces))/3)]
test_set = jajapy.Set([t for t in traces[int((2*len(traces))/3):]])

results = dict()

for i in models:
    results[i] = models[i].logLikelihood(test_set)
