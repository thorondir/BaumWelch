# imports all models from a directory where models are files named
# [no. of states].model

import jajapy, os

models = dict()

dir = os.fsencode('results_oneset/sepsismodels/')

for f in os.listdir(dir):
    fn = os.fsdecode(f)
    if fn.endswith('.model'):
        models[int(fn.removesuffix('.model'))] = jajapy.loadHMM(os.fsdecode(dir) + '/' + fn)
