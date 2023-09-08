# Same as testStates.py, but this version deserialises existing model files to
# continue an aborted training session
#
# Use testStates.py instead to generate nicer, in-order output

import jajapy, pm4py, random, pandas, numpy
import sys, json, os
from sys import argv

from fitnesscalculations import diffLikelihood, likelihood


baum_welch = jajapy.BW()

if (len(argv) <= 1 or argv[1] == 'help' or argv[1] == '-h' or argv[1] == '--help'):
    print(len(argv))
    print([a for a in argv])
    print('usage:')
    print('python testStates.py [event log] [output name] [optional argument if dataset uses BPI Challenge 2020 format]')
    sys.exit()

logs = pm4py.read_xes(argv[1])

match argv[3]:
    case "bpic2013":
        states = logs['lifecycle:transition'].unique()
    case "bpic2020":
        states = logs['concept:name'].unique()
    case "roadfines" | "sepsis" | "bpic2012" | "bpic2020-rf":
        states = logs['concept:name'].unique()

states = numpy.append(states, ['start', 'end'])

with open(argv[2] + 'Traces.json', 'r') as f:
    traces = json.load(f)

training_set = traces

with open(argv[2] + 'TestSet.json', 'r') as f:
    test_set = jajapy.Set(json.load(f))


models = dict()
modelresults = dict()
resultsll = dict()
resultslh = dict()
resultsdl = dict()

for i in range(1,int(len(states)*1.5)+1):
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)
    modelresults[i] = 0

for i in [32,64,128,256]:
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)
    modelresults[i] = 0

os.makedirs(argv[2] + 'models', exist_ok=True)

dir = os.fsencode(argv[2]+'models')

for f in os.listdir(dir):
    fn = os.fsdecode(f)
    if fn.endswith('.model'):
        models[int(fn.removesuffix('.model'))] = jajapy.loadHMM(os.fsdecode(dir) + '/' + fn)
        modelresults[int(fn.removesuffix('.model'))] = 'skip'

for i in models:
    print(i, "states")
    if modelresults[i] != 'skip':
        modelresults[i] = baum_welch.fit(training_set, models[i], epsilon=0, max_it=100, return_data=True)[1]
    ll = models[i].logLikelihood(test_set)
    lh = likelihood(models[i], test_set)
    dl = diffLikelihood(models[i], test_set)

    print("logLikelihood", ll)
    print("likelihood", lh)
    print("diffLikelihood", dl)
    resultsll[i] = ll
    resultslh[i] = lh
    resultsdl[i] = dl

    models[i].save(argv[2]+'models/'+str(i)+'.model')

pandas.DataFrame.from_dict(resultsll, orient='index').to_csv(argv[2] + 'LL.csv', sep=',')
pandas.DataFrame.from_dict(resultslh, orient='index').to_csv(argv[2] + 'LH.csv', sep=',')
pandas.DataFrame.from_dict(resultsdl, orient='index').to_csv(argv[2] + 'DL.csv', sep=',')
pandas.DataFrame.from_dict(modelresults, orient='index').to_csv(argv[2] + 'TrainingStats.csv')
