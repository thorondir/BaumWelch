import jajapy, pm4py, random, pandas, numpy
import sys, json, os
from sys import argv

from fitnesscalculations import *


baum_welch = jajapy.BW()

logs = pm4py.read_xes('logs/bpi_challenge_2013_incidents.xes')

traces = [list(t[1]['lifecycle:transition']) for t in logs.groupby('case:concept:name')]
states = logs['lifecycle:transition'].unique()

states = numpy.append(states, ['start', 'end'])

for trace in traces:
    trace.insert(0, 'start')
    trace.append('end')

random.shuffle(traces)

with open(argv[2] + 'Traces.json', 'w') as f:
    json.dump(traces, f)

training_set = traces

random.shuffle(traces)

test_set = jajapy.Set(traces[:int((2*len(traces))/3)])

with open(argv[2] + 'TestSet.json', 'w') as f:
    json.dump(traces[:int((2*len(traces))/3)], f)


models = dict()
modelresults = dict()
resultsll = dict()
resultslh = dict()
resultsdl = dict()

for i in range(1,int(len(states)*1.5)+1):
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

for i in [32,64,128,256]:
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

os.makedirs(argv[2] + 'models', exist_ok=True)
for i in models:
    print(i, "states")
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

    models[i].save(argv[2]+'models/'+str(i)+'.txt')

pandas.DataFrame.from_dict(resultsll, orient='index').to_csv(argv[2] + 'LL.csv', sep=',')
pandas.DataFrame.from_dict(resultslh, orient='index').to_csv(argv[2] + 'LH.csv', sep=',')
pandas.DataFrame.from_dict(resultsdl, orient='index').to_csv(argv[2] + 'DL.csv', sep=',')
pandas.DataFrame.from_dict(modelresults, orient='index').to_csv(argv[2] + 'TrainingStats.csv')
