import jajapy, pm4py, random, pandas
import sys
from sys import argv
baum_welch = jajapy.BW()

if (argv[1] == 'help' or argv[1] == '-h' or argv[1] == '--help'):
    print(len(argv))
    print([a for a in argv])
    print('usage:')
    print('python testStates.py [event log] [output name] [optional argument if dataset uses BPI Challenge 2020 format]')
    sys.exit()

logs = pm4py.read_xes(argv[1])

if (len(argv) > 3):
    traces = [list(t[1]['concept:name']) for t in logs.groupby('case:id')]
    states = logs['concept:name'].unique()
else:
    traces = [list(t[1]['lifecycle:transition']) for t in logs.groupby('case:concept:name')]
    states = logs['lifecycle:transition'].unique()

random.shuffle(traces)

training_set = traces[:int((2*len(traces))/3)]
test_set = jajapy.Set([t for t in traces[int((2*len(traces))/3):]])


models = dict()
modelresults = dict()
results = dict()

for i in range(1,int(len(states)*1.5)+1):
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

for i in [32,64,128,256]:
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

for i in models:
    print(i, "states")
    modelresults[i] = baum_welch.fit(training_set, models[i], epsilon=0, max_it=100, return_data=True)[1]
    ll = models[i].logLikelihood(test_set)
    print("logLikelihood", ll)
    results[i] = ll

pandas.DataFrame.from_dict(results, orient='index').to_csv(argv[2] + 'LL.csv', sep=',')
pandas.DataFrame.from_dict(modelresults, orient='index').to_csv(argv[2] + 'TrainingStats.csv')
