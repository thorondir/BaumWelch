import jajapy, pm4py, random, pandas
from importTraces import *

baum_welch = jajapy.BW()

training_set = traces[:10]
test_set = jajapy.Set([t for t in traces[int((2*len(traces))/3):]])

states = jajapy.Set(traces).getAlphabet()

results = dict()
resultsSameData = dict()
models = dict()
modelresults = dict()

for i in range(1,129):
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)


for i in reversed(models):
    print(i, "states")
    modelresults[i] = baum_welch.fit(training_set, models[i], epsilon=0.00000000000001, max_it=100, return_data=True)[1]

    results[i] = models[i].logLikelihood(test_set)
    resultsSameData[i] = models[i].logLikelihood(jajapy.Set(training_set))
    print("loglikelihood", results[i])
    print("loglikelihood samedata", resultsSameData[i])
