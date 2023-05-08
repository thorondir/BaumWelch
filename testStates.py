import jajapy, pm4py, random, pandas
baum_welch = jajapy.BW()

logs = pm4py.read_xes('bpi_challenge_2013_incidents.xes')
traces = [list(t[1]['lifecycle:transition']) for t in logs.groupby('case:concept:name')]
random.shuffle(traces)

states = logs['lifecycle:transition'].unique()

training_set = traces[:int((2*len(traces))/3)]
test_set = jajapy.Set([t for t in traces[int((2*len(traces))/3):]])


models = dict()
modelresults = dict()
results = dict()

for i in range(1,len(states)+5):
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

for i in [32,64,128,256]:
    models[i] = jajapy.HMM_random(i, list(states), random_initial_state=True)

for i in models:
    print(i, "states")
    if i > 128:
        modelresults[i] = baum_welch.fit(training_set, models[i], max_it=100, return_data=True)[1]
    else:
        modelresults[i] = baum_welch.fit(training_set, models[i], epsilon=0, max_it=100, return_data=True)[1]

    ll = models[i].logLikelihood(test_set)
    print("loglikelihood", ll)
    results[i] = ll

pandas.DataFrame.from_dict(results, orient='index').to_csv('loglikelihoods.csv', sep=',')
pandas.DataFrame.from_dict(modelresults, orient='index').to_csv('models/modelresults')
