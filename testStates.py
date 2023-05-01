import jajapy, pm4py
baum_welch = jajapy.BW()

logs = pm4py.read_xes('bpi_challenge_2013_incidents.xes')
traces = [t[1]['lifecycle:transition'] for t in logs.groupby('case:concept:name')]

states = logs['lifecycle:transition'].unique()

training_set = traces[:int((2*len(traces))/3)]
test_set = jajapy.Set([list(t) for t in traces[int((2*len(traces))/3):]])


test_models = []


for i in range(2,len(states)+5):
    print(i, "states")
    base_model = jajapy.HMM_random(i,list(states))
    test_models.append(baum_welch.fit(training_set, base_model, epsilon=0, max_it=100))
    print("loglikelihood", test_models[-1].logLikelihood(test_set))
