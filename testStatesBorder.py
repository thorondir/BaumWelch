import jajapy, pm4py, random, pandas
baum_welch = jajapy.BW()

logs = pm4py.read_xes('bpi_challenge_2013_incidents.xes')
traces = [list(t[1]['lifecycle:transition']) for t in logs.groupby('case:concept:name')]
random.shuffle(traces)

states = list(logs['lifecycle:transition'].unique())
states.append('border')

training_traces = traces[:int((2*len(traces))/3)]
test_traces = traces[int((2*len(traces))/3):]


models = dict()
modelresults = dict()
results = dict()

for i in range(1,17):
    models[i] = jajapy.HMM_random(17, states, random_initial_state=True)

for l in range(1,17):

    training_set = []
    test_set = []

    n = int(len(training_traces)/l)

    """
    for i in range(n):
        temp = []
        temp.append('border')
        for trace in training_traces[i*l:(i+1)*l]:
            for log in trace:
                temp.append(log)
            temp.append('border')
        training_set.append(temp)
"""
    for i in range(int(len(test_traces)/l)):
        temp = []
        temp.append('border')
        for trace in test_traces[i*l:(i+1)*l]:
            for log in trace:
                temp.append(log)
            temp.append('border')
        test_set.append(temp)

    test_set = jajapy.Set(test_set)

  #  modelresults[l] = baum_welch.fit(training_set, models[l], max_it=50, return_data=True)

    print("len training trace bunch", l)
    print("loglikelihood", models[l].logLikelihood(test_set))
    results[l] = models[l].logLikelihood(test_set)

    """
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

    """


test_set = []

l = 2

for i in range(int(len(test_traces)/l)):
    temp = []
    temp.append('border')
    for trace in test_traces[i*l:(i+1)*l]:
        for log in trace:
            temp.append(log)
        temp.append('border')
    test_set.append(temp)

test_set = jajapy.Set(test_set)

for l in range(1, 17):
    
