import pandas as pd
import pm4py

def convertTracesTopm4py(traces):
    dfdict = {'CaseID': [], 'Activity': [], 'Timestamp': []}

    for tr in range(len(traces)):
        time = 0
        for act in traces[tr]:
            dfdict['CaseID'].append(str(tr))
            dfdict['Activity'].append(act)
            dfdict['Timestamp'].append(time)
            time += 1

    df = pd.DataFrame.from_dict(dfdict)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = pm4py.format_dataframe(df, case_id='CaseID', activity_key='Activity', timestamp_key='Timestamp')

    return df


def train_dfg_from_traces(traces):
    df = convertTracesTopm4py(traces)

    return pm4py.discover_dfg(df)


def filter_dfg(dfg, epsilon):
    outcounts = dict()
    for edge in dfg[0]:
        if edge[0] in outcounts:
            outcounts[edge[0]] += dfg[0][edge]
        else:
            outcounts[edge[0]] = dfg[0][edge]

    new_dict = dict()

    for edge in dfg[0]:
        if dfg[0][edge]/outcounts[edge[0]] >= epsilon:
            new_dict[edge] = dfg[0][edge]
        else:
            new_dict[edge] = 1

    return (new_dict, dfg[1], dfg[2])

