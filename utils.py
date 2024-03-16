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
