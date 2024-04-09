from jajapy import Set
from math import log

def log_likelihood(dfg, sequences: Set):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()

    num_sequences = len(sequences_sorted)

    outcounts = dict()
    for edge in dfg[0]:
        if edge[0] in outcounts:
            outcounts[edge[0]] += dfg[0][edge]
        else:
            outcounts[edge[0]] = dfg[0][edge]

    likelihood = 0.0
    # calculate likelihood of each trace
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]

            prob = 1.0

            for i in range(len(sequence)-1):
                prob *= (dfg[0][(sequence[i],sequence[i+1])] / outcounts[sequence[i]])

            if prob > 0:
                    likelihood += log(prob) * times
            else:
                print('impossible!!')

    return likelihood / sum(sequences.times)

def log_likelihood_filtered(dfg, sequences: Set, epsilon):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()

    num_sequences = len(sequences_sorted)

    outcounts = dict()
    for edge in dfg[0]:
        if edge[0] in outcounts:
            outcounts[edge[0]] += dfg[0][edge]
        else:
            outcounts[edge[0]] = dfg[0][edge]

    likelihood = 0.0
    # calculate likelihood of each trace
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]

            prob = 1.0

            for i in range(len(sequence)-1):
                prob *= (dfg[0][(sequence[i],sequence[i+1])] / outcounts[sequence[i]])

            if prob > epsilon:
                    likelihood += log(prob) * times
            else:
                print('impossible!!')

    return likelihood / sum(sequences.times)
