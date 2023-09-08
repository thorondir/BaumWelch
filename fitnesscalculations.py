# various functions for calculating fitness

import jajapy
from math import log
from jajapy import Set

# gets the log "likelihood" calculated from set occurences
# sum(log(p_i/total occurences))
def getSetLoglikelihood(sequences: Set):
    logLikelihood = 0.0
    totalTimes = sum(sequences.times)
    for seq in range(len(sequences.sequences)):
        logLikelihood += log(sequences.times[seq]/totalTimes)
    return logLikelihood

# calculates the average likelihood in a test set from the forward algorithm
def likelihood(hmm, sequences: Set):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()
    likelihood = 0.0
    alpha_matrix = hmm._initAlphaMatrix(len(sequences_sorted[0]))
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]
            common = 0
            if seq > 0:
                    while common < min(len(sequences_sorted[seq-1]),len(sequence)):
                            if sequences_sorted[seq-1][common] != sequence[common]:
                                    break
                            common += 1

            alpha_matrix = hmm._updateAlphaMatrix(sequence,common,alpha_matrix)
            if alpha_matrix[-1].sum() > 0:
                    likelihood += alpha_matrix[-1].sum() * times

    return likelihood / sum(sequences.times)


# calculates the KL-divergence between the set likelihood (occurrences) and the
# likelihood calculated from the forward algorithm
def diffLikelihood(hmm, sequences: Set):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()
    totalTimes = sum(sequences.times)
    difflikelihood = 0.0
    alpha_matrix = hmm._initAlphaMatrix(len(sequences_sorted[0]))
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]
            common = 0
            if seq > 0:
                    while common < min(len(sequences_sorted[seq-1]),len(sequence)):
                            if sequences_sorted[seq-1][common] != sequence[common]:
                                    break
                            common += 1

            alpha_matrix = hmm._updateAlphaMatrix(sequence,common,alpha_matrix)

            if alpha_matrix[-1].sum() > 0:
                    P = times/totalTimes
                    difflikelihood += P * log(P / alpha_matrix[-1].sum())

    return difflikelihood

# calculates the jensen shannon divergence between set likelihood and calculated
# likelihood
def jsd(hmm, sequences: Set):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()
    totalTimes = sum(sequences.times)
    JSD = 0.0
    D_P = 0.0
    D_Q = 0.0
    alpha_matrix = hmm._initAlphaMatrix(len(sequences_sorted[0]))
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]
            common = 0
            if seq > 0:
                    while common < min(len(sequences_sorted[seq-1]),len(sequence)):
                            if sequences_sorted[seq-1][common] != sequence[common]:
                                    break
                            common += 1

            alpha_matrix = hmm._updateAlphaMatrix(sequence,common,alpha_matrix)

            P = times/totalTimes
            Q = alpha_matrix[-1].sum()
            M = 0.5*(P+Q)

            D_P +=  P*(log(P/M))
            D_Q +=  Q*(log(Q/M))

    print('Distance(P): ', D_P)
    print('Distance(Q): ', D_Q)
    return 0.5 * (D_P + D_Q)

# calculates the KL-divergence between the set likelihood and the mixture of set
# likelihood and calculated likelihood 
def P_Mdivergence(hmm, sequences: Set):
    sequences_sorted = sequences.sequences[:]
    sequences_sorted.sort()
    totalTimes = sum(sequences.times)
    D_P_M = 0.0
    alpha_matrix = hmm._initAlphaMatrix(len(sequences_sorted[0]))
    for seq in range(len(sequences_sorted)):
            sequence = sequences_sorted[seq]
            times = sequences.times[sequences.sequences.index(sequence)]
            common = 0
            if seq > 0:
                    while common < min(len(sequences_sorted[seq-1]),len(sequence)):
                            if sequences_sorted[seq-1][common] != sequence[common]:
                                    break
                            common += 1

            alpha_matrix = hmm._updateAlphaMatrix(sequence,common,alpha_matrix)

            P = times/totalTimes
            Q = alpha_matrix[-1].sum()
            M = 0.5*(P+Q)

            D_P_M +=  P*(log(P/M))

    return D_P_M


"""
def difflikelihood_multiproc(hmm, sequences: Set) -> float:
    p = Pool(processes = cpu_count()-1)
    tasks = []
    for seq,times in zip(sequences.sequences,sequences.times):
        tasks.append(p.apply_async(self._computeAlphas, [seq, times,]))
    temp = [res.get() for res in tasks if res.get() != False]
    s = sum(temp)
    if s == 0.0:
        print('WARNING: the model is not able to generate any sequence in the set')
        return 0.0
    return s/sum(sequences.times)
"""

if __name__ == '__main__':
    import csv

    traces = []
    model = jajapy.loadHMM('models/128.model')

    with open('randtraces.csv', 'r') as f:
        traces = list(csv.reader(f))

    for trace in traces:
        trace.insert(0, 'start')
        trace.append('end')

    s = Set(traces[2*int(len(traces)/3):])
