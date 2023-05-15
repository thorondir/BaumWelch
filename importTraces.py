import csv

traces = []

with open('randtraces.csv', 'r') as f:
    traces = list(csv.reader(f))
