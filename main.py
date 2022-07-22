from pm4py.objects.log.importer.xes import importer as xes_importer
import discovery, sys

""""
Use this template for your discovery algorithm
"""

# read event log
variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
log = xes_importer.apply(sys.argv[1], variant=variant, parameters=parameters)
#discovery.print_log(log)

# discover a model in the dot format
dfa = discovery.dfa_discovery(log)


# visualize the model
print("Visualization...")
dfa.viz()