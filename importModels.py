import jajapy

models = dict()

for i in range(1,18):
    models[i] = jajapy.loadHMM('models/model' + str(i))
for i in [32,64,128,256]:
    models[i] = jajapy.loadHMM('models/model' + str(i))
