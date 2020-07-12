import pandas as pd
import cProfile, pstats, io
import os

from BertPreprocess import *
from ModelProd import *

data_model = BertPreprocess()
data_model.load_data()

pred_model = ModelProd()
pred_model.load_pretrained()

tmp = data_model.train.sample(100)

pr = cProfile.Profile()
pr.enable()

#### script that will be analized ###########
#### predicting random 100 examples from trainset

res = []
for row_id, row in tmp.iterrows():
    i = pred_model.predict('\n'.join(row.text),
                           row['accepted_function'],
                           row['rejected_function'],
                           row['accepted_product'],
                           row['rejected_product'])
    print(i)
    res.append(i)

#############################################

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

# dumping results
filename = '../data/output.pstats'
pr.dump_stats(filename)

## getting dataframe from results
data=[]
started=False

for l in s.getvalue().split("\n"):
    if not started:
        if l=="   ncalls  tottime  percall  cumtime  percall filename:lineno(function)":
            started=True
            data.append(l)
    else:
        data.append(l)
content=[]
for l in data:
    fs = l.find(" ",8)
    content.append(tuple([l[0:fs] , l[fs:fs+9], l[fs+9:fs+18], l[fs+18:fs+27], l[fs+27:fs+36], l[fs+36:]]))

prof_df = pd.DataFrame(content[1:], columns=['ncalls', 'tottime', 'percall', 'cumtime', 'percall2', 'filename:lineno(function)'])

os.system('gprof2dot -n 10  -f pstats data/output.pstats | dot -Tpng -o Screenshots/output.png')

prof_df.sort_values('percall', ascending = False, inplace = True)

prof_df.to_csv('data/profiling_module.csv', index = False)
