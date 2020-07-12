import pandas as pd
import cProfile, pstats, io
import os


pr = cProfile.Profile()
pr.enable()

#### script that will be analized ###########
#### Script includes loading models and predicting random 100 examples



#############################################

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

# dumping results
filename = 'output.pstats'
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

os.system('gprof2dot -n 2  -f pstats output.pstats | dot -Tpng -o output.png')

prof_df = prof_df.sort_values('percall', ascending = False, inplace = True)

prof_df.to_csv('profiling_module.csv', index = False)