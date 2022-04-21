# test plotting
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from usc_learning.utils.utils import plot_results
from stable_baselines.bench.monitor import load_results

interm_dir = "./logs/intermediate_models/"
log_dirs = ['102620220214', '102520122637', '102620170125']

for i,lgdir in enumerate(log_dirs):
	log_dirs[i] = interm_dir + lgdir
	print(lgdir)
print(log_dirs)

# iterate through to get min time
max_len = np.inf
for folder in log_dirs:
	timesteps = load_results(folder)
	# np.arange(len(timesteps))
	max_len = np.minimum( len(timesteps),max_len )
	print(max_len)

# now actually concatenate data
max_len = int(max_len)
data = []
for folder in log_dirs:
	timesteps = load_results(folder)
	t = timesteps.index.values[:max_len]
	data.append( timesteps.r.values[:max_len])

fig = plt.figure()
sns.tsplot( time=t ,   data=data, color='r',  linestyle='-')
plt.title('RL Runs')
plt.legend()
plt.show()
sys.exit()


def getdata():
	basecond = [[ 18 , 20 , 19 , 18 , 13 , 4 , 1 ],
				[ 20 , 17 , 12 , 9 , 3 , 0 , 0 ],
				[ 20 , 20 , 20 , 12 , 5 , 3 , 0 ] ]
	cond1  =  [ [ 18 , 19 , 18 , 19 , 20 , 15 , 14 ],
				[ 19 , 20 , 18 , 16 , 20 , 15 , 9 ],
				[ 19 , 20 , 20 , 20 , 17 , 10 , 0 ] ,
				[ 20 , 20 , 20 , 20 , 7 , 9 , 1 ] ]
	cond2=  [ 	[ 20 , 20 , 20 , 20 , 19 , 17 , 4 ],
				[ 20 , 20 , 20 , 20 , 20 , 19 , 7 ] ,
				[ 19 , 20 , 20 , 19 , 19 , 15 , 2 ] ]
	cond3  =  [ [ 20 , 20 , 20 , 20 , 19 , 17 , 12 ] ,
				[ 18 , 20 , 19 , 18 , 13 , 4 , 1 ] ,
				[ 20 , 19 , 18 , 17 , 13 , 2 , 0 ] ,
				[ 19 , 18 , 20 , 20 , 15 , 6 , 0 ] ]
	return basecond, cond1, cond2, cond3

results = getdata()
fig = plt.figure()
# plot iterations 0 .. 6
xdata = np.array( [ 0 , 1 , 2 , 3 , 4 , 5 , 6 ] ) / 5 

# plot each line
#  ( may  want   t o   automate   t h i s   p a r t   e . g .   w i t h   a   l o o p ) .
sns.tsplot( time=xdata ,   data=results[0], color='r',  linestyle='-')
sns.tsplot( time=xdata ,   data=results[1], color='g',  linestyle='--')
sns.tsplot( time=xdata ,   data=results[2], color='b',  linestyle=':')
sns.tsplot( time=xdata ,   data=results[3], color='k',  linestyle='-.')
plt.title('RL Runs')
plt.legend()
plt.show()