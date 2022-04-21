# test plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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