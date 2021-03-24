import numpy as np
import matplotlib.pyplot as mpy
import random

# total number of bandit problems
banditProblems=2000
# total number of arms in each bandit problem
k=4
# total number of times to pull each arm
armPulls=1000

# true means generated for each arms for all the bandits
trueMeans=np.random.normal(0,1,(banditProblems,k))
# storing the true optimal arms in each bandit
trueOptimal=np.argmax(trueMeans,1)
# each row represents a bandit problem

# array of values for epsilon
epsilon=[0,0.1]
col=['r','g']

# adding subplots to plot and compare both plots simultaneously
plotFirst=mpy.figure().add_subplot(111)
plotSecond=mpy.figure().add_subplot(111)

for x in range(len(epsilon)) :

	print('The present epsilon value is : ',x)

	# Storing the predicted reward
	Q=np.zeros((banditProblems,k))
	# total number of times each arms is pulled
	N=np.ones((banditProblems,k))
	# assigning initial random arm pulls
	initialArm=np.random.normal(trueMeans,1)

	rewardEps=[]
	rewardEps.append(0)
	rewardEps.append(np.mean(initialArm))
	rewardEpsOptimal=[]

	for y in range(2,armPulls+1) :
		rewardPull=[] # all rewards in this pull/time-step
		optimalPull=0 # number of pulss of best arm in this time step
		for z in range(banditProblems) :

			if random.random()<epsilon[x] :
				i=np.random.randint(k)
			else :
				i=np.argmax(Q[z])

			if i==trueOptimal[z] : # To calculate % optimal action
				optimalPull=optimalPull+1

			rewardTemp=np.random.normal(trueMeans[z][i],1)
			rewardPull.append(rewardTemp)
			N[z][i]=N[z][i]+1
			Q[z][i]=Q[z][i]+(rewardTemp-Q[z][i])/N[z][i]

		rewardAvgPull=np.mean(rewardPull)
		rewardEps.append(rewardAvgPull)
		rewardEpsOptimal.append(float(optimalPull)*100/2000)
	plotFirst.plot(range(0,armPulls+1),rewardEps,col[x])
	plotSecond.plot(range(2,armPulls+1),rewardEpsOptimal,col[x])

mpy.rc('text',usetex=True)
#mpy.ylim(0.5,1.5)
plotFirst.title.set_text(r'$\epsilon$-greedy : Average Reward Vs Steps for 10 arms')
plotFirst.set_ylabel('Average Reward')
plotFirst.set_xlabel('Steps')
plotFirst.legend((r"$\epsilon=$"+str(epsilon[0]),r"$\epsilon=$"+str(epsilon[1])),loc='best')
plotSecond.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
plotSecond.set_ylabel(r'$\%$ Optimal Action')
plotSecond.set_xlabel('Steps')
plotSecond.set_ylim(0,100)
plotSecond.legend((r"$\epsilon=$"+str(epsilon[0]),r"$\epsilon=$"+str(epsilon[1])),loc='best')
mpy.show()