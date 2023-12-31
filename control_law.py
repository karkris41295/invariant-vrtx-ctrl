# Code which plots the control law given single invariant data generated by fix_act_mv_inv.py
# Import statements 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.size'] = '25'
#%% Loading the results

filename = 'vrtxA2.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]
vrtx_trajs, vrtx_cumins, vrtx_hamils, vrtx_ins, vrtx_hamils2, t, href, aref = data
gridsize = 10

#%% Vortex trajectories

vrtx_error_final = vrtx_hamils[:,-1] - href
vrtx_ang_error_final = vrtx_hamils2[:,-1] - aref
vrtx_error_final = vrtx_error_final.reshape(gridsize,gridsize)
vrtx_ang_error_final = vrtx_ang_error_final.reshape(gridsize,gridsize)
vrtx_cumins = vrtx_cumins.reshape(gridsize,gridsize)

#%% Control law


index_1, index_2 = 5,5

plt.plot(t, vrtx_ins[index_1*gridsize + index_2,:, 0],  '-', c = 'red')
plt.xlim(0,100)
#%% Loading the results

filename = 'vrtxAm2.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]
vrtx_trajs, vrtx_cumins, vrtx_hamils, vrtx_ins, vrtx_hamils2, t, href, aref = data
gridsize = 10

#%% Vortex trajectories

vrtx_error_final = vrtx_hamils[:,-1] - href
vrtx_ang_error_final = vrtx_hamils2[:,-1] - aref
vrtx_error_final = vrtx_error_final.reshape(gridsize,gridsize)
vrtx_ang_error_final = vrtx_ang_error_final.reshape(gridsize,gridsize)
vrtx_cumins = vrtx_cumins.reshape(gridsize,gridsize)

#%% Control law


index_1, index_2 = 5,5

plt.plot(t, vrtx_ins[index_1*gridsize + index_2,:, 0], '-', c = 'blue')
plt.xlim(0,100)


plt.xticks([0 , 100])
plt.yticks([-1,0,1])


#%%
plt.figure()
act_locx, act_locy = np.meshgrid(np.linspace(0.1,3,gridsize),np.linspace(0.1,3,gridsize))
act_sum = (act_locx**2 + act_locy**2)**.5

plt.scatter(act_sum, vrtx_cumins, c='blue')

filename = 'vrtxA2.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]
vrtx_trajs, vrtx_cumins, vrtx_hamils, vrtx_ins, vrtx_hamils2, t, href, aref = data
gridsize = 10

act_locx, act_locy = np.meshgrid(np.linspace(0.1,3,gridsize),np.linspace(0.1,3,gridsize))
act_sum = (act_locx**2 + act_locy**2)**.5

plt.scatter(act_sum, vrtx_cumins, c='red')

plt.xticks([0 , 4])
