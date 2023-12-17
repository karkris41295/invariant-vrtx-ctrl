import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

recall = np.load('vrtxdata5.npz')
data = [recall[key] for key in recall]
R0, act, n, ctrl, hamil, href, kappa0, sol, sol_complex, sol, t, t_star, tmax, x, y = data

xp, yp = np.meshgrid(np.arange(-5, 5., 1),np.arange(-5, 5., 1))
dt = .25 #samples
#ctrl = 0*ctrl
sol = [[xp,yp]]

def single_step(x0, y0, idx):
    d_dt = [xp*0, yp*0]
    
    for i in range(n-act):
        #i = j%2
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + .00000001
        d_dt[0] += -(kappa0[i]/(2*np.pi))*(y0 - y[idx, i])/den
        d_dt[1] += (kappa0[i]/(2*np.pi))*(x0 - x[idx, i])/den
    '''  
    for i in range(n-act,n):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + .00000001
        d_dt[0] += -(ctrl[idx]/(2*np.pi))*(y0 - y[idx, i])/den
        d_dt[1] += (ctrl[idx]/(2*np.pi))*(x0 - x[idx, i])/den
    '''   
    return x0 + dt*d_dt[0], y0 + dt*d_dt[1]
        
for i in range(16000):
    xp, yp = single_step(xp, yp, i)
    sol += [[xp,yp]]
    
#%%    
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', xlim=(-10, 10), ylim=(-10, 10), xlabel = '$x/R_0$', ylabel = '$y/R_0$')
tracer, = plt.plot([], [], marker='.', linewidth = 0, color='red')

particle, = ax.plot([],[], marker='o', color='black', linewidth = .01)
particle2, = ax.plot([],[], marker='o', color='grey', linewidth = .01)

#time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#hamil_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
#k_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
#colors = plt.cm.bwr(np.sign(kappa0[:-act]))
colors = plt.cm.brg(np.linspace(0, 1, n))
tr = []

for k in range(n-act):
    traj, = ax.plot([],[], color=colors[k], alpha=0.6)
    tr += traj,
    
def update2(i):
    i = i*10
    
    particle.set_data(x[i,:-act]/R0,y[i,:-act]/R0)
    particle2.set_data(x[i,-act:]/R0,y[i,-act:]/R0)
    
    tracer.set_data(sol[i][0].flatten()/R0,sol[i][1].flatten()/R0)

ani = animation.FuncAnimation(fig, update2, interval=1)