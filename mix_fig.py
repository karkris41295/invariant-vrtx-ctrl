# Plot the KL-divergence vs energy spent by actuator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#%% Loading the results

filename = 'chamilfe3600_9000.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]
t, vrtx_trajs, vrtx_cumins, vrtx_ins1, vrtx_hamils, vrtx_angs, vrtx_exes, vrtx_whys, inv_vals, act_locx, act_locy, kappa0, rg = data

### Parameter Setup ###########################################################
n = 5 # number of vortices
act = 1 # number of fixed vortex actuators
ofst = 8400 # start paricle integration after this many samples
###############################################################################
 
offsetB = np.deg2rad(90)
offsetC = np.deg2rad(90)
r0 = 1*np.array([[1*np.cos(offsetC), 1*np.sin(offsetC)], 
           [1., 0.], 
            [-1*np.cos(offsetB), -1*np.sin(offsetB)],
            [-1, 0],
            [-2, 0],
            [0, -2],
           [0, 2],
           [2, 0]], dtype = 'float64')

samples = len(t) # no of samples/grid points 2000
tmax = max(t) #samples #final time 800

### Organizing data

def H_cal(r):
    '''
    calculate hamiltonian at instant
    '''
    hamil = 0
    for i in range(n-act): # These loops creates a basis for the Hamiltonian
        for j in range(i+1,n-act):
            hamil += kappa0[i]*kappa0[j]*(np.log(np.linalg.norm(r[i]-r[j])))
        
    hamil = -hamil/(4*np.pi)
    return hamil

def ang_imp(r):
    '''
    calculate angular impulse
    '''
    temp = np.dot(kappa0[:-act],(r[:-act, 0]**2 + r[:-act, 1]**2))
    return temp

def x_imp(r):
    '''
    calculate linear impulse in the x-direction
    '''
    
    temp = np.dot(kappa0[:-act], r[:-act, 0])
    return temp

def y_imp(r):
    '''
    calculate linear impulse in the y-direction
    '''
    
    temp = np.dot(kappa0[:-act], r[:-act, 1])
    return temp

aref = ang_imp(rg[:n])
xref = x_imp(rg[:n])
yref = y_imp(rg[:n])
href = H_cal(rg[:n])

        
href+= inv_vals

# Set number and arrangement of tracer particles here
xp, yp = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10,10, 0.1))
dt = t[1] - t[0]
#%% Function definitions
def single_step(x0, y0, idx, x, y, ctrl):
    d_dt = [xp*0, yp*0]
    
    for i in range(n-act):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[0] += -(kappa0[i]/(2*np.pi))*(y0 - y[idx, i])/den
    
    for i in range(n-act,n):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[0] += -(ctrl[idx]/(2*np.pi))*(y0 - y[idx, i])/den
        
    x0 = x0 + dt*d_dt[0]
        
    for i in range(n-act):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[1] += (kappa0[i]/(2*np.pi))*(x0 - x[idx, i])/den
        
    for i in range(n-act,n):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[1] += (ctrl[idx]/(2*np.pi))*(x0 - x[idx, i])/den
    
    
    return x0, y0 + dt*d_dt[1]

def tracer_soln(vrtx_trajs, vrtx_ins ,xp,yp,idx):
    sol = [[xp,yp]]
    
    x = vrtx_trajs.real[idx]
    y = vrtx_trajs.imag[idx]
    ctrl = vrtx_ins[idx]
    
    r0[n-act] = np.array([act_locx, act_locy])

    R0 = np.average(np.linalg.norm(r0[:n-act], axis = 1))
    
    for i in range(ofst,samples):
        xp, yp = single_step(xp, yp, i, x, y, ctrl)
        sol += [[xp,yp]]
    
    return sol,x,y,R0

#%% Plotting section


print('Final position of tracers for ' + filename)
d_imshow1 = np.zeros(len(href))
for i in range(len(href)):
    
    print(i)
    #plt.subplot(5,5,(i*5 + j) + 1)
    sol,x,y,R0 = tracer_soln(vrtx_trajs, vrtx_ins1,xp,yp,i)
    
    # Kullback-Leibler Divergence
    hist_grid_size = 10
    xedges = np.linspace(-hist_grid_size,hist_grid_size,4*hist_grid_size+1)
    yedges = np.linspace(-hist_grid_size,hist_grid_size,4*hist_grid_size+1)
    
    xd_initial = sol[0][0].flatten()
    yd_initial = sol[0][1].flatten()
    H_initial,xedges,yedges = np.histogram2d(xd_initial,yd_initial,bins = [xedges,yedges])
    
    xd_final = sol[-1][0].flatten()
    yd_final = sol[-1][1].flatten()
    H_final,xedges,yedges = np.histogram2d(xd_final,yd_final,bins = [xedges,yedges])
    
    divergence = 0
    for l in range(H_initial.shape[0]):
        for m in range(H_initial.shape[1]):
            if H_final[l,m] == 0 or H_initial[l,m]==0:
                continue
            divergence += H_final[l,m]*np.log(H_final[l,m]/H_initial[l,m])
    divergence = np.abs(np.round(divergence,2))
    
    d_imshow1[i] = divergence
# =============================================================================
#         print(index_1*gridsize + index_2)
#         print(divergence)
# =============================================================================
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal', xlim=(-10, 10), ylim=(-10,10), xlabel = '$x/R_0$', ylabel = '$y/R_0$  \n KL diver:'+str(divergence))
    #tracer, = plt.plot([], [], marker='.', linewidth = 0, color='red')
    tracer = plt.scatter([], [], marker='.', linewidth = 0)
    grid_length = len(sol[1][0].flatten())
    offset = np.zeros((grid_length,2))
    offset[:,0] = sol[0][0].flatten()/R0
    offset[:,1] = sol[0][1].flatten()/R0
    tracer.set_offsets(offset)
    color_map = plt.get_cmap('jet')
    colors = np.zeros((grid_length,4))
    # Gaussian color map
    for k in range(grid_length):
        A = 1 # Amplitude of Gaussian
        sigma = 3 # Variance
        xg = offset[k,0]
        yg = offset[k,1]
        z = A*np.exp(-(xg**2/(2*sigma**2)+yg**2/(2*sigma**2)))
        colors[k,:] = color_map(z)
    tracer.set_color(colors)
    offset[:,0] = sol[-1][0].flatten()/R0
    offset[:,1] = sol[-1][1].flatten()/R0
    tracer.set_offsets(offset)
    particle, = ax.plot([],[], marker='o', color='black', linewidth = .01)
    particle2, = ax.plot([],[], marker='o', color='grey', linewidth = .01)
    
    particle.set_data(x[-1,:-act]/R0,y[-1,:-act]/R0)
    particle2.set_data(x[-1,-act:]/R0,y[-1,-act:]/R0)
      
    colors = plt.cm.brg(np.linspace(0, 1, n))
    plt.axis('equal')

#%% Loading the results

filename = 'phamil3600_9000.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]
t, vrtx_trajs, vrtx_cumins, vrtx_ins2, vrtx_hamils, vrtx_angs, vrtx_exes, vrtx_whys, inv_vals, act_locx, act_locy, kappa0, rg = data

### Parameter Setup ###########################################################
n = 5 # number of vortices
act = 1 # number of fixed vortex actuators
ofst = 8400 # start paricle integration after this many samples
###############################################################################
 
offsetB = np.deg2rad(90)
offsetC = np.deg2rad(90)
r0 = 1*np.array([[1*np.cos(offsetC), 1*np.sin(offsetC)], 
           [1., 0.], 
            [-1*np.cos(offsetB), -1*np.sin(offsetB)],
            [-1, 0],
            [-2, 0],
            [0, -2],
           [0, 2],
           [2, 0]], dtype = 'float64')

samples = len(t) # no of samples/grid points 2000
tmax = max(t) #samples #final time 800

### Organizing data

def H_cal(r):
    '''
    calculate hamiltonian at instant
    '''
    hamil = 0
    for i in range(n-act): # These loops creates a basis for the Hamiltonian
        for j in range(i+1,n-act):
            hamil += kappa0[i]*kappa0[j]*(np.log(np.linalg.norm(r[i]-r[j])))
        
    hamil = -hamil/(4*np.pi)
    return hamil

def ang_imp(r):
    '''
    calculate angular impulse
    '''
    temp = np.dot(kappa0[:-act],(r[:-act, 0]**2 + r[:-act, 1]**2))
    return temp

def x_imp(r):
    '''
    calculate linear impulse in the x-direction
    '''
    
    temp = np.dot(kappa0[:-act], r[:-act, 0])
    return temp

def y_imp(r):
    '''
    calculate linear impulse in the y-direction
    '''
    
    temp = np.dot(kappa0[:-act], r[:-act, 1])
    return temp

aref = ang_imp(rg[:n])
xref = x_imp(rg[:n])
yref = y_imp(rg[:n])
href = H_cal(rg[:n])

        
href+= inv_vals

# Set number and arrangement of tracer particles here
xp, yp = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10,10, 0.1))
dt = t[1] - t[0]
#%% Function definitions
def single_step(x0, y0, idx, x, y, ctrl):
    d_dt = [xp*0, yp*0]
    
    for i in range(n-act):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[0] += -(kappa0[i]/(2*np.pi))*(y0 - y[idx, i])/den
    
    for i in range(n-act,n):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[0] += -(ctrl[idx]/(2*np.pi))*(y0 - y[idx, i])/den
        
    x0 = x0 + dt*d_dt[0]
        
    for i in range(n-act):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[1] += (kappa0[i]/(2*np.pi))*(x0 - x[idx, i])/den
        
    for i in range(n-act,n):
        den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-5
        d_dt[1] += (ctrl[idx]/(2*np.pi))*(x0 - x[idx, i])/den
    
    
    return x0, y0 + dt*d_dt[1]

def tracer_soln(vrtx_trajs, vrtx_ins ,xp,yp,idx):
    sol = [[xp,yp]]
    
    x = vrtx_trajs.real[idx]
    y = vrtx_trajs.imag[idx]
    ctrl = vrtx_ins[idx]
    
    r0[n-act] = np.array([act_locx, act_locy])

    R0 = np.average(np.linalg.norm(r0[:n-act], axis = 1))
    
    for i in range(ofst,samples):
        xp, yp = single_step(xp, yp, i, x, y, ctrl)
        sol += [[xp,yp]]
    
    return sol,x,y,R0

#%% Plotting section


print('Final position of tracers for ' + filename)
d_imshow2 = np.zeros(len(href))
for i in range(len(href)):
    
    print(i)
    #plt.subplot(5,5,(i*5 + j) + 1)
    sol,x,y,R0 = tracer_soln(vrtx_trajs, vrtx_ins1,xp,yp,i)
    
    # Kullback-Leibler Divergence
    hist_grid_size = 10
    xedges = np.linspace(-hist_grid_size,hist_grid_size,4*hist_grid_size+1)
    yedges = np.linspace(-hist_grid_size,hist_grid_size,4*hist_grid_size+1)
    
    xd_initial = sol[0][0].flatten()
    yd_initial = sol[0][1].flatten()
    H_initial,xedges,yedges = np.histogram2d(xd_initial,yd_initial,bins = [xedges,yedges])
    
    xd_final = sol[-1][0].flatten()
    yd_final = sol[-1][1].flatten()
    H_final,xedges,yedges = np.histogram2d(xd_final,yd_final,bins = [xedges,yedges])
    
    divergence = 0
    for l in range(H_initial.shape[0]):
        for m in range(H_initial.shape[1]):
            if H_final[l,m] == 0 or H_initial[l,m]==0:
                continue
            divergence += H_final[l,m]*np.log(H_final[l,m]/H_initial[l,m])
    divergence = np.abs(np.round(divergence,2))
    
    d_imshow2[i] = divergence
# =============================================================================
#         print(index_1*gridsize + index_2)
#         print(divergence)
# =============================================================================
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal', xlim=(-10, 10), ylim=(-10,10), xlabel = '$x/R_0$', ylabel = '$y/R_0$  \n KL diver:'+str(divergence))
    #tracer, = plt.plot([], [], marker='.', linewidth = 0, color='red')
    tracer = plt.scatter([], [], marker='.', linewidth = 0)
    grid_length = len(sol[1][0].flatten())
    offset = np.zeros((grid_length,2))
    offset[:,0] = sol[0][0].flatten()/R0
    offset[:,1] = sol[0][1].flatten()/R0
    tracer.set_offsets(offset)
    color_map = plt.get_cmap('jet')
    colors = np.zeros((grid_length,4))
    # Gaussian color map
    for k in range(grid_length):
        A = 1 # Amplitude of Gaussian
        sigma = 3 # Variance
        xg = offset[k,0]
        yg = offset[k,1]
        z = A*np.exp(-(xg**2/(2*sigma**2)+yg**2/(2*sigma**2)))
        colors[k,:] = color_map(z)
    tracer.set_color(colors)
    offset[:,0] = sol[-1][0].flatten()/R0
    offset[:,1] = sol[-1][1].flatten()/R0
    tracer.set_offsets(offset)
    particle, = ax.plot([],[], marker='o', color='black', linewidth = .01)
    particle2, = ax.plot([],[], marker='o', color='grey', linewidth = .01)
    
    particle.set_data(x[-1,:-act]/R0,y[-1,:-act]/R0)
    particle2.set_data(x[-1,-act:]/R0,y[-1,-act:]/R0)
      
    colors = plt.cm.brg(np.linspace(0, 1, n))
    plt.axis('equal')

#%% Changing KL divergence in time
from matplotlib import cm
plt.rcParams['font.size'] = '15'

plt.figure()
energy_spent1 = np.abs(vrtx_ins1[:,:,0])
energy_spent1 = np.sum(energy_spent1, axis = 1) * dt

energy_spent2 = np.abs(vrtx_ins2[:,:,0])
energy_spent2 = np.sum(energy_spent2, axis = 1) * dt

reversed_color_map1 = cm.Reds.reversed()

plt.scatter(energy_spent1[5:10:1], d_imshow1[5:10:1], c = 'black', s=110, marker = '*')
plt.scatter(energy_spent2[5:10:1], d_imshow2[5:10:1], c = 'black', s=50, marker = 'D')
plt.scatter(energy_spent1[5:10:1], d_imshow1[5:10:1], c = href[5:10:1], cmap=reversed_color_map1, s=30, marker = '*')
plt.scatter(energy_spent2[5:10:1], d_imshow2[5:10:1], c = href[5:10:1], cmap=reversed_color_map1, s=20, marker = 'D')
plt.xlim(0,250)
plt.ylim(1000,2700)

#%% Changing KL divergence in time
from matplotlib import cm
plt.rcParams['font.size'] = '15'

plt.figure()


reversed_color_map2 = cm.Blues

plt.scatter(energy_spent1[10:15:1], d_imshow1[10:15:1], c = 'black', s=110, marker = '*')
plt.scatter(energy_spent2[10:15:1], d_imshow2[10:15:1], c = 'black', s=50, marker = 'D')
plt.scatter(energy_spent1[10:15:1], d_imshow1[10:15:1], c = href[10:15:1], cmap=reversed_color_map2, s=30, marker = '*')
plt.scatter(energy_spent2[10:15:1], d_imshow2[10:15:1], c = href[10:15:1], cmap=reversed_color_map2, s=20, marker = 'D')
plt.xlim(0,250)
plt.ylim(1000,2700)

#%%

import matplotlib.colors as mcolors

data = np.diag(inv_vals[5:15:1])
data[data == 0] = inv_vals[10]

colors1 = reversed_color_map1(np.linspace(0., 1, 128))
colors2 = reversed_color_map2(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

plt.pcolor(data, cmap=mymap)
plt.colorbar()