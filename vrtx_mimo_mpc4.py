# Vortex Control MIMO MPC (4 Invariants) [Paper figs come from here]
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
import mpctools as mpc

#%% Parameter Setup #############################################################
n = 5 # number of vortices
act = 1 # number of fixed vortex actuators
h_omega = 0#0.01 # frequency of oscillation of href
ep = -.0 # amplitude of oscillation


# Run this block if you DO want random vortex configs (good for playing) ######
# note: subtract .5 in rand() so that vortices can have -ve values, same for kappa0
#np.random.seed(0)
'''
r0 = np.random.multivariate_normal(mean = [5,5], cov = [[20, 0], [0, 20]], size = n)
r0[:-act] = r0[:-act]+10
kappa0 = 2*(np.random.rand(n)-.5)
'''
###############################################################################

# Run this block if you don't want radnomly generated vortex configs ##########
'''
r0 = 5*np.array([[2, 0], 
               [4, 0],
                [0, 0],
                [-2,0],
                [-10, 1],
                [1, 1],
               [0, -1],
               [0, 1]], dtype = 'float64')
'''
'''
r0 = 15*np.array([[.5, 0], 
               [1, 0],
                [0, 0],
                [-1, 1],
                [1, 1],
               [0, -1],
               [0, 1]], dtype = 'float64')

'''
offset1 = np.deg2rad(90)
offset2 = np.deg2rad(90)
rg = 1*np.array([[1*np.cos(offset1), 1*np.sin(offset1)], 
               [1., 0.], 
                [-1*np.cos(offset2), -1*np.sin(offset2)],
                [-1, 0],
                [2, 2],
                [0, -2],
               [0, 2],
               [2, 0]], dtype = 'float64')

offsetB = np.deg2rad(90)
offsetC = np.deg2rad(90)
r0 = 1*np.array([[1*np.cos(offsetC), 1*np.sin(offsetC)], 
               [1., 0.], 
                [-1*np.cos(offsetB), -1*np.sin(offsetB)],
                [-1, 0],
                [2 ,2],
                [0, -2],
               [0, 2],
               [2, 0]], dtype = 'float64')

kappa0 = 1*np.ones(n) # initial vortex strength
#kappa0[:n-act] = np.array([.5, 1, -2, -1.4])
#kappa0[1] = -2
###############################################################################
#k1 = 2
#kappa0[2:4] = -kappa0[2:4]
ctrl = []
t_print = []
 
samples = 10000 # no of samples/grid points
tmax = 2000 #samples #final time
t = np.linspace(0, tmax, samples) # time array
#%%############################################################################
# MPC Tools variables

Delta = .01
Nt = 3
Nx = 4*act+8
Nu = act

def cons_law(x, u):
    """Continuous-time ODE model."""
    
    dxdt = []
    for i in range(4*(act+1)):
        dxdt+=[0]
    
    dxdt += [np.dot(x[:act], u)]
    dxdt += [np.dot(x[act:2*act], u)]
    dxdt += [np.dot(x[2*act:3*act], u)]
    dxdt += [np.dot(x[3*act:4*act], u)]
    
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
vrtics = mpc.DiscreteSimulator(cons_law, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(cons_law, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)

# Define stage cost and terminal weight.
Q1 = 100000*0
Q2 = 200*0
Q3 = 400*0
Q4 = 400*0
R = 1

def lfunc(x,u):
    """Standard quadratic stage cost."""
    return mpc.mtimes(u.T, R, u) + mpc.mtimes((x[-3]-x[-7]).T, Q2, (x[-3]-x[-7])) + mpc.mtimes((x[-4]-x[-8]).T, Q1, (x[-4]-x[-8])) + mpc.mtimes((x[-2]-x[-6]).T, Q3, (x[-2]-x[-6])) + mpc.mtimes((x[-1]-x[-5]).T, Q4, (x[-1]-x[-5]))

l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -1*np.ones((Nu,))}
ub = {"u" : 1*np.ones((Nu,))}

# Make optimizers.
N = {"x":Nx, "u":Nu, "t":Nt}

#%% 
def k_cross(r):
    '''
    returns cross product of a 2D vector r with k^
    '''
    cross = np.array([-r[1], r[0]])
    return cross

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

def feedback3(r, href):
    '''
    KRONIC u calculator
    '''

    B = np.zeros((n-act,2,act))
    gH = np.zeros((n-act,2))
    gA = np.zeros((n-act,2))
    gX = np.zeros((n-act,2))
    gY = np.zeros((n-act,2))
    
    # Gradient of Hamiltonian
    for i in range(n-act):
        for j in range(n-act):
            if j != i:
                gH[i] += -(kappa0[j]*kappa0[i]/(2*np.pi))*(r[i]-r[j])/np.linalg.norm(r[i]-r[j])**2
                
    # Gradient of Angular Impulse
    for i in range(n-act):
        gA[i] = 2*kappa0[i]*r[i]   
         
    # Gradient of Linear impulse in X
    for i in range(n-act):
        gX[i,0] = kappa0[i]  
         
    # Gradient of Linear impulse in Y
    for i in range(n-act):
        gY[i,1] = kappa0[i] 
            
    # ordering is last vortex first row and so on. Must reverese order on output
    for i in range(n-act):
        for j in range(1,act+1):
            B[i,:,j-1] = (1/(2*np.pi))*k_cross(r[i]-r[-j])/np.linalg.norm(r[i]-r[-j])**2
    
    gHB = np.zeros(act)
    for i in range(-act+1, 1):
        gHB[i] = np.dot(gH.flatten(), B[:,:,i].flatten())
        
    gAB = np.zeros(act)
    for i in range(-act+1, 1):
        gAB[i] = np.dot(gA.flatten(), B[:,:,i].flatten())
        
    gXB = np.zeros(act)
    for i in range(-act+1, 1):
        gXB[i] = np.dot(gX.flatten(), B[:,:,i].flatten())
    
    gYB = np.zeros(act)
    for i in range(-act+1, 1):
        gYB[i] = np.dot(gY.flatten(), B[:,:,i].flatten())
        
    xi = np.concatenate((gHB[::-1],gAB[::-1], gXB[::-1], gYB[::-1], np.array([href]), np.array([aref]), np.array([xref]), np.array([yref]), np.array([H_cal(r)]), np.array([ang_imp(r)]), np.array([x_imp(r)]), np.array([y_imp(r)])))
    #print(xi)
    solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=xi, lb=lb, ub=ub,verbosity=0)
    Nsim = 1
    #print(xi)
    
    #times = Delta*Nsim*np.linspace(0,1,Nsim+1)
    xo = np.zeros((Nsim+1,Nx))
    xo[0,:] = xi
    inp = np.zeros((Nsim,Nu))
    
    for cnt in range(Nsim):
        solver.fixvar("x", 0, xo[cnt,:])  
    
        # Solve nlp.
        solver.solve()   
    
        inp[cnt,:] = np.array(solver.var["u",0,:]).flatten() 
        xo[cnt+1,:] = vrtics.sim(xo[cnt,:],inp[cnt,:])
    '''    
    global ctrl
    ctrl += [inp[0]]
    ''' 
    return inp[0]

def biot_sav(t, r):
    '''
    Function to throw into odeint
    '''
    r = r.reshape(n,2)
    dr_dt = np.zeros((n,2))

    kappa0[-act:] = feedback3(r, href+ep*np.cos(h_omega*t)) # KRONIC 
    #kappa0[-act:] = np.cos(feedback1(r)*t)
    #kappa0[-1] = k1 # constant
    
    for i in range(n-act): # in range, subtract number of vortices you want to keep still
        for j in range(n):
            if j != i:
                dr_dt[i] += (kappa0[j]/(2*np.pi))*k_cross(r[i]-r[j])/np.linalg.norm(r[i]-r[j])**2
    
    print(t)    
    '''
    global t_print
    t_print += [t]
    '''
    dr_dtflat = dr_dt.flatten()
    return dr_dtflat

r0flat = r0[:n].flatten()
href = H_cal(rg[:n])
aref = ang_imp(rg[:n]) + 125
xref = x_imp(rg[:n])
yref = y_imp(rg[:n])
check = solve_ivp(lambda t, r: biot_sav(t, r), [t[0], t[-1]], r0flat, t_eval=t, method = 'LSODA', rtol = 1e-8, atol = 1e-8)
sol = check.y.T

x = sol[:,::2]
y = sol[:,1::2]
#%% Reconstructing the input

R0 = np.average(np.linalg.norm(r0[:n-act], axis = 1))
t_star = np.average(abs(kappa0[:n-act]))*t/(2*np.pi*R0**2)
#t_print = np.average(abs(kappa0[:n-act]))*np.array(t_print)/(2*np.pi*R0**2)

sol_complex = sol[:,::2] + 1j*sol[:,1::2] # Using complex positions to simplify some algebra
ctrl = np.zeros((samples, act))
sol_ctrl = sol.reshape((-1, n, 2))
refer = href+ep*np.cos(h_omega*t)

for i in range(samples):
    ctrl[i,:] = feedback3(sol_ctrl[i], refer[i])
    print(t[i])
    #ctrl[i,:] = np.cos(feedback1(sol_ctrl[i])*t[i])
    #ctrl[i,:] = k1 #feedback3(sol_ctrl[i], refer[i])
  
#%% Data-processing ###########################################################

ang = np.zeros(samples)
for i in range(n-act):
    ang += kappa0[i]*abs(sol_complex[:,i])**2
    
x_im = np.zeros(samples)
for i in range(n-act):
    x_im += kappa0[i]*sol_complex[:,i].real
    
y_im = np.zeros(samples)
for i in range(n-act):
    y_im += kappa0[i]*sol_complex[:,i].imag
    
hamil = 0
for i in range(n-act): # These loops creates a basis for the Hamiltonian
    for j in range(i+1,n-act):
        hamil += kappa0[i]*kappa0[j]*(np.log(abs(sol_complex[:,i]-sol_complex[:,j])))
        
hamil = -hamil/(4*np.pi)
        
#%% Animation Block ###########################################################

plt.rcParams['font.size'] = '14'

xp, yp = np.meshgrid(np.arange(-3, 3, 0.05),np.arange(-3,3, 0.05))
dt = t[1] - t[0]
samples = len(t)

# def single_step(x0, y0, idx):
#     d_dt = [xp*0, yp*0]
    
#     for i in range(n-act):
#         den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-7
#         d_dt[0] += -(kappa0[i]/(2*np.pi))*(y0 - y[idx, i])/den
#         d_dt[1] += (kappa0[i]/(2*np.pi))*(x0 - x[idx, i])/den
    
#     for i in range(n-act,n):
#         den = (((x0-x[idx, i])**2 + (y0-y[idx, i])**2)**.5) + 1e-7
#         d_dt[0] += -(ctrl[idx,i-(n-act)]/(2*np.pi))*(y0 - y[idx, i])/den
#         d_dt[1] += (ctrl[idx,i-(n-act)]/(2*np.pi))*(x0 - x[idx, i])/den
    
    
#     return x0 + dt*d_dt[0], y0 + dt*d_dt[1]

# def tracer_soln(xp,yp):
#     sol = [[xp,yp]]
    
#     for i in range(samples):
#         xp, yp = single_step(xp, yp, i)
#         sol += [[xp,yp]]
    
#     return sol

fig = plt.figure()
# sol = tracer_soln(xp,yp)

ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-3, 30), ylim=(-3, 30)) # reg +/- 1.7, mixing +/- 5
grid_length = len(sol[1][0].flatten())


# tracer = plt.scatter([], [], marker='.', linewidth = 0)
# grid_length = len(sol[1][0].flatten())
# offset = np.zeros((grid_length,2))
# offset[:,0] = sol[0][0].flatten()/R0
# offset[:,1] = sol[0][1].flatten()/R0
# tracer.set_offsets(offset)
# color_map = plt.get_cmap('jet')
# colors = np.zeros((grid_length,4))
# # Gaussian color map
# for i in range(grid_length):
#     A = 1 # Amplitude of Gaussian
#     sigma = 2 # Variance
#     xg = offset[i,0]
#     yg = offset[i,1]
#     z = A*np.exp(-(xg**2/(2*sigma**2)+yg**2/(2*sigma**2)))
#     colors[i,:] = color_map(z)
    
# tracer.set_color(colors)


fig2 = plt.figure()
ax2 = fig2.add_subplot(511,ylabel = '$H$', xlim = (0,max(t_star)))
ax3 = fig2.add_subplot(513,ylabel = '$u$', xlabel = '$t*$', xlim = (0,max(t_star)))
ax4 = fig2.add_subplot(512,ylabel = '$A$', xlim = (0,max(t_star)))
ax5 = fig2.add_subplot(514,ylabel = '$X$', xlim = (0,max(t_star)))
ax6 = fig2.add_subplot(515,ylabel = '$Y$', xlim = (0,max(t_star)))

ax2.plot(t_star, hamil, color = 'grey', label = 'instantaneous $H$')
ax2.plot(t_star, refer, color = 'orange', label = 'reference')
ax4.plot(t_star, ang, color = 'grey', label = 'instantaneous $A$')
ax4.plot(t_star, aref+0*t_star, color = 'orange', label = 'reference $A$')
ax5.plot(t_star, x_im, color = 'grey', label = 'instantaneous $X$')
ax5.plot(t_star, xref+0*t_star, color = 'orange', label = 'reference $X$')
ax6.plot(t_star, y_im, color = 'grey', label = 'instantaneous $Y$')
ax6.plot(t_star, yref+0*t_star, color = 'orange', label = 'instantaneous $Y$')
ax3.plot(t_star, ctrl, color = 'grey', label = 'fixed vortex circulation')
#ax3.plot(t_print, ctrl, color = 'grey', label = 'fixed vortex circulation')

ax2.legend()
ax3.legend()

particle, = ax.plot([],[], marker='o', color='black', linewidth = .0,markersize = 8) # was 15 for v1, mixing is 8
particle2, = ax.plot([],[], marker='o', color='black', linewidth = .0,markersize = 15)

particle3, = ax2.plot([],[], marker='o', color='black', linewidth = .0)
traj2, = ax2.plot([],[], color='red', alpha=0.6)
particle4, = ax3.plot([],[], marker='o', color='black', linewidth = .0)
traj3, = ax3.plot([],[], color='blue', alpha=0.6)
#time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#hamil_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
#k_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
#colors = plt.cm.bwr(np.sign(kappa0[:-act]))
colors = plt.cm.brg(np.linspace(0, 1, 7))
tr = []

for k in range(n-act):
    traj, = ax.plot([],[], color=colors[k], alpha=0.6,linewidth = 3)
    tr += traj,

def update1(i):
    particle.set_data(x[i,:-act]/R0,y[i,:-act]/R0)
    particle2.set_data(x[i,-act:]/R0,y[i,-act:]/R0)
    '''
    if ctrl[i]/abs(ctrl[i]) >0:
        ax.set_facecolor('xkcd:pale blue')
    else: 
        ax.set_facecolor('xkcd:pale pink')
    '''   
    for l in range(n-act):
        tr[l].set_data(x[i-500:i+1,l]/R0,y[i-500:i+1,l]/R0)
    
    #time_text.set_text('$t^*$ = %.2f' % t_star[i])
    #hamil_text.set_text('$H$ = %.6f' % hamil[i])
    
    #tracers 
    grid_length = len(sol[1][0].flatten())
    
    offset = np.zeros((grid_length,2))
    offset[:,0] = sol[i][0].flatten()/R0
    offset[:,1] = sol[i][1].flatten()/R0

    #tracer.set_offsets(offset)
    
    return particle, particle2, tr

def update2(i):
    particle3.set_data(t_star[i], hamil[i])
    traj2.set_data(t_star[:i+1], hamil[:i+1])
    particle4.set_data(t_print[i], ctrl[i])
    traj3.set_data(t_print[:i+1], ctrl[:i+1])
#ani2 = animation.FuncAnimation(fig2, update2, frames=range(0, len(t), 1+int(samples/500)), interval=1)
ani = animation.FuncAnimation(fig, update1, frames=range(0, len(t), 1+int(samples/8000)), interval=1)

#plt.tight_layout()
plt.show()

#%% Cumulative sum of input**2

# abs_ctrl = ctrl**2
# ucum = np.cumsum(abs_ctrl.sum(axis = 1))
# figcum = plt.figure()
# axcum = figcum.add_subplot(111)

# axcum.plot(t_star, ucum)
# plt.xlim(0, max(t_star))
# plt.ylabel('$\sum_0^i u_i^Tu_i$')
# plt.title('cumulative sum of input (MPC)')
# plt.xlabel('$t*$')
# plt.tight_layout()


#%% Save variables back for Python
#np.savez('vrtxdata3.npz', R0, act, n, ctrl, hamil, href, kappa0, sol, sol_complex, sol, t, t_star, tmax, x, y)

# %% Save variables in MATLAB

'''
from scipy import io

io.savemat('chaos2periodic.mat',{'Q1':Q1, 'Q2':Q2, 'Q3':Q3, 'Q4':Q4, 'act':act, 'aref':aref, 'ctrl':ctrl, 'href':href, 'xref':xref, 'yref':yref,'n':n, 't':t, 'r0':r0, 'rg':rg, 'sol':sol, 'sol_complex':sol_complex, 'kappa0':kappa0})
test = io.loadmat('chaos2qp.mat')
'''

