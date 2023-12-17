# Code which generates the data for vortex trajectory figures. Sweep through multiple Invariant values with a fixed actuator position

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
import mpctools as mpc

#%% Parameter Setup ###########################################################
n = 5 # number of vortices
act = 1 # number of fixed vortex actuators

###############################################################################

# Run this block if you don't want radnomly generated vortex configs ##########

offset1 = np.deg2rad(90)
offset2 = np.deg2rad(90)
rg = 1*np.array([[1*np.cos(offset1), 1*np.sin(offset1)], 
               [1., 0.], 
                [-1*np.cos(offset2), -1*np.sin(offset2)],
                [-1, 0],
                [-2, 2],
                [0, -2],
               [0, 2],
               [2, 0]], dtype = 'float64')

kappa0 = 1*np.ones(n) # initial vortex strength
#kappa0[1] = -2
###############################################################################
kappa0[-act:] = 1
#k1 = 2
#kappa0[2:4] = -kappa0[2:4]
 
samples = 9000 # no of samples/grid points 2000
tmax = 3600 #samples #final time 800
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

# Define stage cost and terminal weight. Change these weights depending on the case one is interested in (if you want to enforce symmetry for eg.)
Q1 = 1000000
Q2 = 800*0
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



#act_locx, act_locy = 2,2 initial set of plots
act_locx, act_locy = 2, 2
inv_vals = np.linspace(-.2,.2,20) # for hamiltonian
#inv_vals = np.linspace(-4,4,20) # for linear impulse
#inv_vals = np.linspace(-5,5,20) # for angular impulse

#%% Kernel

def run_sim(idx):  
    try:
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
        
            kappa0[-act:] = feedback3(r, href) # KRONIC 
            #kappa0[-act:] = np.cos(feedback1(r)*t)
            #kappa0[-1] = k1 # constant
            
            for i in range(n-act): # in range, subtract number of vortices you want to keep still
                for j in range(n):
                    if j != i:
                        dr_dt[i] += (kappa0[j]/(2*np.pi))*k_cross(r[i]-r[j])/np.linalg.norm(r[i]-r[j])**2
               
            '''
            global t_print
            t_print += [t]
            '''
            
            print(t)
            dr_dtflat = dr_dt.flatten()
            return dr_dtflat
        
        aref = ang_imp(rg[:n])
        xref = x_imp(rg[:n])
        yref = y_imp(rg[:n])
        href = H_cal(rg[:n])
                
        href+= inv_vals[idx]
        #xref+= inv_vals[idx]
        #yref+= inv_vals[idx]
        #aref+= inv_vals[idx]

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
    
        r0[n-act] = np.array([act_locx, act_locy])
        r0flat = r0[:n].flatten()
        
        check = solve_ivp(lambda t, r: biot_sav(t, r), [t[0], t[-1]], r0flat, t_eval=t, method = 'LSODA', rtol = 1e-8, atol = 1e-8, min_step = 1e-4)
        sol = check.y.T
        
        x = sol[:,::2]
        y = sol[:,1::2]
        
        sol_complex = x + 1j*y # Using complex positions to simplify some algebra
        ctrl = np.zeros((samples, act))
        sol_ctrl = sol.reshape((-1, n, 2))
        refer = href
        for i in range(samples):
            ctrl[i,:] = feedback3(sol_ctrl[i], refer)
            print(t[i])
            
        peri = 0
        for i in range(n-act):
            for j in range(i+1, n-act):
                peri += abs(sol_complex[:,i]-sol_complex[:,j])
                
        # Data-processing #############################################################
        
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
        
        print("idx " + str(idx) + " done")
            
        return [sol_complex, ctrl, hamil, ang, x_im, y_im]
    
    except:
        cmplx_act = act_locx+ 1j*act_locy
        bad_array = np.zeros((samples, n), dtype='complex128')
        bad_array[:,-1] += cmplx_act
        return [bad_array, np.zeros((samples,1)),np.zeros(samples), np.zeros(samples), np.zeros(samples), np.zeros(samples)]


import multiprocessing as mp

pool = mp.Pool(20)
results = pool.map(run_sim, [idx for idx in range(0,len(inv_vals))])
pool.close()

#%% Organizing data

vrtx_trajs = []
vrtx_ins = []
vrtx_hamils = []
vrtx_angs = []
vrtx_cumins = []
vrtx_exes = []
vrtx_whys = []

for i in range(0,len(inv_vals)):
    vrtx_trajs += [results[i][0]]
    vrtx_ins += [results[i][1]]
    vrtx_hamils += [results[i][2]]
    vrtx_angs += [results[i][3]]
    vrtx_cumins += [sum(abs(results[i][1]))]
    vrtx_exes += [results[i][4]]
    vrtx_whys += [results[i][5]]
    
vrtx_trajs = np.array(vrtx_trajs)
vrtx_cumins = np.array(vrtx_cumins)
vrtx_ins = np.array(vrtx_ins)
vrtx_hamils = np.array(vrtx_hamils)
vrtx_angs = np.array(vrtx_angs)
vrtx_exes = np.array(vrtx_exes)
vrtx_whys = np.array(vrtx_whys)


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
        
vrtx_error_final = vrtx_hamils[:,-1] - href
vrtx_ang_error_final = vrtx_angs[:,-1] - aref

#%% Plotting the landscape

# plt.figure()
# plt.imshow(vrtx_error_final.reshape(1,-1), origin = 'left')
# plt.title('final hamiltonian error')
# plt.colorbar()

# plt.figure()
# plt.imshow(vrtx_ang_error_final.reshape(1,-1), origin = 'left')
# plt.title('final ang imp error')
# plt.colorbar()

# plt.figure()
# plt.imshow(vrtx_cumins.reshape(1,-1), origin = 'left')
# plt.title('cumulative "energy" used')
# plt.colorbar()

# plt.figure()
# plt.suptitle('cells full trajectory')
# for i in range(1,len(href),2):

#     plt.subplot(1,len(href),i+1)
#     plt.plot(vrtx_trajs[i,:,:-1].real, vrtx_trajs[i,:,:-1].imag)
#     plt.scatter(vrtx_trajs[i,0,:-1].real, vrtx_trajs[i,0,:-1].imag, c = 'black')
#     #plt.scatter(vrtx_trajs[i,0,-1].real, vrtx_trajs[i,0,-1].imag, c = 'blue')
#     plt.axis('equal')
#     plt.axis('off')
    
# #%% cells steady state
# plt.rcParams['font.size'] = '25'
# c = 1
# t_p = np.linspace(0,2*np.pi,100)
# for i in range(1,len(href),1):

#     plt.figure()
#     plt.plot(vrtx_trajs[i,:samples-300,:-1].real, vrtx_trajs[i,:samples-300,:-1].imag, 'grey', alpha = .2, linewidth = 3)
#     plt.plot(vrtx_trajs[i,samples-300:,:-1].real, vrtx_trajs[i,samples-300:,:-1].imag,linewidth = 3)
#     #ax.scatter(vrtx_trajs[i,0,:-1].real, vrtx_trajs[i,0,:-1].imag, c = 'black')
#     plt.plot(np.cos(t_p), np.sin(t_p), c='black', linestyle='--', linewidth = 3)
#     plt.scatter(vrtx_trajs[i,0,-1].real, vrtx_trajs[i,0,-1].imag, c = 'black' ,s = 300, marker = 'X')
#     plt.xlim(-2.5,2.5)
#     plt.ylim(-2.5,2.5)
#     #print(aref[i])
    
#     #plt.axis('off')
#     #plt.axis('equal')
#     plt.gca().set_aspect('equal', adjustable='box')
#     #plt.savefig('fig'+str(c)+'.pdf')
#     c+=1
# #%%
# plt.figure()
# plt.suptitle('cells hamiltonian')
# for i in range(len(href)):
#         plt.subplot(1,len(href),i+1)
#         plt.plot(t, vrtx_hamils[i,:])
#         plt.plot(t, href[i] + 0*vrtx_hamils[i,:])
        
# plt.figure()
# plt.suptitle('cells angular impulse')
# for i in range(len(href)):
        
#         plt.subplot(1,len(href),i+1)
#         plt.plot(t, vrtx_angs[i,:])
#         plt.plot(t, aref + 0*vrtx_angs[i,:])
        
# plt.figure()
# plt.suptitle('cells X')
# for i in range(len(href)):
#         plt.subplot(1,len(href),i+1)
#         plt.plot(t, vrtx_exes[i])
#         plt.plot(t, xref + 0*vrtx_angs[i])

# plt.figure()
# plt.suptitle('cells Y')
# for i in range(len(href)):
#         plt.subplot(1,len(href),i+1)
#         plt.plot(t, vrtx_whys[i])
#         plt.plot(t, yref + 0*vrtx_angs[i])

#%%
# plt.rcParams['font.size'] = '35'
# c = 1

# lnst = ['solid', 'dashed', 'dashdot', 'dotted']
# for i in range(1,len(href)):
#     plt.figure()
#     for j in range(4):
#         N=samples - 1000
#         T = t[1] - t[0]
#         yf = np.fft.fftpack.fft(abs(vrtx_trajs[i,1000:,j]))
#         #xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#         xf = np.fft.fftfreq(N, T)
        
#         #nrm = max(2.0/N * np.abs(yf[1:N//2]))
#         nrm = 1
        
#         plt.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2])/nrm, linestyle = lnst[j], linewidth = 5)
#         #plt.xlim(0,.6)
#         plt.ylim(0,1.2)
#         plt.xlim(0,.4)
#         plt.tight_layout()
#         plt.yticks([.5, 1])
#     #plt.savefig('fig'+str(c)+'.pdf')
#     c+=1
    

#%% Save variables
#np.savez('phamil3600_9000.npz', t, vrtx_trajs, vrtx_cumins, vrtx_ins, vrtx_hamils, vrtx_angs, vrtx_exes, vrtx_whys, inv_vals, act_locx, act_locy, kappa0, rg)