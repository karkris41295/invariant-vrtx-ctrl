# Code which reads the fix_act_mv_inv.py generated data to produce plots

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation


filename = '/Users/kartikkrishna/Downloads/vrtxdata/vrtx_fix_act_data/chamilfe3600_9000.npz'
recall = np.load(filename)
data = [recall[key] for key in recall]

t, vrtx_trajs, vrtx_cumins, vrtx_ins, vrtx_hamils, vrtx_angs, vrtx_exes, vrtx_whys, inv_vals, act_locx, act_locy, kappa0, rg = data

### Parameter Setup ###########################################################
n = 5 # number of vortices
act = 1 # number of fixed vortex actuators

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

#%%
vrtx_error_final = (vrtx_hamils[:,-1000] - (href - inv_vals*0))/(href - vrtx_hamils[:,0]) * 100
vrtx_ang_error_final = (vrtx_angs[:,-1000] - (aref + inv_vals*1))* 100/(vrtx_angs[:,0] - (aref + inv_vals*1))

vrtx_x_error_final = vrtx_exes[:,-1000] - (xref + inv_vals*0)
vrtx_y_error_final = vrtx_whys[:,-1000] - (yref + inv_vals*0)

#plt.plot(np.arange(0,20)[5:14], vrtx_y_error_final[5:14])
#plt.plot(np.arange(0,20,1)[5:14], vrtx_x_error_final[5:14], np.arange(0,20,1)[5:14], vrtx_y_error_final[5:14])
# #%% Plotting the landscape

# plt.rcParams['font.size'] = '13'
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
# #%%
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
    
#     col = 700
#     plt.figure()
#     plt.plot(vrtx_trajs[i,:samples-col,:-1].real, vrtx_trajs[i,:samples-col,:-1].imag, 'grey', alpha = .2, linewidth = 3)
#     plt.plot(vrtx_trajs[i,samples-col:,:-1].real, vrtx_trajs[i,samples-col:,:-1].imag,linewidth = 3)
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
# plt.rcParams['font.size'] = '13'


# for i in range(len(href)):
#         plt.figure()
#         plt.title('hamiltonian')
#         plt.plot(t, vrtx_hamils[i,:])
#         plt.plot(t, href[i]-(inv_vals[i]*0) + 0*vrtx_hamils[i,:])
# #%%  

# for i in range(len(href)):
#         plt.figure()
#         plt.title('angular impulse')

#         plt.plot(t, vrtx_angs[i,:])
#         plt.plot(t, aref+(inv_vals[i]*1) + 0*vrtx_angs[i,:])
        
# #%%

# for i in range(len(href)):
#         plt.figure()
#         plt.title('x-impulse')
#         plt.plot(t, vrtx_exes[i])
#         plt.plot(t, xref+(inv_vals[i] *1)+ 0*vrtx_angs[i])

# #%%

# for i in range(len(href)):
#         plt.figure()
#         plt.title('y-impulse')
#         plt.plot(t, vrtx_whys[i])
#         plt.plot(t, yref+(inv_vals[i]*0) + 0*vrtx_angs[i])

#%%
plt.rcParams['font.size'] = '30'
c = 1

lnst = ['solid', 'dashed', 'dashdot', 'dotted']
for i in range(1,len(href)):
    plt.figure()
    for j in range(4):
        N=samples - 1000
        T = t[1] - t[0]
        yf = np.fft.fftpack.fft(vrtx_trajs[i,1000:,j].real)
        #xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        xf = np.fft.fftfreq(N, T)
        
        nrm = max(2.0/N * np.abs(yf[1:N//2]))
        #nrm = 1
        
        plt.plot(xf[0:N//2], 2.0/N * np.abs(yf[0:N//2])/nrm, linestyle = lnst[j], linewidth = 5)
        #plt.xlim(0,.6)
        plt.ylim(0,1.2)
        plt.xlim(0,.1)
        plt.tight_layout()
        plt.yticks([.5, 1])
    #plt.savefig('fig'+str(c)+'.pdf')
    c+=1
    
#%% waterfalll plot

Z = []
for i in range(1,len(href),2):
    plt.figure()
    for j in range(1):
        N=samples - 1000
        T = t[1] - t[0]
        yf = np.fft.fftpack.fft(vrtx_trajs[i,1000:,j].real)
        #xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        xf = np.fft.fftfreq(N, T)
        
        nrm = max(2.0/N * np.abs(yf[1:N//2]))
        #nrm = 1
        
        Z += [2.0/N * np.abs(yf[0:N//2])/nrm]

        plt.plot(xf[0:N//2],2.0/N * np.abs(yf[0:N//2])/nrm, linestyle = lnst[j], linewidth = 5)
        #plt.xlim(0,.6)
        plt.ylim(0,1.2)
        plt.xlim(0,.1)
        plt.tight_layout()
        plt.yticks([.5, 1])
    #plt.savefig('fig'+str(c)+'.pdf')
    c+=1
    
#%%
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.size'] = '10'
figidk = plt.figure()
axidk = figidk.add_subplot(111, projection='3d')

Z = np.array(Z)
#Z[Z < .25] = 0
X = xf[1:N//2]
Y = inv_vals[1::2]
X, Y = np.meshgrid(X,Y)
cutidk = 315
axidk.plot_wireframe(X[1:,1:cutidk],Y[1:,1:cutidk],Z[1:,1:cutidk])

axidk.view_init(53,65)
axidk.set_xlim(0, .1)
axidk.set_zlim(0, 1.6)

#%% Save variables
#np.savez('ang22_3600_9000.npz', t, vrtx_trajs, vrtx_cumins, vrtx_ins, vrtx_hamils, vrtx_angs, vrtx_exes, vrtx_whys, inv_vals, act_locx, act_locy, kappa0, rg)