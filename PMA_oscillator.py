import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':28})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

########### Dimensionless LLGS #####################
def llgs(m,h,beta,Hk,mp,epsilon):

	global epsilon_prime, alpha
	
	precission=-np.cross(m,h)
	damping=-alpha*np.cross(m,np.cross(m,h))
	dl_stt=-(beta/Hk)*(epsilon+alpha*epsilon_prime)*(np.cross(m,np.cross(m,mp)))
	fl_stt=-(beta/Hk)*(epsilon_prime-alpha*epsilon)*np.cross(m,mp)
	#print('damping = ' + str(damping))
	#print('Tstt = ' + str(dl_stt+fl_stt))
	dm_dtau=precission+damping+dl_stt+fl_stt
	
	return dm_dtau
##################### Vector magnitude ##########################
def mag(M):
	magnitude = np.sqrt((M[0])**2+(M[1])**2+(M[2])**2)
	return magnitude	
#################### dimensionless h_eff #########################
def h_vec(m):
	mz=m[2]
	#h=np.array([0,0,mz])
	h=np.array([0,0,1.0])
	return h
################### epsilon_calculation ##########################
def epsilon(m):
	global P, LAMBDA, mp
	e=(P*LAMBDA**2)/((LAMBDA**2+1)+(LAMBDA**2-1)*np.dot(m,mp))
	
	return e 
#################### constant parameters ############################
gamma=1.76e11;           # Gyromagnetic ratio [(rad)/(s.T)]
mu0=4*np.pi*1e-7 ;      # in T.m/A

q=1.6e-19;               # in Coulomb
hbar=1.054e-34;          # Reduced Planck's constant (J-s)
K_B=1.38064852e-23    #in J/K
#################### parameters related to nanomagnet ################
alpha=0.01              # Gilbert dasigma_SHEing parameter
Ms=1100e3               # in A/m

t_FL=1.6e-9
MTJ_length=75e-9
MTJ_width=35e-9
Area=(np.pi*MTJ_length*MTJ_width)
magVolume = t_FL *  Area  # in m^3
V=magVolume
P=0.57
LAMBDA=2.25
epsilon_prime=-0.1
################ Anisotropy field related ################
mu0_Hk=0.044     # in Tesla
Hk=0.044/mu0
Ku2=mu0*Hk*Ms/2.0
############### External magnetic field #################
#H_ext=np.array([)
#h_ext=H_ext/Hk
############## Spin current portion ######################
I_curr=-0.2e-3
J=I_curr/Area
beta=(hbar/(mu0*q))*(J/(t_FL*Ms))
print(beta)
mp=np.array([1.0, 0, 0])
############ time related portion #######################
stop_time=10.0             # in ns
n=100001
t_ns=np.linspace(0,stop_time,n) # in ns
t=t_ns*1e-9              # converted into second
dim_less_time_factor=(gamma*mu0*Hk)/(1+alpha**2)

tau=dim_less_time_factor*t
h_tau=tau[1]-tau[0]
############# Initial Magnetization #####################
m_vec=np.zeros((n,3))
mx0=-0.999
mz0=np.sqrt(1-mx0**2)
my0=0
m_vec[0,:]=[mx0,my0,mz0]
#print(m[0,:])
ti=0
m_mag=np.zeros(n)
m_mag[ti]=mag(m_vec[ti,:])

########### Loop starts ##############################

while ti<(n-1):
	print('--------------------------------------------------------------------')
	print(' i = ' + str(ti))
	m=m_vec[ti,:]
	k1=llgs(m,h_vec(m),beta,Hk,mp,epsilon(m))
	m=m+h_tau*k1/2.0
	k2=llgs(m,h_vec(m),beta,Hk,mp,epsilon(m))
	m=m+h_tau*k2/2.0
	k3=llgs(m,h_vec(m),beta,Hk,mp,epsilon(m))
	m=m+h_tau*k3
	k4=llgs(m,h_vec(m),beta,Hk,mp,epsilon(m))

	m_vec[ti+1,:]=m_vec[ti,:]+(h_tau/6.0)*(k1+2*k2+2*k3+k4)
	m_mag[ti+1]=mag(m_vec[ti+1,:])
	ti=ti+1

############## Plot the result ########################
fig = plt.figure(figsize=(15,10))
xtick_array=np.array([0,0.2,0.4,0.6,0.8,1.0])
plt.subplot(2,2,1)
plt.plot(t*1e9,m_vec[:,0],linewidth=2.5)
plt.grid()
#plt.xticks(xtick_array)
#plt.xlabel('Time(ns)')
plt.ylabel(r"$m_x$")
plt.ylim([-1.2,1.2])

plt.subplot(2,2,2)
plt.plot(t*1e9,m_vec[:,1],linewidth=2.5)
plt.grid()
#plt.xlabel('Time(ns)')
plt.ylabel(r"$m_y$")
plt.ylim([-1.2,1.2])

plt.subplot(2,2,3)
plt.plot(t*1e9,m_vec[:,2],linewidth=2.5)
plt.grid()
plt.xlabel('Time(ns)')
plt.ylabel(r"$m_z$")
plt.ylim([-1.2,1.2])

plt.subplot(2,2,4)
plt.plot(t*1e9,m_mag,linewidth=2.5)
plt.grid()
plt.xlabel('Time(ns)')
plt.ylabel(r"$m$")
plt.ylim([-1.2,1.2])
plt.tick_params(which='both', direction='in', length=6, width=2, colors='k')
plt.savefig('PMA_Oscillation.pdf', bbox_inches='tight', pad_inches=0.3)
plt.show()
	
