import numpy as np
import matplotlib.pyplot as plt

def generate_cosmic_muon():
    step = .0005
    angular_range = np.pi/2*np.arange(0,1,step)
    angular_distribution = np.square(np.cos(angular_range)) #Define vertical angular distribution
    angular_distribution = angular_distribution/np.sum(angular_distribution) #Normalisation
    theta = np.random.choice(angular_range,p = angular_distribution)

    phi = 2*np.pi*np.random.random() #Azimuth distribution

    E_range = 10*np.arange(0,1,step) #Muon energy up to 10 GeV
    E_distribution = np.power(E_range + 4.29, -3)/(1 + E_range/854) #Approximate muon energy distribution not including angular dependence
    E_distribution = E_distribution/np.sum(E_distribution)
    Energy = np.random.choice(E_range,p = E_distribution)
    
    return Energy, theta, phi

muon_mass = .106

class Muon:

    def __init__(self, init_pos):        
        m = generate_cosmic_muon()
        
        #Defining things
        self.direction = np.array([np.sin(m[2])*np.sin(m[1]),np.cos(m[2])*np.sin(m[1]),np.cos(m[1])]) #Direction of muon, in the coordinate where phi = 0 is along x and theta = 0 is along z
        self.energy = m[0]
        self.position = init_pos

        #Start the simulation with the muon inside of the matrix at t = 0, not decayed, and in motion.
        self.in_matrix = True #Is the muon still inside of the scintillating array?
        self.time = 0 #Is the muon in motion?
        self.decayed = False #Has the muon decayed?

    def iterate(self, dt):
        #Simple numerical integration scheme, iterating first the position then the energy.
        if self.in_matrix and not self.decayed:
            K = 5.1*10**-28 #4πr^2mc^2, in units of GeVcm^2
            electron_mass = .511*10**-3 #In units of GeV/c^2
            n_e = 3.33*10**23 #Electrons per cm^3
            I = 64.7*10**-9 #Mean excitation energy of polyvinyltoluene
            
            if self.energy > muon_mass:
                beta = np.sqrt(1-1/np.square(self.energy/muon_mass))
                gamma = self.energy/muon_mass
                dx = beta*dt

                W_max = 2*electron_mass*(beta*gamma)**2/(1+2*gamma*electron_mass/muon_mass+(electron_mass/muon_mass)**2) #Maximum energy transfer
                dE = K*n_e/(beta)**2*(np.log(2*electron_mass*(beta*gamma)**2*W_max/I**2)/2-(beta)**2)*dx #Bethe formula without corrections
            else:
                dE = 0
                dx = 0
                gamma = 1
                self.energy = muon_mass
            
            
            proper_time = dt/gamma
            lifetime = 65900 #muon lifetime in cm/c
            if np.random.random() > np.exp(-proper_time/lifetime): #Muon decays
                self.decayed = True 
            
            self.position += dx*self.direction #Iterate position

            if np.abs(dE) < self.energy - muon_mass: #Iterate energy
                self.energy -= dE
            else:
                self.energy = muon_mass

            self.time += 1

            if self.decayed: #Outputs energy when decaying
                return muon_mass
            if dE >= 0: #Outputs energy when losing energy
                return dE
            

#Simulate muons travelling through the detector, with simulated spacial resolution of 1cm and time steps of 1cm/c
     
Detector_dimension = np.array([50,50,50]) #Total detector dimension, in cm according to coordinates defined earlier with origin on one corner
Unit_dimension = np.array([5,1,5]) #Number of units along any axis

Detector_space = np.zeros((Detector_dimension[0],Detector_dimension[1],Detector_dimension[2]))
Detector_output = np.zeros((Unit_dimension[0],Unit_dimension[2]))


init_position = np.array([np.random.random()*Detector_dimension[0],np.random.random()*Detector_dimension[1],0]) #Choosing to generate muons on the top layer of the detector
My_muon = Muon(init_position)

Scintillation_efficiency = 10000 #Photons/MeV of energy dissipated into scintillator
Detection_efficiency = .1 #Probability a photon is detected by SiPM after creation

dt = 1 #Time steps at 1cm/c

while My_muon.in_matrix and not My_muon.decayed: #Simulating muon through detector, first interating muon position before recording said position with deposited energy

    Energy = My_muon.iterate(dt)
    x = np.floor(My_muon.position[0]).astype(int)
    y = np.floor(My_muon.position[1]).astype(int)
    z = np.floor(My_muon.position[2]).astype(int)

    i = np.floor(x/Detector_dimension[0]*Unit_dimension[0]).astype(int)
    j = np.floor(z/Detector_dimension[2]*Unit_dimension[2]).astype(int)
    
    if x >= 0 and x < Detector_dimension[0] and y >= 0 and y < Detector_dimension[1] and z >= 0 and z < Detector_dimension[2]:
        Detector_space[x,y,z] = np.add(Energy,Detector_space[x,y,z]) #Energy inputted into each cell
        Detector_output[i,j] = np.add(Energy,Detector_output[i,j]) #Energy inputted into each detector
        
    else:
        print("exit")
        My_muon.in_matrix = False


if  My_muon.decayed:
    print("Decayed")
print(np.transpose(Detector_output))

x,y,z = np.indices((Detector_dimension[0]+1,Detector_dimension[1]+1,Detector_dimension[2]+1)) #Prepare coordinates
colors = np.zeros(Detector_space.shape + (3,)) #Prepare colors
colors[...,0] = Detector_space/np.max(Detector_space)
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x,y,-z,Detector_space > 0,facecolors=colors)

plt.show()

