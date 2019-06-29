#using numpy arrays

#time taken for given 1000000 iterations is 27m 3s(avg of 3 runs)
#iterations reduced to 100000 for more manageable levels 

#importing required libraries
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
pos = np.array([[-1, 0], [1, 0]])
vel = np.array([[0, -1], [0, 1]])
nob = len(pos)                          #no of bodies in space
nod = 2                                 #no of dimensions
mass = np.array([4 / GRAVITATIONAL_CONSTANT, 4 / GRAVITATIONAL_CONSTANT])
TIME_STEP = 0.001  # s
NUMBER_OF_TIME_STEPS = 100000
PLOT_INTERVAL = 100
POSI = np.reshape(pos, (nob,2,1))

#asserting

assert len(pos) == len(vel) == len(mass)
assert len(pos[:]) == nod 
assert len(vel[:]) == nod

#Main Execution


for step in tqdm(range(NUMBER_OF_TIME_STEPS + 1)):                          #repeating for every time
    # if step % PLOT_INTERVAL == 0:
    #     fig, ax = plt.subplots()
    #     ax.scatter(pos[:,0], pos[:,1])
    #     ax.set_aspect("equal")
    #     ax.set_xlim(-1.5, 1.5)
    #     ax.set_ylim(-1.5, 1.5)
    #     ax.set_title("t = {:8.4f} s".format(step * TIME_STEP))
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     output_file_path = Path("positions", "{:016d}.png".format(step))
    #     output_file_path.parent.mkdir(exist_ok=True)
    #     fig.savefig(output_file_path)
    #     plt.close(fig)
    acceleration = np.array([])                                            #an empty numpy array to store the acceleration of bodies for current time step
    for i in range(nob):
        for j in range(nob):
            if i==j:                                                       #to not calculate for itself
                continue
            
            dist_vect = pos[j,:]-pos[i,:]                                  #x2-x1 and y2-y1 are stored in this vector
            dist = np.sqrt(np.sum(dist_vect**2))                           #distance between points given by sqrt of sum of x2-x1 and y2-y1
            acc = (GRAVITATIONAL_CONSTANT*mass[j]*dist_vect)/(dist**2)     #acceleration of body due to force
            acceleration = np.hstack([acceleration, acc])                #acceleration of all bodies for current time step
    acceleration = np.reshape(acceleration, (nob,2))                       #reshaping to make it homogenous with data structure
    
    # Semi implicit Euler method to find new position and velocity
    new_vel = vel[:,:]+(acceleration[:,:]*TIME_STEP)
    vel = new_vel
    temppos = pos[:,:]+(vel*TIME_STEP)
    POSI = np.dstack([POSI, temppos])
    pos = temppos

