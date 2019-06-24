#using numpy arrays
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
pos = np.array([[-1, 0], [1, 0]])
vel = np.array([[0, -1], [0, 1]])
pos = np.reshape(pos, (2,2,1))
vel = np.reshape(vel, (2,2,1))
mass = np.array([4 / GRAVITATIONAL_CONSTANT, 4 / GRAVITATIONAL_CONSTANT])
TIME_STEP = 0.1  # s
NUMBER_OF_TIME_STEPS = 10
PLOT_INTERVAL = 1

ac_s = np.zeros((2,2,1))
for step in tqdm(range(NUMBER_OF_TIME_STEPS + 1)):
    if step % PLOT_INTERVAL == 0:
        fig, ax = plt.subplots()
        ax.scatter(pos[:,0,step], pos[:,1,step])
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("t = {:8.4f} s".format(step * TIME_STEP))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        output_file_path = Path("positions", "{:016d}.png".format(step))
        output_file_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_file_path)
        plt.close(fig)
    acceleration = np.array([])
    for i in range(len(pos)):
        #acceleration = np.array([])
        for j in range(len(pos)):
            if i==j:
                continue
            
            dist_vect = pos[j,:,step]-pos[i,:,step]
            dist = np.sqrt(np.sum(dist_vect**2))
            acc = (GRAVITATIONAL_CONSTANT*mass[j]*dist_vect)/(dist**2)
            acceleration = np.hstack([acceleration, acc])
    acceleration = np.reshape(acceleration, (2,2,1))
    ac_s = np.dstack([ac_s, acceleration])
    new_pos = pos[:,:,step]+(vel[:,:,step]*TIME_STEP)
    pos = np.dstack([pos, new_pos])
    new_vel = vel[:,:,step]+(ac_s[:,:,step+1]*TIME_STEP)
    vel = np.dstack([vel, new_vel])
#print(ac_s)







