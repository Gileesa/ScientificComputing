import numpy as np
import matplotlib.pyplot as plt
import random 
from matplotlib.animation import FuncAnimation

#random.seed(42)
Nx = 100
Ny = 100


def walker(cluster, x, y, cluster_over_time, ps, save_frame):
    global Nx, Ny
    walk_done = False
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    while True:
        dx, dy = random.choice(directions)
        x = (x + dx) % Nx
        y += dy
        right = (x + 1) % Nx
        left = (x - 1) % Nx
        up = (y + 1) if ((y + 1) < Ny and (y + 1) >= 0) else y
        down = (y - 1) if ((y - 1) < Ny and (y - 1) >= 0) else y
        if y >= Ny or y < 0:
            return walk_done, -1, -1, cluster_over_time
        if cluster[y,x] != 1:
            if cluster[y, right] == 1 or cluster[y, left] == 1 or cluster[down, x] == 1 or cluster[up, x] == 1:
                stick = random.random()
                if stick < ps:
                    cluster[y, x] = 1
                    walk_done = True
                    if save_frame == True: 
                        cluster_over_time.append(cluster.copy())
                    return walk_done, x, y, cluster_over_time

def mc_DLA(cluster, num_walkers, ps, save_frame = False):
    global Nx, Ny
    cluster_over_time = []
    cluster = cluster.copy()
    for i in range(num_walkers):
        x0 = random.randint(0, Nx - 1)
        walk_done, x, y, cluster_over_time = walker(cluster, x0, Ny - 1, cluster_over_time, ps, save_frame)
        if walk_done:
            cluster[y,x] = 1
    return cluster, np.array(cluster_over_time)

def animate_cluster(c_over_time, title, name):
        time_step = c_over_time.shape[0]

        fig, ax = plt.subplots()
        
        im = ax.imshow(c_over_time[0], origin="lower", cmap="binary")
        #ins_mask = np.ma.masked_where(obj_mat == 0, obj_mat)
        #ins_img = ax.imshow(ins_mask, origin="lower", cmap="Reds", alpha=0.5)

        ax.set_title(title)

        def update(t):
            im.set_data(c_over_time[t])
            return im,
        
        ani = FuncAnimation(
            fig, 
            update, 
            frames=time_step, 
            interval=100, 
            blit=True)
        ani.save(f"Figures/2.2/{name}.gif", fps=10, dpi=200)
        plt.show()

#question C
#Initializing the cluster
cluster_init = np.zeros((Nx,Ny), dtype=bool)
mid = int(Nx / 2)
cluster_init[0, mid] = 1

cluster, cluster_over_time = mc_DLA(cluster_init, 50000, 1, True)
plt.imshow(cluster, origin='lower', cmap='binary')
plt.title("Monte Carlo DLA cluster")
plt.show()

animate_cluster(cluster_over_time, 'Monte Carlo DLA', 'animation_cluster')

#question D
ps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for ps in ps_list:
    cluster_init = np.zeros((Nx,Ny), dtype=bool)
    mid = int(Nx / 2)
    cluster_init[0, mid] = 1
    cluster, cluster_over_time = mc_DLA(cluster_init, 50000, ps, False)
    plt.imshow(cluster, origin='lower', cmap='binary')
    plt.savefig(f"Figures/2.2/run_ps={ps}.png")
    plt.show()


