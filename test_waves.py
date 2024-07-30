import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from tqdm import trange
# Define the grid size
n = 50  # number of grid cells (Squared grid)

dt = 0.005  # sec
dx = 1  # meter
dy = 1  # meter
g = 9.8  # m/s^2
SimTime = 50  # Simulation time in sec

# We do not need to see all the timesteps
FrameOut = 10  # Number of timesteps between frames

H = np.ones((n+2, n+2))  # displacement matrix (this is what gets drawn)
U = np.zeros((n+2, n+2))  # x velocity
V = np.zeros((n+2, n+2))  # y velocity

# create initial displacement (the initial condition is made as a lump on the water table)
x, y = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
R = np.sqrt(x**2 + y**2)
Z = (np.sin(R) / R)
Z = np.maximum(Z, 0)
Hmat=scipy.io.loadmat("H.mat")["Hall"]

# add displacement to the height matrix
w = Z.shape[0]
i = slice(9, 9+w)
j = slice(19, 19+w) 
ip1=slice(10, 10+w)
im1=slice(8, 8+w)
jp1=slice(20, 20+w)
jm1=slice(18, 18+w)
H[i,j] = H[i,j] + Z

Hx = np.zeros((n+1, n+1))
Hy = np.zeros((n+1, n+1))
Ux = np.zeros((n+1, n+1))
Uy = np.zeros((n+1, n+1))
Vx = np.zeros((n+1, n+1))
Vy = np.zeros((n+1, n+1))

# Set up the figure and axis
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(np.arange(n+2),  np.arange(n+2))

Hall=np.zeros((int(SimTime/dt)+1,n+2,n+2))
Hall[0]=H
for k in trange(int(SimTime/dt)):
    # blending the edges keeps the function stable
    H[:, 0] = H[:, 1]
    H[:, n+1] = H[:, n]
    H[0, :] = H[1, :]
    H[n+1, :] = H[n, :]

    # reverse direction at the x edges
    U[0, :] = -U[1, :]
    U[n+1, :] = -U[n, :]

    # reverse direction at the y edges
    V[:, 0] = -V[:, 1]
    V[:, n+1] = -V[:, n]

    # First half step (The Lax-Wendroff method)
    i = slice(0, n+1)
    j = slice(0, n+1)
    ip1=slice(1, n+2)
    jp1=slice(1, n+2)



    # height
    Hx[i, j] = (H[ip1,jp1] + H[i, jp1]) / 2 - dt / (2 * dx) * (U[ip1, jp1] - U[i, jp1])
    Hy[i, j] = (H[ip1, jp1] + H[ip1, j]) / 2 - dt / (2 * dy) * (V[ip1, jp1] - V[ip1, j])

    # x momentum
    Ux[i, j] = (U[ip1, jp1] + U[i, jp1]) / 2 - dt / (2 * dx) * (U[ip1, jp1]**2 / H[ip1, jp1]  
            - U[i, jp1]**2 / H[i, jp1] + 
        g / 2 * H[ip1, jp1]**2 - g / 2 * H[i, jp1]**2)
    
    
    Uy[i, j] = (U[ip1, jp1] + U[ip1, j]) / 2 - dt / (2 * dy) * ( (V[ip1, jp1] * (U[ip1, jp1] / H[ip1, jp1])) - 
        (V[ip1, j] * (U[ip1, j] / H[ip1, j])))

    # y momentum
    Vx[i, j] = (V[ip1, jp1] + V[i, jp1]) / 2 - dt / (2 * dx) * (
        (U[ ip1, jp1] * (V[ip1, jp1] / H[ip1, jp1])) - 
        (U[i, jp1] * (V[i, jp1] / H[i, jp1])))
    
    Vy[i, j] = (V[ip1, jp1] + V[ip1, j]) / 2 - dt / (2 * dy) * (
        (V[ip1, jp1]**2 / H[ip1, jp1] + g / 2 * H[ip1, jp1]**2) - 
        (V[ip1, j]**2 / H[ip1, j] + g / 2 * H[ip1, j]**2))

    # Second half step (The Lax-Wendroff method)
    i = slice(1, n+1)
    j = slice(1, n+1)
    im1= slice(0,n)
    jm1= slice(0,n)

    # height
    H[i, j] = H[i, j] - (dt / dx) * (Ux[i,jm1] - Ux[im1,jm1]) -  (dt / dy) * (Vy[im1, j] - Vy[im1,jm1])

    # x momentum
    U[i, j] = U[i, j] - (dt / dx) * (
        (Ux[i,jm1]**2 / Hx[i,jm1] + g / 2 * Hx[i,jm1]**2) - (Ux[im1,jm1]**2 / Hx[im1,jm1] + g / 2 * Hx[im1,jm1]**2)) -   (dt / dy) * ((Vy[im1, j] * Uy[im1, j] / Hy[im1, j]) -        (Vy[im1,jm1] * Uy[im1,jm1] / Hy[im1,jm1]))

    # y momentum
    V[i, j] = V[i, j] - (dt / dx) * (
        (Ux[i,jm1] * Vx[i,jm1] / Hx[i,jm1]) - 
        (Ux[im1,jm1] * Vx[im1,jm1] / Hx[im1,jm1])) -(dt / dy) * ((Vy[im1, j]**2 / Hy[im1, j] + g / 2 * Hy[im1, j]**2) - (Vy[im1,jm1]**2 / Hy[im1,jm1] + g / 2 * Hy[im1,jm1]**2))
    
    Hall[k+1]=H

np.save("Hall.npy",Hall)