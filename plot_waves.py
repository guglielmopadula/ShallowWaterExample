import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

n=50
X, Y = np.meshgrid(np.arange(n+2),  np.arange(n+2))
Hall=np.load("Hall.npy")
Hall=Hall[0:-1:10]

#plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.plot_surface(X,Y,Hall[0])
ax.set_zlim(0.5,3)
def animate(i):
    ax.cla()
    ax.plot_surface(X, Y, Hall[i],cmap=cm.coolwarm)
    ax.set_zlim(0.5,3)
    return fig
lin_ani=animation.FuncAnimation(fig,animate,frames=len(Hall))
FFwriter = animation.FFMpegWriter( fps=100)
lin_ani.save('animation.mp4', writer = FFwriter)
