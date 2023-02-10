import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import paths

import sys
import subprocess

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{physics}')

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(dpi=80)
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 180)
v = np.linspace(0, np.pi, 180)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

#Plot the surface
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.1)
elev = 20.0
rot = 30.0 / 180 * np.pi

#calculate vectors for "vertical" circle
a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
b = np.array([0, 1, 0])
b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed')
horiz_front = np.linspace(0, np.pi, 100)
ax.plot(np.sin(horiz_front),np.cos(horiz_front),0,color='k')

vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),
        color='k',lw=1)
ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), 
        a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),
        color='k',lw=1)

ax.quiver(0, 0, 0, 1.8, 0, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # x-axis
ax.quiver(0, 0, 0, 0, 1.5, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # y-axis
ax.quiver(0, 0, 0, 0, 0, 1.5, color='k', arrow_length_ratio=0.05, lw=1.5) # z-axis

ax.text(2.1, 0, -0.05, r'$\textbf{Obs}$',fontsize=12)
#ax.text(0, 0, 1.55, r"$\vb*{n}_{\textbf{orb}}$",fontsize=14)

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

psi = 45*np.pi/180
phi = np.linspace(-np.pi/2,np.pi/2,1000)
x = np.cos(phi)*np.sin(psi)
y = np.sin(phi)*np.sin(psi)
z = np.cos(psi)
ax.plot(x,y,z,c='grey')

psi = 45*np.pi/180
phi = np.linspace(np.pi/2,3*np.pi/2,1000)
x = np.cos(phi)*np.sin(psi)
y = np.sin(phi)*np.sin(psi)
z = np.cos(psi)
ax.plot(x,y,z,c='grey')

psi = 45*np.pi/180
phi = 60*np.pi/180
u = np.cos(phi)*np.sin(psi)
v = np.sin(phi)*np.sin(psi)
w = np.cos(psi)
ax.quiver(0,0,0,u,v,w,color='k', arrow_length_ratio=0.1, lw=1.5, normalize=True)

ax.quiver(0,0,0,u,v,0, color='k', arrow_length_ratio=0., lw=1.2, ls='--')
ax.quiver(u,v,0,0,0,w, color='k', arrow_length_ratio=0., lw=1.2, ls='--')

psi = 90*np.pi/180
phi = np.linspace(0,60*np.pi/180,100)
x = np.cos(phi)*np.sin(psi)*1/4
y = np.sin(phi)*np.sin(psi)*1/4
z = np.cos(psi)*1/4
ax.plot(x,y,z,c='k',lw=1.2)

psi = np.linspace(0,55*np.pi/180,100)
phi = 45*np.pi/180
r = np.linspace(1/6,1/3,100)
x = np.cos(phi)*np.sin(psi)*r
y = np.sin(phi)*np.sin(psi)*r
z = np.cos(psi)*r
ax.plot(x,y,z,c='k',lw=1.2)

# ax.text(0.9,1., 0.83, r"$\vb*{n_{\star}}$",fontsize=14)
# ax.text(-1.25,-0.8,-0.78, r"$\vb*{\theta}$",fontsize=14)
# ax.text(1.2,0.72,0.71, r"$\vb*{\psi}$",fontsize=14)

# Set an equal aspect ratio
ax.set_aspect('equal')

ax.view_init(elev=20., azim=30, roll=0)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(paths.figures / "coord_psi.pdf", bbox_inches="tight", dpi=600)

plt.close()



fig = plt.figure(dpi=80)
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 180)
v = np.linspace(0, np.pi, 180)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

#Plot the surface
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.1)

elev = 10
rot = 65 / 180 * np.pi

#calculate vectors for "vertical" circle
a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
b = np.array([0, 1, 0])
b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))

vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),
        color='k',lw=1)

ax.quiver(0, 0, 0, 1.4, 0, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # x-axis
ax.quiver(0, 0, 0, 0, 1.8, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # y-axis
ax.quiver(0, 0, 0, 0, 0, 1.5, color='k', arrow_length_ratio=0.05, lw=1.5) # z-axis

ax.text(1.7, 0, -0.05, r'$\textbf{Obs}$',fontsize=12)
#ax.text(0, 0, 1.55, r"$\vb*{n}_{\textbf{orb}}$",fontsize=14)

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

i = 40*np.pi/180
lam = 40*np.pi/180
u = np.cos(i)
v = np.sin(i)*np.sin(lam)
w = np.sin(i)*np.cos(lam)

ax.quiver(0,0,0,u,v,w,color='k', arrow_length_ratio=0.1, lw=1.5, normalize=True)

i = np.arccos(u)
lam = np.linspace(-np.pi,np.pi,1000)
x = np.cos(i)*np.ones(len(lam))
y = np.sin(i)*np.sin(lam)
z = np.sin(i)*np.cos(lam)
ax.plot(x,y,z,c='grey')

i = np.pi/2
lam = np.linspace(0,np.pi,1000)
x = np.cos(i)*np.ones(len(lam))
y = np.sin(i)*np.sin(lam)
z = np.sin(i)*np.cos(lam)
ax.plot(x,y,z,c='k')

i = np.pi/2
lam = np.linspace(-np.pi,0,1000)
x = np.cos(i)*np.ones(len(lam))
y = np.sin(i)*np.sin(lam)
z = np.sin(i)*np.cos(lam)
ax.plot(x,y,z,c='k',ls='--')

ax.quiver(0,0,0,0,v,w, color='k', arrow_length_ratio=0., lw=1.2, ls='--')
ax.quiver(0,v,w,u,0,0, color='k', arrow_length_ratio=0., lw=1.2, ls='--')

i = np.linspace(0,40*np.pi/180,100)
lam = 40*np.pi/180
r = np.linspace(1/6,1/4,100)
x = np.cos(i)*r
y = np.sin(i)*np.sin(lam)*r
z = np.sin(i)*np.cos(lam)*r
ax.plot(x,y,z,c='k',lw=1.2)


i = np.pi/2
lam = np.linspace(0,40*np.pi/180,100)
r = np.linspace(1/5,2/7,100)
x = np.cos(i)*r
y = np.sin(i)*np.sin(lam)*r
z = np.sin(i)*np.cos(lam)*r
ax.plot(x,y,z,c='k',lw=1.2)

# ax.text(0.85, 0.1, 0.35, r"$\vb*{n_{\star}}$",fontsize=14)
# ax.text(0,0.01,0.25, r"$\vb*{\lambda}$",fontsize=14)
# ax.text(0.35,0,0.05, r"$\vb*{i_\star}$",fontsize=14)

# Set an equal aspect ratio
ax.set_aspect('equal')

ax.view_init(elev=10, azim=65, roll=0)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(paths.figures / "coord_lam.pdf", bbox_inches="tight", dpi=600)

plt.close()

