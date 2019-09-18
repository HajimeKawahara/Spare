import matplotlib.pyplot as plt
import numpy as np
import cumodule
from cumodule import *
importlib.reload(cumodule)
import importlib
import readpng as rpng
import colour 
from colour.plotting import *
import itertools

img=rpng.get_img("./car.png")

fig=plt.figure()
timg=img[::8,::8,:]
plt.imshow(timg)
plt.savefig("carcar.png")

Nx, Ny, Nc = np.shape(timg)
N_data = 1000
rand_now = cumodule.random_generator(N_data, Nx, Ny)
dRGB, g=rand_now.make_colordata(timg,10) 

meanlc=np.mean(dRGB)
noise=np.random.normal(0,meanlc*0.01,np.shape(dRGB))
dRGB+=dRGB+noise

g=np.array(g)

print("start optimization...")
I_init=np.ones((Nx,Ny))
y = mfista_grouplasso(I_init, dRGB, g, lambda_gl = 1.e-1)

plt.imshow(y)
plt.savefig("carGroupLasso.png")

ti=timg.reshape(np.shape(timg)[0]*np.shape(timg)[1],3)
#XYZからxyへ変換
ti=timg.reshape(np.shape(timg)[0]*np.shape(timg)[1],3)
XYZ = colour.sRGB_to_XYZ(ti)
xy = colour.XYZ_to_xy(XYZ)

#CIE_1931_chromaticity_diagram_colours_plot(bounding_box=(-0.1, 0.9, -0.1, 0.9), standalone=False)
#plot_chromaticity_diagram_CIE1931(bounding_box=(0.15, 0.65, 0.15, 0.65), standalone=False)
#sRGB領域へプロット
#plt.plot(xy[:,0], xy[:,1], 'o', markersize=2, label="sRGB",color="gray",alpha=0.2)
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(121,aspect=1.0)
ax.scatter(xy[:,0], xy[:,1],facecolors=ti,alpha=1,s=2)
ax.set_xlim(0.15,0.62)
ax.set_ylim(0.2,0.62)
ax.set_title("Original")
#plt.legend() 

yi=y.reshape(np.shape(y)[0]*np.shape(y)[1],3)
XYZ = colour.sRGB_to_XYZ(yi)
xy = colour.XYZ_to_xy(XYZ)

print(np.shape(yi),np.shape(xy[:,0]),np.shape(xy[:,1]))
ax=fig.add_subplot(122,aspect=1.0)

print("MEAN,MEDIAN,MIN,MAX")
print(np.mean(yi),np.median(yi),np.max(yi),np.min(yi))
yi=yi/2.0
yim=np.copy(yi)
yim[yim>1.0]=1.0
yim[yim<0.0]=0.0

ax.scatter(xy[:,0], xy[:,1],facecolor=yim,alpha=1,s=2)
ax.set_xlim(0.15,0.62)
ax.set_ylim(0.2,0.62)
ax.set_title("reconstruct")
plt.savefig("color.png")
#display(standalone=True)

