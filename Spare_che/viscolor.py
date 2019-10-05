import colorvector as cv
import matplotlib.pyplot as plt
import numpy as np
def plotplane(timg,rgbvec,mask,filename):
    #XYZからxyへ変換
    ti=timg.reshape(np.shape(timg)[0]*np.shape(timg)[1],3)
    xy,ti=cv.convimg2rgb(ti)

    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(121,aspect=1.0)
    ax.scatter(xy[:,0], xy[:,1],facecolors=ti,alpha=1,s=2)
    ax.set_xlim(0.15,0.62)
    ax.set_ylim(0.2,0.62)
    ax.set_title("Original")
    #plt.legend() 

    xy,yi=cv.convimg2rgb(rgbvec[mask,:]) #rgbvec[mask,:]
    yi[yi>1]=1.0
    yi[yi<0]=0

    xyc,yic=cv.convimg2rgb(rgbvec)
    yic[yic>1]=1.0
    yic[yic<0]=0
    
    ax=fig.add_subplot(122,aspect=1.0)
    ax.scatter(xy[:,0], xy[:,1],facecolors=yi,alpha=1,s=20,label="Selected")
    ax.scatter(xyc[:,0], xyc[:,1],facecolors=yic,alpha=1,s=1,label="Pallet color")
    plt.legend()
    ax.set_xlim(0.15,0.62)
    ax.set_ylim(0.2,0.62)
    ax.set_title("reconstruct")
#    display(standalone=True)
    plt.savefig(filename)
    
