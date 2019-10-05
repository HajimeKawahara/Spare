import matplotlib.pyplot as plt
import numpy as np
import readpng as rpng
from cumodule_glassoL2 import *
import viscolor as vc
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', nargs=1, required=True, help='log10 lambda_glasso',type=float)
    parser.add_argument('-t', nargs=1, required=True, help='log10 lambda_tikhonov',type=float)
    args = parser.parse_args()

    lambda_gl=10**(args.g[0])
    lambda_tik=10**(args.t[0])
    
    timg,rgbvec,I_init,gall,dRGB=rpng.toycar()
    yR = mfista_func(I_init, dRGB, gall, lambda_gl= lambda_gl, lambda_tik =lambda_tik, print_func=True)

    crit=1.e-2
    mask=np.sum(yR[:,:,:],axis=(0,1))>crit
    ysel=yR[:,:,mask]
    ypredrgb=np.einsum("cl,jkc->jkl",rgbvec[mask,:],ysel)
    print(np.shape(ysel)[2],"/",np.shape(yR)[2])
    plt.imshow(ypredrgb)
    plt.savefig("pred"+str(args.g[0])+"_"+str(args.t[0])+".png")

    vc.plotplane(timg,rgbvec,mask,"pal"+str(args.g[0])+"_"+str(args.t[0])+".png")
