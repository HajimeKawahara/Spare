import numpy as np
import colour 

def generate_palette(Npal):

    a=np.linspace(0.01,0.99,Npal)
    aa,bb=np.meshgrid(a,a)
    xy=np.array([aa.flatten(),bb.flatten()]).T
    
    vec=colour.xy_to_XYZ(xy)    
    vec=(vec.T/np.linalg.norm(vec,axis=1)).T
    illuminant_XYZ = np.array([0.34570, 0.35850])
    illuminant_RGB = np.array([0.31270, 0.32900])
    chromatic_adaptation_transform = 'Bradford'
    XYZ_to_RGB_matrix = np.array([[3.24062548, -1.53720797, -0.49862860],[-0.96893071, 1.87575606, 0.04151752],[0.05571012, -0.20402105, 1.05699594]])
    rgb=colour.XYZ_to_RGB(z, illuminant_XYZ, illuminant_RGB, XYZ_to_RGB_matrix,chromatic_adaptation_transform)
    
    return rgb
    

def color_weight_function():
    return
