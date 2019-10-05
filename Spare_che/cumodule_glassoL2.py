import numpy as np
import time
import cupy as cp

# Derivative of ||y-Ax||^2 
##  cp.dot(A.T, data_dif) is n_image vecgtor, d_TSV(x_d) is the n_image vecgtor or matrix

##
#  i: time jk: 2Dmap
#  i -> il     l:lambda
#  jk -> jkc   c:component
#
##


def dF_dx(data, A, x_d, lambda_tik):
    data_dif = -(data - cp.einsum("ijkcl,jkc->il", A, x_d))
    return cp.einsum("ijkcl,il->jkc", A, data_dif) + 2.0*lambda_tik*x_d

def F_likeL2(data, A,  x_d, lambda_tik):
    data_dif = data -  cp.einsum("ijkcl,jkc->il", A, x_d)
    L2=lambda_tik*cp.einsum("jkc,jkc->", x_d, x_d)
    return (cp.einsum("il,il->",data_dif, data_dif)/2) + L2

### backtracking right term
def calc_Q_part(data, A,  x_d2, x_d, df_dx, L, lambda_tik):
    Q_core = F_likeL2(data, A, x_d, lambda_tik) ## f(y) 
    Q_core += cp.sum((x_d2 - x_d)*df_dx) + 0.5 * L * cp.sum( (x_d2 - x_d) * (x_d2 - x_d))
    return Q_core


## Calculation of soft_thresholding (prox)
#  \ nx, ny = np.shape(x_d)
# x_d : jkc

def soft_threshold_glasso(x_d, eta):
    vec = cp.zeros(np.shape(x_d))
    xgnorm=cp.linalg.norm(x_d,axis=(0,1))
    mask=xgnorm>eta
    vec[:,:,mask]= (1.0 - eta/xgnorm[mask])*x_d[:,:,mask]            
    return vec


## Function for MFISTA Group Lasso
def mfista_func(np_I_init, np_d, np_A_ten, lambda_gl,lambda_tik,L_init= 1e4, eta=1.1, maxiter= 10000, max_iter2=100, miniter = 100, TD = 30, eps = 1e-5, print_func = False):
    
    #convert to cupy
    I_init = cp.asarray(np_I_init)
    d = cp.asarray(np_d)
    A_ten = cp.asarray(np_A_ten)

    ## Initialization
    mu, mu_new = 1, 1
    y = I_init  # jkc
    x_prev = I_init
    cost_arr = []
    L = L_init
    
    ## The initial cost function
    cost_first = F_likeL2(d, A_ten, I_init, lambda_tik)
    cost_first += lambda_gl * cp.sum(cp.sqrt(cp.einsum("jkc,jkc->c",I_init,I_init)))
    cost_temp, cost_prev = cost_first, cost_first

    ## Main Loop until iter_now < maxiter
    ## PL_(y) & y are updated in each iteration
    p1=0.
    p2=0.
    p3=0.
    for iter_now in range(maxiter):
        s1=time.time()
        cost_arr.append(cost_temp)
        
        ##df_dx(y)
        df_dx_now = dF_dx(d,A_ten, y,lambda_tik) 
        
        ## Loop to estimate Lifshitz constant (L)
        ## L is the upper limit of df_dx_now
        s2=time.time()
        ## Backtracking
        for iter_now2 in range(max_iter2):
        
            y_now = soft_threshold_glasso(y - (1/L) * df_dx_now, lambda_gl/L)
            Q_now = calc_Q_part(d, A_ten, y_now, y, df_dx_now, L, lambda_tik)
            F_now = F_likeL2(d, A_ten, y_now, lambda_tik)
            
            ## If y_now gives better value, break the loop
            if F_now <Q_now:
                break
            L = L*eta

        L = L/eta #Here we get Lifshitz constant
        s3=time.time()

        #Nesterov acceleration
        mu_new = (1+cp.sqrt(1+4*mu*mu))/2
        F_now += lambda_gl * cp.sum(cp.sqrt(cp.einsum("jkc,jkc->c",y_now,y_now)))
        if print_func:
            if iter_now % 50 == 0:
                print ("Current iteration: %d/%d,  L: %f, cost: %f, cost_chiquare:%f" % (iter_now, maxiter, L, cost_temp, F_likeL2(d, A_ten, y_now, lambda_tik)))

        ## Updating y & x_k
        if F_now < cost_prev:
            cost_temp = F_now
            tmpa = (1-mu)/mu_new
            x_k = soft_threshold_glasso(y - (1/L) * df_dx_now, lambda_gl/L)
            y = x_k + ((mu-1)/mu_new) * (x_k - x_prev) 
            x_prev = x_k
            
        else:
            cost_temp = F_now
            tmpa = 1-(mu/mu_new)
            tmpa2 =(mu/mu_new)
            x_k = soft_threshold_glasso(y - (1/L) * df_dx_now, lambda_gl/L)
            y = tmpa2 * x_k + tmpa * x_prev       
            x_prev = x_k
            
        if(iter_now>miniter) and cp.abs(cost_arr[iter_now-TD]-cost_arr[iter_now])<cost_arr[iter_now]*eps:
            break

        mu = mu_new
        s4=time.time()
        p1+=s2-s1
        p2+=s3-s2
        p3+=s4-s3
    print(p1,p2,p3,"SEC in total")
    return cp.asnumpy(y)

