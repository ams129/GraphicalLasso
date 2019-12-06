import numpy as np
import copy

#block
def block(X, t):
    xij = X[t,t]
    x = X[:, t]
    x = np.delete(x, t)
    X = np.delete(X, t, axis = 0)
    X = np.delete(X, t, axis = 1)
    return X, x, xij

#update
def update(X, x, xij, t):
    x = np.insert(x, t, xij)
    X[:,t] = x
    X[t,:] = x
    return X

def solve(data, rho):
    #parameters
    size = data.shape[1]
    iters = 0
    max_iters = 100
    thr = 1e-4

    #initialize
    S = np.cov(data.T)
    Theta_inv = copy.deepcopy(S)
    Theta = np.linalg.inv(S)

    #start iteration
    while iters < max_iters:
        Theta_pre = copy.deepcopy(Theta)
        iters = iters + 1

        for i in range(size):
            #block
            L, l, lmd = block(Theta, i)
            W, w, sigma = block(Theta_inv, i)
            S_, s, sii = block(S, i)

            #optimezing
            sigma = sii + rho
            beta =  np.dot(np.linalg.inv(W), w)
            alpha = [0] * (size - 1)
            for j in range(len(alpha)):
                alpha[j] = s[j] - (np.dot(W[j, :], beta) - W[j,j]* beta[j])
            for j in range(len(beta)):
                if alpha[j] > rho:
                    beta[j] = (alpha[j] - rho) / W[j,j]
                elif alpha[j] < -rho:
                    beta[j] = (alpha[j] + rho) / W[j,j]
                else:
                    beta[j] = 0
            lmd = 1 / (sigma - np.dot(np.dot(beta.T, W), beta))
            l = beta * (-1 / (sigma - np.dot(np.dot(beta.T, W), beta)))
            w = np.dot(W, l) * (-1 / lmd)

            #update theta and theta_inv
            Theta = update(Theta, l, lmd, i)
            Theta_inv = update(Theta_inv, w, sigma, i)

        #check stop
        if np.linalg.norm(Theta - Theta_pre) / (size ** 2) < thr:
            break

    #print iterations number
    if iters < max_iters:
        print("iters:", iters)
    else:
        print("max iters:", max_iters)

    #scaling
    scale = 0
    for i in range(size):
        scale = scale + Theta[i,i]
    Theta = Theta * (1 / (scale / size))

    return Theta
