import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import minimize
import pickle as pkl


def query_preprocess(query):

    x_vals = query[:,0]
    y_vals = query[:,1]

    x_peaks = x_vals[y_vals>=tau]

    return y_vals, x_vals, x_peaks
    
def cos_sim(a,b): return dot(a, b)/(norm(a)*norm(b))

def sum_sq_diff(a,b): return np.sum(np.square(a/np.sum(a) - b/np.sum(b)))

def estimate(x, mu, sigma):

    out = np.zeros(len(x))
    for j in range(n_shifts):
        out = out + np.exp(-np.abs(x - mu[j] - shifts_aligned[j])/sigma[j])
        
    return out

def get_spect(var, x_vals):

    mu = var[:n_shifts]
    sigma = np.exp(var[n_shifts:])
    
    return estimate(x_vals, mu, sigma)

def opt_fun(var):

    mu = var[:n_shifts]
    sigma = np.exp(var[n_shifts:])

    estimated_y_vals = estimate(query_x_vals, mu, sigma)
    
    obj = - cos_sim(query_y_vals, estimated_y_vals)\
          + sum_sq_diff(query_y_vals, estimated_y_vals)\
          + np.sum( np.square(mu) )\
          + np.sum( np.square(sigma) )\
          + np.sum( np.clip(eps + mu[:-1] + shifts_aligned[:-1] - mu[1:] - shifts_aligned[1:], 0, 100) ** 2 )
  
    return obj

   
## hyperparameter settings
tau = 0.05
h = 1
eps = 0.01
thr = 10
alpha = 0.05


## sample data import
with open('sample_data.pickle', 'rb') as f:
    (query, pool) = pkl.load(f)

query_y_vals, query_x_vals, query_x_peaks = query_preprocess(query)


## molecule search
results = {}  
for mol_id in pool.keys():

    shifts = np.sort(pool[mol_id])
    n_shifts = len(shifts)
   
    ## align
    shifts_aligned = np.array([query_x_peaks[np.argmin(np.abs(query_x_peaks - s))] for s in shifts])
    shifts_aligned[0] = np.min(query_x_peaks)
    shifts_aligned[-1] = np.max(query_x_peaks)
    
    ## optimize
    if np.max(np.abs(shifts_aligned - shifts)) > thr or np.max([np.min(np.abs(shifts_aligned - x)) for x in query_x_peaks]) > thr:
    
        continue

    else:
    
        x0 = np.concatenate([np.zeros(n_shifts), np.log(h)*np.ones(n_shifts)], 0)
        opt_res = minimize(opt_fun, x0, method='L-BFGS-B')
        shifts_optimized = opt_res.x[:n_shifts] + shifts_aligned
        estimated_y_vals = get_spect(opt_res.x, query_x_vals)
        
        score = cos_sim(query_y_vals, estimated_y_vals) - alpha * np.sqrt(np.sum(np.square(shifts_optimized - shifts)))

        results[mol_id] = score

results = dict(sorted(results.items(), reverse=True, key=lambda item: item[1]))
print(results)