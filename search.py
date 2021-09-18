import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import minimize


def matching_score(query, shifts, tau = 0.05, h = 1, eps = 0.01, thr = 10, alpha = 0.05):
    
    def query_preprocess(query):
    
        x_vals = query[:,0]
        y_vals = query[:,1]
    
        x_peaks = x_vals[y_vals>=tau]
    
        return y_vals, x_vals, x_peaks
        
    def cos_sim(a,b): return dot(a, b)/(norm(a)*norm(b))
    
    def sum_sq_diff(a,b): return np.sum(np.square(a/np.sum(a) - b/np.sum(b)))
    
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    
    def kernel_f(z, m):

        gaussian = np.exp(- 4 * np.log(2) * (z ** 2)) 
        lorentzian = 1 / (1 + 4 * (z ** 2))

        return (1-m) * gaussian + m * lorentzian
        
    def estimate(x, mu, sigma, m): return np.sum([kernel_f((x - mu[j] - shifts_aligned[j])/sigma[j], m[j]) for j in range(n_shifts)], 0)

    def opt_fun(var):
    
        mu = var[:n_shifts]
        sigma = np.exp(var[n_shifts:2*n_shifts])
        m = sigmoid(var[2*n_shifts:])

        estimated_y_vals = estimate(query_x_vals, mu, sigma, m)
        
        obj = - cos_sim(query_y_vals, estimated_y_vals)\
              + sum_sq_diff(query_y_vals, estimated_y_vals)\
              + np.sum( np.square(mu) )\
              + np.sum( np.square(sigma) )\
              + np.sum( np.clip(eps + mu[:-1] + shifts_aligned[:-1] - mu[1:] - shifts_aligned[1:], 0, 100) ** 2 )
      
        return obj
        
    def get_spect(var, x_vals): return estimate(x_vals, var[:n_shifts], np.exp(var[n_shifts:2*n_shifts]), sigmoid(var[2*n_shifts:]))


    query_y_vals, query_x_vals, query_x_peaks = query_preprocess(query)
    n_shifts = len(shifts)    

    ## align
    shifts_aligned = np.array([query_x_peaks[np.argmin(np.abs(query_x_peaks - s))] for s in shifts])
    shifts_aligned[0] = np.min(query_x_peaks)
    shifts_aligned[-1] = np.max(query_x_peaks)
    
    ## optimize
    if np.max(np.abs(shifts_aligned - shifts)) > thr or np.max([np.min(np.abs(shifts_aligned - x)) for x in query_x_peaks]) > thr:
    
        score = - 100

    else:
    
        x0 = np.concatenate([np.zeros(n_shifts), np.log(h)*np.ones(n_shifts), np.zeros(n_shifts)], 0)
        opt_res = minimize(opt_fun, x0, method='L-BFGS-B')
        shifts_optimized = opt_res.x[:n_shifts] + shifts_aligned
        estimated_y_vals = get_spect(opt_res.x, query_x_vals)
        
        score = cos_sim(query_y_vals, estimated_y_vals) - alpha * norm(shifts_optimized - shifts)

    return score



## sample data import
[query, pool] = np.load('./sample_data.npz', allow_pickle = True)['data']


## molecule search
results = {}  
for mol_id in pool.keys():

    shifts = np.sort(pool[mol_id])
    results[mol_id] = matching_score(query, shifts)   


results = dict(sorted(results.items(), reverse=True, key=lambda item: item[1])[:10])
print(results)
