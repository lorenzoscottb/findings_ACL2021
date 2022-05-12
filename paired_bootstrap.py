# author = "lorenzoscottb"
# adapted from bootsa py library

import numpy as np
from tqdm import tqdm

def paird_bootstrap(targs, h0_preds, h1_preds, metric, num_rounds=1000, sample_size=.1, 
                    alpha=0.05):
    
    """
    Nonparametric two-tailed bootstrap test, comparing metric scores

    Parameters
    -------------
    targs :     list or numpy array with shape (n_datapoints,)
                A list or 1D numpy array of the first sample
                gold standard/dataset items   
    
    h0_preds :  list or numpy array with shape (n_datapoints,)
                A list or 1D numpy array of the first sample
                Output of model A

    h1_preds :  list or numpy array with shape (n_datapoints,)
                A list or 1D numpy array of the first sample
                Output of model B
        
    metric: function, metric to use as evaluation for the models
    
    num_rounds : int (default: 1000), number of permutation samples.
                 set to 0 to have it estimanted by n of test itmes
    
    sample_size : n. of itmes to subsample
    
    alpha : alpha threshold
     
    Returns
    ----------
    p-value under the null hypothesis
                                       
    """ 
    
    assert sample_size == 1 or (.05 <= sample_size <= .5), 'sample_size must be between .05 and .5'
    
    # number of permutation based on items and alpha
    mx_R       = int(max(10000, int(len(targs) * (1 / float(alpha))))) 
    num_rounds = num_rounds if num_rounds > 0 else mx_R
    
    overall_size = len(targs)
    sample_size  = int(len(targs) * sample_size)
    
    targs    = np.array(targs)
    h0_preds = np.array(h0_preds)
    h1_preds = np.array(h1_preds)
    
    original_diff = np.abs(metric(h1_preds, targs) - metric(h0_preds, targs))

    twice_diff  = 0
    
    for _ in range(num_rounds):
        if sample_size < 1.: # use boostsa subsample withot replacement strategy
            i_sample = np.random.choice(range(overall_size), size=sample_size, replace=False)
            sample_h0_preds = h0_preds[i_sample]
            sample_h1_preds = h1_preds[i_sample]
            sample_targs    = targs[i_sample]        
        
        else: # use full-sample with replacement strategy 
            i_sample        = np.random.randint(0, overall_size, overall_size)
            sample_h0_preds = [h0_preds[i] for i in i_sample]
            sample_h1_preds = [h1_preds[i] for i in i_sample]
            sample_targs    = [targs[i] for i in i_sample]
                
        sample_diff = np.abs(metric(sample_h1_preds, sample_targs) - metric(sample_h0_preds, sample_targs))
        
        if sample_diff >= 2*original_diff: 
            twice_diff += 1

    p_val = twice_diff / num_rounds

    return p_val
  
