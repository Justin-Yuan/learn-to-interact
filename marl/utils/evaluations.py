import numpy as np 


#####################################################################################
### communication evaluations
#####################################################################################





#####################################################################################
### mutual information  
#####################################################################################

def estimate_mi(rollouts):
    """ 
    """
    return 


def to_int(n):
    # Converts various things to integers
    if type(n) is int:
        return n
    elif type(n) is float:
        return int(n)
    else:
        return int(n.data.numpy())


def probs_from_counts(l, ldim, eps=0):
    # Outputs a probability distribution (list) of length ldim, by counting event occurrences in l
    l_c = [eps] * ldim
    for i in l:
        l_c[i] += 1. / len(l)
    return l_c


""" Calculating statistics about comms and acts """


def calc_stats(comms, acts, n_comm, n_acts, stats):
    # Produces a matrix ('stats') that counts co-occurrences of messages and actions
    # Can update an existing 'stats' matrix (=None if there is none)
    # Calls bin_acts to do the heavy lifting
    comms = [to_int(m) for m in comms]
    acts = [to_int(a) for a in acts]
    stats = bin_acts(comms, acts, n_comm, n_acts, stats)
    return stats


def bin_acts(comms, acts, n_comm, n_acts, b=None):
    # Binning function that creates a matrix that counts co-occurrences of messages and actions
    if b is None:
        b = np.zeros((n_comm, n_acts))
    for a, c in zip(acts, comms):
        b[c][a] += 1
    return b


def calc_mutinfo(acts, comms, n_acts, n_comm):
    # Calculate mutual information between actions and messages
    # Joint probability p(a, c) is calculated by counting co-occurences, *not* by performing interventions
    # If the actions and messages come from the same agent, then this is the speaker consistency (SC)
    # If the actions and messages come from different agents, this is the instantaneous coordinatino (IC)
    comms = [U.to_int(m) for m in comms]
    acts = [U.to_int(a) for a in acts]

    # Calculate probabilities by counting co-occurrences
    p_a = U.probs_from_counts(acts, n_acts)
    p_c = U.probs_from_counts(comms, n_comm)
    p_ac = U.bin_acts(comms, acts, n_comm, n_acts)
    p_ac /= np.sum(p_ac)  # normalize counts into a probability distribution

    # Calculate mutual information
    mutinfo = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                mutinfo += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return mutinfo


def calc_entropy(comms, n_comm):
    # Calculates the entropy of the communication distribution
    # p(c) is calculated by averaging over episodes
    comms = [U.to_int(m) for m in comms]
    eps = 1e-9

    p_c = U.probs_from_counts(comms, n_comm, eps=eps)
    entropy = 0
    for c in range(n_comm):
        entropy += - p_c[c] * math.log(p_c[c])
    return entropy



def calc_context_indep(acts, comms, n_acts, n_comm):
    # Calculates the context independence (Bogin et al., 2018)
    comms = [U.to_int(m) for m in comms]
    acts = [U.to_int(a) for a in acts]
    eps = 1e-9

    p_a = U.probs_from_counts(acts, n_acts, eps=eps)
    p_c = U.probs_from_counts(comms, n_comm, eps=eps)
    p_ac = U.bin_acts(comms, acts, n_comm, n_acts)
    p_ac /= np.sum(p_ac)

    p_a_c = np.divide(p_ac, np.reshape(p_c, (-1, 1)))
    p_c_a = np.divide(p_ac, np.reshape(p_a, (1, -1)))

    ca = np.argmax(p_a_c, axis=0)
    ci = 0
    for a in range(n_acts):
        ci += p_a_c[ca[a]][a] * p_c_a[ca[a]][a]
    ci /= n_acts
    return ci
