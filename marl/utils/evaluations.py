import numpy as np 
import matplotlib.pyplot as plt 


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


#####################################################################################
### plotters
#####################################################################################

# Colorizor
# reference: https://github.com/openai/spinningup/blob/97c8c342c45e5bb51005a8515df23ba9c48f0782/spinup/utils/logx.py
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


# simple plotter 
# reference: https://github.com/openai/spinningup/blob/e76f3cc1dfbf94fe052a36082dbd724682f0e8fd/spinup/utils/plot.py
def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    import seaborn as sns 
    import pandas as pd 
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)
    # plt.legend(legends)
    # plt.xlabel('Episodes')
    # plt.ylabel('Mean reward')
    # dir_path = os.path.dirname(os.path.realpath(file_path))
    # plt.savefig(os.path.join(dir_path, 'mean_reward.png'))


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
    plt.show()





#####################################################################################
### main & tests
#####################################################################################

if __name__ == "__main__":
    pass 

