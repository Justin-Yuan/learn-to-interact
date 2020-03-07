""" set of action selectors given distrition (specified by action spaces)
and policy output parameters 
- support exploration, log prob and entropy evaluation
- reference: https://github.com/oxwhirl/pymarl/blob/master/src/components/action_selectors.py
"""
import numpy as np 
import torch 
import torch.distributions as D

""" 
can be used to get log_prob and entropy
- dist.log_prob(actions),   # (B,1),
- ist.entropy(),   # (B,1)

NOTE: can also make selectors with specific noisee types
e.g. epsilon-greedy
"""


class DiscreteActionSelector():
    def __init__(self):
        self.dist_fn = D.relaxed_categorical.RelaxedOneHotCategorical

    def select_action(self, logits, hard=True, temperature=1.0, 
        reparameterize=True, explore=True, **kwargs
    ):
        """ sample an action with evaluation 
        Arguments:
            - logits: (B,D)
            - hard: if output action is hard one-hot or soft logits
        Returns:
            - out: dict of action, log_prob, entropy and dist (for KL if needed)
        """
        # make distribution
        dist = self.dist_fn(torch.tensor([temperature]), logits=logits)
        if explore:
            if reparameterize:
                actions = dist.rsample()    # (B,D)
            else:
                actions = dist.sample()
        else:
            actions = logits
        # convert to one-hot (if needed)
        if hard:
            max_logit = actions.max(-1, keepdims=True)[0]
            onehot = torch.eq(actions, max_logit).float()
            actions = (one_hot - actions).detach() +  actions
        return actions, dist 
        

class ContinuousActionSelector():
    def __init__(self):
        self.dist_fn = D.multivariate_normal.MultivariateNormal
        self.std = 1e-4

    def select_action(self, mean, logstd=None, reparameterize=True,
        explore=True, **kwargs
    ):
        """ sample an action with evaluation 
        Arguments:
            - mean: (B,D)
            - logstd: (B,D,D)
        Returns:
            - out: dict of action, log_prob, entropy and dist (for KL if needed)
        """
        if logstd is None:
            covar = torch.eye() * self.std
        else:
            covar = torch.diag(torch.exp(logstd))
        # make distribution
        dist = self.dist_fn(mean, covar)

        if explore:
            if reparameterize:
                actions = dist.rsample()    # (B,D)
            else:
                actions = dist.sample()
        else:
            actions = mean 
        return actions, dist