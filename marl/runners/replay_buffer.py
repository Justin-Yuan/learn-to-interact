import torch
import numpy as np
from runners.sample_batch import SampleBatch, EpisodeBatch


#####################################################################################
### batch container for transitions  
#####################################################################################

class ReplayBuffer(SampleBatch):
    """ for mlp policy training, return batch of transitions (may not from same episode)
    """
    def __init__(self, scheme, buffer_size, device="cpu", prefill_num=1024):
        super(ReplayBuffer, self).__init__(scheme, buffer_size, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.samples_in_buffer = 0 
        self.prefill_num = prefill_num
    
    def __len__(self):
        return self.samples_in_buffer

    def insert_batch(self, batch):
        """ ep_batch is SampleBatch 
        """
        if self.buffer_index + batch.batch_size <= self.buffer_size:
            self.update(batch.data, slice(
                self.buffer_index, 
                self.buffer_index + batch.batch_size
            ))
            self.buffer_index = (self.buffer_index + batch.batch_size)
            self.samples_in_buffer = max(self.samples_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_batch(batch[0:buffer_left, :])
            self.insert_batch(batch[buffer_left:, :])
 
    def can_sample(self, batch_size):
        return self.samples_in_buffer >= self.prefill_num

    def sample(self, batch_size):
        """ return SampleBatch """
        assert self.can_sample(batch_size)
        if self.samples_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.samples_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def make_index(self, batch_size):
        """ return list of indices (length is batch size) to sample from """
        assert self.can_sample(batch_size)
        if self.samples_in_buffer == batch_size:
            return [i for i in range(batch_size)]
        else:
            return np.random.choice(self.samples_in_buffer, batch_size, replace=False)

    def sample_index(self, index, norm_rews=False):
        """ sample data given list of indices """
        return self[index]

    def make_latest_index(self, batch_size):
        """ return list of latest indices (length is batch size) to sample from """
        idx = [
            (self.buffer_index - 1 - i) % self.samples_in_buffer 
            for i in range(batch_size)
        ]
        np.random.shuffle(idx)
        return idx

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{}".format(
            self.samples_in_buffer, self.buffer_size, self.scheme.keys()
        )



#####################################################################################
### batch container for episodes  
#####################################################################################

class EpisodeReplayBuffer(EpisodeBatch):
    """ for recurrent policy training, return batch of episodes 
    """
    def __init__(self, scheme, buffer_size, max_seq_length, device="cpu", prefill_num=1024):
        super(EpisodeReplayBuffer, self).__init__(scheme, buffer_size, max_seq_length, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.prefill_num = prefill_num

    def __len__(self):
        return self.episodes_in_buffer

    def insert_episode_batch(self, ep_batch):
        """ ep_batch is EpisodeBatch 
        """
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= self.prefill_num #batch_size

    def sample(self, batch_size):
        """ return EpsideBatch """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def make_index(self, batch_size):
        """ return list of indices (length is batch size) to sample from """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return [i for i in range(batch_size)]
        else:
            return np.random.choice(self.episodes_in_buffer, batch_size, replace=False)

    def sample_index(self, index, norm_rews=False):
        """ sample data given list of indices """
        return self[index]

    def make_latest_index(self, batch_size):
        """ return list of latest indices (length is batch size) to sample from """
        idx = [
            (self.buffer_index - 1 - i) % self.episodes_in_buffer 
            for i in range(batch_size)
        ]
        np.random.shuffle(idx)
        return idx

    def __repr__(self):
        return "EpisodeReplayBuffer. {}/{} episodes. Keys:{}".format(
            self.episodes_in_buffer, self.buffer_size, self.scheme.keys()
        )
                    