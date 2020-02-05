import torch
import numpy as np
from types import SimpleNamespace as SN


#####################################################################################
### batch container for transitions  
#####################################################################################

class SampleBatch(object):
    def __init__(self, scheme, batch_size, data=None, device="cpu"):
        """ 
        each field in transition_data is (B,D)
        Arguments:
            - scheme (dict): specifies fields (name, shape, type) to record
            - data (SampleBatch): (partially) built samples 
        """
        self.scheme = scheme.copy()
        self.batch_size = batch_size
        self.device = device

        if data is not None:
            self.data = data 
        else:
            self.data = {}
            self._setup_data(self.scheme, batch_size)

    def __repr__(self):
        return "SampleBatch. Batch Size:{}, Keys:{}".format(
            self.batch_size, self.scheme.keys())

    def _setup_data(self, scheme, batch_size, max_seq_length):
        """ initialize sample containers 
        """
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)
            # initialize container 
            self.data[field_key] = torch.zeros((batch_size, *vshape), 
                                    dtype=dtype, device=self.device)
    
    def extend(self, scheme):
        """ update scheme with new initialized containers, return new EpisodeBatch """
        self._setup_data(scheme, self.batch_size)

    def to(self, device):
        """ transfer data to device """
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device

    def concat(self, samples):
        """ concatenate with another (list) SampleBatch, return new SampleBatch 
        reference: https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py
        Arguments: 
            samples: SampleBatch or list of SampleBatch
        Returns:
            ret: data concatenated SampleBatch 
        """
        if len(samples) < 1:
            return self 
        if not all([isinstance(s, EpisodeBatch) for s in samples]):
            raise Exception("Require all EpisodeBatch to concatenate!")
        
        # check matching 
        if not all([self.scheme == s.scheme for s in samples[1:]]):
            raise Exception("Samples scheme do not match!")
        if not all([self.device == s.device for s in samples[1:]]):
            raise Exception("Samples device do not match!")
        samples = [self] + samples
        
        new_data = {}
        new_batch_size = sum([s.batch_size for s in samples])

        for field_key, field_info in self.scheme.items():
            new_data[field_key] = torch.cat([
                s.data[field_key] for s in samples
            ], 0)
        ret = SampleBatch(self.scheme, new_batch_size, data=new_data, device=self.device)
        return ret  

    def _parse_slices(self, items):
        """ get valid slices for batch, each slice is a:b or None
            Convert single indice to slice
        """
        return slice(item, item+1) if isinstance(item, int) else item 

    def update(self, data, bs=slice(None)):
        """ build up episode batch incrementally 
        """
        slices = self._parse_slices(bs)
        # update each field
        for k, v in data.items():
            if not k in self.data:
                raise KeyError("{} not found in transition or episode data".format(k))
            dtype = self.scheme[k].get("dtype", torch.float32)
            v = torch.tensor(v, dtype=dtype, device=self.device)
            self.data[k][bs] = v.view_as(self.data[k][bs])

    def __getitem__(self, item):
        """ collect data according to item and pack into EpisodeBatch or torch array
        """
        if isinstance(item, str):
            # if request only single field --> return torch tensor directly
            if item in self.data:
                return self.data[item]
            else:
                raise ValueError

        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            # request list of fields -> pack to SampleBatch
            new_data = {}
            for key in item:
                if key in self.data:
                    new_data[key] = self.data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            ret = SampleBatch(new_scheme, self.batch_size, data=new_data, device=self.device)
            return ret

        else:
            # request a slice (batch_slice,) -> pack to SampleBatch
            item = self._parse_slices(item)
            new_data = {}
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            ret_bs = self._get_num_items(item, self.batch_size)
            ret = SampleBatch(self.scheme, ret_bs, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]



#####################################################################################
### batch container for episodes  
#####################################################################################

class EpisodeBatch(object):
    """ container for samples, either as transitions or episodes
    reference: https://github.com/oxwhirl/pymarl/blob/master/src/components/episode_buffer.py
    """
    def __init__(self, scheme, batch_size, max_seq_length, data=None, 
                device="cpu"):
        """ 
        each field in transition_data is (B,T,D)
        Arguments:
            - scheme (dict): specifies fields (name, shape, type) to record
            - data (EpisodeBatch): (partially) built samples 
        """
        self.scheme = scheme.copy()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device

        if data is not None:
            self.data = data 
        else:
            self.data = self._new_data_sn()
            self._setup_data(self.scheme, batch_size, max_seq_length)

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{}".format(
            self.batch_size, self.max_seq_length, self.scheme.keys())

    def _new_data_sn(self):
        """ empty namespace data container """
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _setup_data(self, scheme, batch_size, max_seq_length):
        """ initialize sample containers 
        """
        # for rnn truncation
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": torch.long},
        })
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", torch.float32)
            episodic = field_info.get("episodic", False)

            if isinstance(vshape, int):
                vshape = (vshape,)
            if episodic:
                # data pertained to entire episode, i.e. win/lose of game 
                target = self.data.episode_data
            else:
                target = self.data.transition_data
                vshape = (max_seq_length,) + vshape
            # initialize container 
            target[field_key] = torch.zeros((batch_size, *vshape), 
                                    dtype=dtype, device=self.device)
    
    def extend(self, scheme):
        """ update scheme with new initialized containers, return new EpisodeBatch """
        self._setup_data(scheme, self.batch_size, self.max_seq_length)

    def concat(self, samples):
        """ concatenate with another (list) EpisodeBatch, return new EpisodeBatch 
        reference: https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py
        Arguments: 
            samples: EpisodeBatch or list of EpisodeBatch
        Returns:
            ret: data concatenated EpisodeBatch 
        """
        if len(samples) < 1:
            return self 
        if not all([isinstance(s, EpisodeBatch) for s in samples]):
            raise Exception("Require all EpisodeBatch to concatenate!")
        
        # check matching 
        if not all([self.scheme == s.scheme for s in samples[1:]]):
            raise Exception("Samples scheme do not match!")
        if not all([self.max_seq_length == s.max_seq_length for s in samples[1:]]):
            raise Exception("Samples max_seq_length do not match!")
        if not all([self.device == s.device for s in samples[1:]]):
            raise Exception("Samples device do not match!")
        samples = [self] + samples
        
        new_data = self._new_data_sn()
        new_batch_size = sum([s.batch_size for s in samples])

        for field_key, field_info in self.scheme.items():
            episodic = field_info.get("episodic", False)
            target = lambda x: x.episode_data if episodic else x.transition_data
            target(new_data)[field_key] = torch.cat([
                target(s.data)[field_key] for s in samples
            ], 0)

        ret = EpisodeBatch(self.scheme, new_batch_size, 
                self.max_seq_length, data=new_data, device=self.device)
        return ret 

    def _parse_slices(self, items):
        """ get valid slices for episode batch update 
        slice  is tuple of (batch_slice, time_slice), each slice is a:b or None
        """
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, torch.LongTensor, torch.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        parsed = []
        for item in items:
            if isinstance(item, int):
                # Convert single indice to slice
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def to(self, device):
        """ transfer data to device """
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def max_t_filled(self):
        """ filled (B,T) -> get filled length from each batch -> take max """
        return torch.sum(self.data.transition_data["filled"], 1).max(0)[0]
    
    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        """ build up episode batch incrementally 
        """
        slices = self._parse_slices((bs, ts))
        # update each field
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", torch.float32)
            v = torch.tensor(v, dtype=dtype, device=self.device)
            # allow smart reshaping i.e. when data has no time dim or 1
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        """ collect data according to item and pack into EpisodeBatch or torch array
        """
        if isinstance(item, str):
            # if request only single field --> return torch tensor directly
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError

        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            # request list of fields -> pack to EpisodeBatch
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            ret = EpisodeBatch(new_scheme, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret

        else:
            # request a slice (batch_slice, time_slice) -> pack to EpisodeBatch
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]




    





