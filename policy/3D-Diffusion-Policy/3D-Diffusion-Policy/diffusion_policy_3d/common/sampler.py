from typing import Optional
import numpy as np
import numba
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
#创建索引
#返回： indices索引列表  形状 二维度【episode_num X (episode长度+开头填充+结尾填充-horizon+1)】 X 4
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    #先循环剧集数
    for i in range(len(episode_ends)):
        #如果mask为false，跳过
        if not episode_mask[i]:
            # skip episode
            continue
        #设置开始的所以为0
        start_idx = 0
        #如果i>0,后续剧集的开始索引从episode_end[i-1] 开始
        if i > 0:
            start_idx = episode_ends[i-1]
        #结束的索引 为episode_end[i]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        #设置最小和最大起始位置
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        #循环
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer,     #数据集缓冲
        sequence_length:int,             #序列长度(horizon)
        pad_before:int=0,                #前面填充
        pad_after:int=0,                 #后面填充
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,   #剧集mask
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        #如果没有指定key，则使用replay_buffer中的所有key
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        #如果没有mask，则生成全为1的mask
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        #np.any 用于检查数族中是至少存在一个true元素
        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
      
        #获取对应编号的索引
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        #创建结果字典
        result = dict()
        #循环key
        for key in self.keys:
            #取对应key的数据
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            #是否值加载前k个数据点
            if key not in self.key_first_k:
                #加载从buffer_start到buffer_end_idx之间的数据
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            #数据为采样到的数据
            data = sample
            #填充数据
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
