from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import torch
import torch.utils.data as data

# all the labels: ndarray (1280, 1), dtype: np.uint8
# videos: ndarray (1280, 63, 32, 32, 3), dtype: np.float64
dataset_root = '/home/liusheng/Dataset/EEG/'
# dataset_root = 'C:/Study/Data/'
arousal_name = 'arousal_labels.npy'
valience_name = 'valience_labels.npy'
four_cls_name = 'four_cls_labels.npy'
video_name = 'videos.npy'

seed = 1


class EEGDataset(data.Dataset):
    """DEAP dataset"""
    valid_split = {'train': 0.6, 'val': 0.2, 'test': 0.2}

    def __init__(self, split='valid_split'):
        super(EEGDataset, self).__init__()

        if split not in self.valid_split.keys():
            raise ValueError('Unknown split {:s}'.format(split))
        if not os.path.exists(dataset_root):
            raise ValueError('{:s} does not exist'.format(dataset_root))

        self.arousal_lbs = np.load(os.path.join(dataset_root, arousal_name))
        self.valience_lbs = np.load(os.path.join(dataset_root, valience_name))
        self.four_cls_lbs = np.load(os.path.join(dataset_root, four_cls_name))
        self.videos = np.load(os.path.join(dataset_root, video_name))

        self.total_sample = self.videos.shape[0]
        self.num_sample = int(self.total_sample * self.valid_split[split])

        np.random.seed(seed)
        self.indices = np.random.choice(self.total_sample, size=self.num_sample)

    def __getitem__(self, index):
        ix = self.indices[index]
        ar_lb = int(self.arousal_lbs[ix])
        vl_lb = int(self.valience_lbs[ix])
        fcls_lb = int(self.four_cls_lbs[ix])
        video = self.videos[ix, :, :, :, :]   # (63, 32, 32, 3)
        # (D, H, W, C) -> (C, D, H, W): (3, 63, 32, 32)
        video = np.transpose(video, (3, 0, 1, 2))
        video = video[np.newaxis, :]
        return ar_lb, vl_lb, fcls_lb, video

    def __len__(self):
        return self.num_sample

    def collate_fn(self, batch):
        arousal_lbs = [b[0] for b in batch]
        valience_lbs = [b[1] for b in batch]
        four_cls_lbs = [b[2] for b in batch]

        arousal_lbs = torch.LongTensor(arousal_lbs)
        valience_lbs = torch.LongTensor(valience_lbs)
        four_cls_lbs = torch.LongTensor(four_cls_lbs)

        # (B, C, D, H, W): (1, 3, 63, 32, 32)
        video = [torch.FloatTensor(b[3]) for b in batch]
        video = torch.cat(video, dim=0)
        return arousal_lbs, valience_lbs, four_cls_lbs, video


if __name__ == '__main__':
    dataset = EEGDataset('train')
    loader = data.DataLoader(dataset,
                             batch_size=6,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=dataset.collate_fn)
    sample = next(iter(loader))
    print sample