import numpy as np
import h5py
import os
import torch
import torch.utils.data as data

DATASET = '/home/liusheng/Dataset/EEG/'


class EEGImage(data.Dataset):
    """DEAP EEG dataset"""
    VALID_SPLIT = ('train', 'val', 'test')

    def __init__(self, data_dir, split='train'):
        super(EEGImage, self).__init__()

        if split not in self.VALID_SPLIT:
            raise ValueError('Unknown split {:s}'.format(split))
        if not os.path.exists(data_dir):
            raise ValueError('{:s} does not exist'.format(data_dir))

        self.split = split
        self.data = h5py.File(data_dir, 'r')
        self.key = list(self.data.keys())

    def __getitem__(self, index):
        key = self.key[index]
        label = int(self.data[key]['label'][...])
        video = self.data[key]['video'][...]
        # (D, H, W, C) -> (C, D, H, W): (3, 63, 32, 32)
        video = np.transpose(video, (3, 0, 1, 2))
        return label, video

    def __len__(self):
        return len(self.key)

    def collate_fn(self, batch):
        label = [b[0] for b in batch]
        # (C, D, H, W): (3, 63, 32, 32)
        video = [b[1] / 255.0 for b in batch]
        video = [torch.FloatTensor(v).unsqueeze(0) for v in video]
        return torch.LongTensor(label), torch.cat(video, dim=0)


if __name__ == '__main__':
    dataset = EEGImage("/home/liusheng/Dataset/EEG/train.h5", 'train')
    print len(dataset)
    label, video = dataset.__getitem__(0)
    print label
    print video.shape

    loader = data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    sample = next(iter(loader))
    print sample[0]
    print sample[1]

