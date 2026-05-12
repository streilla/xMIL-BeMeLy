import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py

# Custom dataset
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512, shuffle=False, max_bag_size=None, min_bag_size=None, seed=1234):
        self.df = df[df["fold_0"] == split]
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        max_bag_size = torch.inf if (max_bag_size is None or max_bag_size < 0) else max_bag_size
        min_bag_size = 0 if (min_bag_size is None or min_bag_size < 0) else min_bag_size
        bag_sizes = self.df['n_patches'].to_list()
        slide_ids = self.df['slide_id'].to_list()
        keep_slides = [idx for idx, k in zip(slide_ids, bag_sizes) if min_bag_size <= k <= max_bag_size]
        print(f"Dropping {self.df.shape[0] - len(keep_slides)} slides with more than {max_bag_size} "
                f"or fewer than {min_bag_size} patches - {split}.")
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #bag_size = None
        with h5py.File(os.path.join(self.feats_path, row['slide_id'] + '.h5'), "r") as f:
            if self.shuffle:
                rng = np.random.default_rng(seed=hash(row['slide_id']) % 2**32)
                features = f["features"][:]
                coords = f["coords"][:]
                perm = rng.permutation(len(features))
                features = torch.from_numpy(features[perm])
                coords = torch.from_numpy(coords[perm])
            else:  
                features = torch.from_numpy(f["features"][:])
                coords = torch.from_numpy(f["coords"][:])

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(self.seed))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(self.seed))  # Oversampling
            features = features[indices]
            coords = coords[indices]
            
        else:
            indices = torch.tensor(range(features.shape[0]))

        label = torch.tensor(row["label"], dtype=torch.long).unsqueeze(dim=-1)
        slide = row['slide_id']
        batch = {'features': features, 'targets': label, 'bag_size': features.shape[0], 'patch_ids': indices, 'sample_ids': {'slide_id': slide}, 'patch_coords': coords}
        return batch
    
def bag_collate_fn(batch_list):
        """
        Custom collate function for this dataset.
        """
        col_batch = {}
        for key in batch_list[0].keys():
            if key in ['features', 'patch_ids', 'patch_coords']:
                col_batch[key] = torch.concat([batch[key] for batch in batch_list])
            elif key == 'targets':
                col_batch[key] = torch.stack([batch[key] for batch in batch_list])
            elif key == 'sample_ids':
                col_batch[key] = {col: [batch[key][col] for batch in batch_list] for col in batch_list[0][key]}
            else:
                col_batch[key] = torch.tensor([batch[key] for batch in batch_list])
        return col_batch