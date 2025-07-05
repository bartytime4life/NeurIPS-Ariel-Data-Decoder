import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class ArielV18_Dataset(Dataset):
    def __init__(self, root_dir, planet_ids, star_info_df=None, n_segments=8, is_train=False, mask_prob=0.1):
        self.root = root_dir
        self.planet_ids = planet_ids
        self.n_segments = n_segments
        self.is_train = is_train
        self.mask_prob = mask_prob
        self.star_info_df = star_info_df.set_index("planet_id") if star_info_df is not None else None

    def __len__(self):
        return len(self.planet_ids)

    def _load_map(self, path, shape):
        if not os.path.exists(path):
            return torch.zeros(self.n_segments, *shape)
        arr = pd.read_parquet(path).values.astype(np.float32).reshape(-1, *shape)
        indices = np.linspace(0, arr.shape[0] - 1, self.n_segments, dtype=int)
        return torch.from_numpy(arr[indices])

    def __getitem__(self, ix):
        pid = self.planet_ids[ix]
        path = os.path.join(self.root, pid)
        maps = {inst: {m: self._load_map(os.path.join(path, f"{inst}_{m}.parquet"), (32, 356) if 'AIRS' in inst else (32,32))
                       for m in ["signal", "dark", "flat"]} for inst in ["AIRS-CH0", "FGS1"]}

        if self.is_train:
            if torch.rand(1).item() < self.mask_prob: maps["AIRS-CH0"]["dark"].zero_()
            if torch.rand(1).item() < self.mask_prob: maps["AIRS-CH0"]["flat"].fill_(1.0)

        x_airs = (maps["AIRS-CH0"]["signal"] - maps["AIRS-CH0"]["dark"]) / (maps["AIRS-CH0"]["flat"] + 1e-6)
        x_fgs = (maps["FGS1"]["signal"] - maps["FGS1"]["dark"]) / (maps["FGS1"]["flat"] + 1e-6)

        x_meta = torch.zeros(9)
        if self.star_info_df is not None and pid in self.star_info_df.index:
            x_meta = torch.tensor(self.star_info_df.loc[pid].values.astype(np.float32))

        dummy_y = torch.randn(283).float()
        return pid, x_airs.unsqueeze(1), x_fgs.unsqueeze(1), x_meta, dummy_y