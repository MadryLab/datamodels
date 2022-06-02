import numpy as np
import os
from typing import Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    out_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    x_name=Param(str, 'What portion of the data to write', default='masks'),
    y_name=Param(str, 'What portion of the data to write', required=True)
)

class RegressionDataset(Dataset):
    def __init__(self, *, masks_path: str, y_path: str, 
                subset: Optional[Sequence[int]]=None):
        super().__init__()
        self.masks_fp = np.lib.format.open_memmap(masks_path, mode='r')
        self.y_vals_fp = np.lib.format.open_memmap(y_path, mode='r') 
        self.subset = range(self.masks_fp.shape[0]) if subset is None else subset
    
    def __getitem__(self, idx):
        inds = self.subset[idx]
        x_val, y_val = self.masks_fp[inds], self.y_vals_fp[inds].astype('float32')
        return x_val, y_val, inds
    
    def shape(self):
        return self.masks_fp.shape[1], self.y_vals_fp.shape[1]
    
    def __len__(self):
        return len(self.subset)

@param('cfg.data_dir')
@param('cfg.out_dir')
@param('cfg.x_name')
@param('cfg.y_name')
def main(data_dir: str, out_dir: str, x_name: str, y_name: str):
    ds = RegressionDataset(
            masks_path=os.path.join(data_dir, f'{x_name}.npy'), 
            y_path=os.path.join(data_dir, f'{y_name}.npy'))

    x_dim, y_dim = ds.shape()
    print(x_dim, y_dim)
    writer = DatasetWriter(os.path.join(data_dir, 'regression_data.beton'), {
        'mask': NDArrayField(dtype=np.dtype('bool'), shape=(x_dim,)),
        'targets': NDArrayField(dtype=np.dtype('float32'), shape=(y_dim,)),
        'idx': IntField()
    })

    writer.from_indexed_dataset(ds)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()