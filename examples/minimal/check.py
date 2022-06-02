from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np

from datamodels.training.utils import load_memmap
from datamodels.training.spec import COMPLETED

Section('trainer').params(
    multiple=Param(int, 'multiple', required=True),
)

Section('checking').params(
    logdir=Param(str, 'log directory', required=True),
)

@param('checking.logdir')
@param('trainer.multiple')
def main(logdir, multiple):
    x = load_memmap(logdir, 'random_numbers', 'r')
    assert (x[0] == np.ones(256, dtype=np.uint8) * multiple).all()
    assert x[1:].sum() == 0

    x = load_memmap(logdir, COMPLETED, 'r')
    assert x.sum() == 1

    print('checks all passed!')

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == '__main__':
    make_config()
    main()
