from fastargs.decorators import param
from fastargs import Param, Section

import numpy as np
import json

from .spec import preprocess_spec
from .utils import memmap_path, make_config

Section('logging').params(
    logdir=Param(str, 'file with main() to run'),
    spec=Param(str, 'file with spec')
)

@param('logging.logdir')
@param('logging.spec')
def main(logdir, spec):
    assert logdir is not None
    assert spec is not None
    spec = json.loads(open(spec, 'r').read())
    spec = preprocess_spec(spec)

    num_models = spec["num_models"]
    for key, metadata in spec["schema"].items():
        dtype = getattr(np, metadata['dtype'])
        shape = (num_models,) + tuple(metadata['shape'])

        this_filename = memmap_path(logdir, key)
        mmap = np.lib.format.open_memmap(this_filename, mode='w+', dtype=dtype,
                                         shape=shape)
        mmap.flush()

if __name__ == '__main__':
    make_config()
    main()
    print('Done!')
