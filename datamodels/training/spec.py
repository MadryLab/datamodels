from copy import deepcopy
import numpy as np

COMPLETED = '_completed'
NUM_MODELS = 'num_models'
SCHEMA = 'schema'
DTYPE = 'dtype'
SHAPE = 'shape'

def _add_completed(spec):
    spec = deepcopy(spec)
    spec[SCHEMA][COMPLETED] = {
        DTYPE:"bool_",
        SHAPE:[]
    }

    return spec

def preprocess_spec(spec):
    verify_spec(spec)
    spec = _add_completed(spec)
    return spec

def verify_spec(spec):
    assert NUM_MODELS in spec, f'you need a {NUM_MODELS} attribute'
    assert SCHEMA in spec,  f'you need a {SCHEMA} attribute'
    assert not COMPLETED in spec[SCHEMA], f'no schema dtypes called {COMPLETED} allowed'

    schema = spec[SCHEMA]
    for _, v in schema.items():
        assert DTYPE in v, 'you need a dtype'
        assert SHAPE in v, 'you need a shape'

        this_dtype = v[DTYPE]
        this_shape = v[SHAPE]
        assert type(this_shape) is list, "your shape must be a list"
        assert hasattr(np, this_dtype), f"your dtype {this_dtype} is not a numpy dtype"
