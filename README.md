
<h1>Datamodels model training code release</h1>
<p align='center'>
Make <a href='https://arxiv.org/abs/2202.00622'>datamodels</a> easily with <code>datamodels</code>!
</p>
<p align='center'>
        [<a href='#examples'>examples</a>] [<a href='#tutorial'>overview</a>]
        <br/>
        <img src="static/clusters.png"/>
        <it>One application: find opposing clusters in your datasets</it>
        <br/>
</p>

## Examples
Use the `datamodels` library to (a) train a large number of models (any model!) with different training sets and (b) store their outputs (any output!). `datamodels` is best learned through example; however, if you don't like examples, there is a [tutorial](#tutorial) below.

<p><b>Simple examples</b> for a <b>toy setting</b> and <b>CIFAR</b> in the <a href="examples/">examples directory</a>:
        <ul>
                <li> <b>Toy Setting</b>: See <a href="examples/minimal/">example.sh</a> to see how the entire pipeline works together.</li>
<li> <b>CIFAR10</b>: See <a href="examples/cifar10/example.sh">example.sh</a>. It is a complete example, including data store creation, real model training, and logging. You must run this script from the root of the <code>datamodels</code> directory for it to work properly. </li>
                </ul>
</p>

## Tutorial
An overview of how the datamodel data collection system here works

### Quick overview
- Make a store directory with empty tensors of data for each type of data that you want your models to train
- Make a python file that executes a training task given an index `i` (different `i` could sample different datasets, for example, as in datamodels)
- Then run a worker for each model that you want to train; each model `i` will write in row `i` of the empty tensors with its data

### Detailed overview
(follow along in `examples/minimal/` and in particular `examples/cifar10/example.sh`)

**Setting up the data store.** First make a store directory containing contiguous arrays that will store all your datamodel results. Each array will be of size (n x ...) where n is the number of models to train, and ... is the shape of the "rows" that you want to write (could be scalar, could be vector, could be any shape but there will be n of them concatenated together. This step is:
```
python -m datamodels.initialize_logdir \
    --logging.logdir=$tmp_dir \
    --logging.spec=examples/minimal/spec.json
```

The spec is json formatted like this:
```
{
    "num_models": 10,
    "schema": {
        "masks": {
            "shape": [50000],
            "dtype": "bool_"
        },
        "margins": {
            "shape": [10000],
            "dtype": "float16"
        }
    }
}
```
`dtype` field is any attribute of `np` (i.e. `float` or `uint8` or `bool_`, the numpy boolean type). `num_models` controls how many models are trained.
The output directory looks like:
```
> ls -l $tmp_dir
-rw-rw-r--  1 engstrom engstrom    138 Jun  1 20:07 _completed.npy
-rw-rw-r--  1 engstrom engstrom 500128 Jun  1 20:07 masks.npy
-rw-rw-r--  1 engstrom engstrom 200128 Jun  1 20:07 margins.npy
```
That is, it will contain a numpy matrix (shape: (num_models, schema\[`key`\]))
for each key in the spec schema.

**Making a training script.**
Then make a training script with a `main` file that just takes an index and a logging directory.
The training script should return a `dict` python type mapping keys `data name` -> values `row of data to write`.
You don't have to do any logging in `logdir` unless you want to write something
other than arrays (this should be handled by the `worker.py` file and the
dictionary that you're returning. This is exactly the `train_minimal.py` file:
```
def main(*_, index, logdir):
    make_config()
    return execute(index=index)
    
# Could return something like (if you had 256 training examples and a
#   fractionally valued datamodels alpha)
{
    'masks': np.random.choice(a=[False, True], size=(256,), p=[1 - alpha, alpha])
}
```
The worker will automatically log each key, value in this dictionary as a row `<value>` in `logdir/<key>.npy`.
Note that unless you use one of the example scripts you will need to implement
your own datamodel masking system (`datamodels` can log it for you but you will
need to generate your own masks and then apply them to the training data).

**Running the workers.**
Pass the training script to the worker. Each worker run will write into a single index of each contiguous array with the datamodel results. Train `num_models` models, each outputting matrices corresponding to the (`n`, shape)-shape matrices found in the spec file.


```
python -m datamodels.worker \
    --worker.index=0 \
    --worker.main_import=examples.minimal.train_minimal \
    --worker.logdir=$tmp_dir \
    --trainer.multiple=2
```

Here all the arguments passed to the script, even those not specified in the `datamodels/worker.py` file (i.e. `--trainer.multiple` here that the `main_import` uses) will be passed directly to the training script that your `main` method is located in.

**NOTE: you will need to figure out your own way to run thousands of workers; we use GNU parallel in conjunction with our SLURM cluster.**
