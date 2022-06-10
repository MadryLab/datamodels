
<h1>Datamodels</h1>
<p align='center'>
Make <a href='https://arxiv.org/abs/2202.00622'>datamodels</a> easily with <code>datamodels</code>!
</p>
<p align='center'>
        [<a href='#overview'>Overview</a>]
        [<a href='#examples'>Examples</a>]
        [<a href='#tutorial'>Tutorial</a>]
        <br/>
        <img src="static/clusters.png"/>
        <it>Figure: One application of datamodels is finding feature directions in a dataset. Above shows examples from a top principal component of datamodel embeddings on CIFAR-10. </it>
        <br/>
</p>

This repository contains the code to estimate *datamodels*, as introduced in our paper:

**Datamodels: Predicting Predictions with Training Data** <br>
*Andrew Ilyas\*, Sung Min Park\*, Logan Engstrom\*, Guillaume Leclerc, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2112.01008 <br>

## Overview
Use the `datamodels` library to computate datamodels for *any* model and dataset:
    (a) train a large number of models (any model!) on different subsets of the training set;
    (b) store their outputs (any output!); and
    (c) compute datamodels using our [fast_l1](https://github.com/MadryLab/fast_l1) library.

### Examples
The easiest way to learn how to use `datamodels` is by example, but we also provide a general [tutorial](#tutorial) below.


<b>Simple examples</b> for a <b>toy setting</b> and <b>CIFAR</b> in the <a href="examples/">examples directory</a>:
<ul>
    <li> <b>Toy Setting</b>: See <a href="examples/minimal/">examples/minimal</a> to see how the entire pipeline works together.</li>
    <li> <b>CIFAR10</b>: See <a href="examples/cifar10">examples/cifar</a>.
    <ul>
    <li> To train, run `example.sh`, which will run data store creation, real model training, and logging. You must run this script from the root of the <code>datamodels</code> directory for it to work properly. </li>
    <li> To compute datamodels, run `example_reg.sh` </li>
    </ul>
    </li>
</ul>

### Pre-computed data on CIFAR-10
For pre-computed datamodels on CIFAR-10, check out <a href='github.com/MadryLab/datamodels-data'>here</a>.


## Tutorial
Computing datamodels with the `datamodels` library has two main stages:

A. Training models and storing arbitrary data in the form of tensors (these
   could include predictions, logits, training data masks, or even model
   weights).
B. Fitting a sparse linear model from any binary covariates (e.g., training
   masks, as in [our paper]()) to a continuous outcome (e.g., margins, as in our
   paper).

We describe each stage in detail below.

### 1. Training models and storing data

There are four key steps here:
1. Write a specification file (in JSON) that pre-declares
   the data you'll want to save, as well as how many models will be
   trained.
2. Create a "store directory" with empty memory-mapped `numpy` arrays based on the above a spec.
3. Make a Python file that implements a `main` function;
   given an index `i` and a log directory `logdir`, the main function should
   execute the appropriate training task, and return a dictionary whose keys
   match the datatypes declared in the specification file.
4. Run a worker for each model that you want to train,
   passing the above Python file and a different index `i` to each one. Each worker will
   call the specified file with the given index.

#### Detailed overview
(follow along in `examples/minimal/` and in particular `examples/cifar10/example.sh`)

1. **Making a spec file**: the first step is to make a specification file, which
   tells `datamodeler` what data is going to be recorded during model training.
   The spec file is written in JSON and has two fields, "num_models" (the number
   of models that will be trained) and "schema", which maps keys to dictionaries
   containing shape and dtype information:
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
    The `dtype` field is any attribute of `np` (i.e., `float` or `uint8` or `bool_`, the numpy boolean type). `num_models` controls how many models are trained.

2. **Setting up the data store**: We use the spec to  make a store
   directory containing contiguous arrays that will store all your datamodel results:
    ```
    python -m datamodels.initialize_logdir \
        --logging.logdir=$tmp_dir \
        --logging.spec=examples/minimal/spec.json
    ```
    The output directory after initialization:
    ```
    > ls $tmp_dir
    _completed.npy
    masks.npy
    margins.npy
    ```
    Each array will be of size ($N$ x ...) where $N$ is the number of models to train, and ... is the shape of
   the "rows" that you want to write (could be scalar, could be vector, could be
   any shape but there will be $N$ of them concatenated together.
    That is, it will contain a numpy array (shape: `(num_models, schema[key]))`
    for each key in the spec schema.


3. **Making a training script**: Next, make a training script with a `main`
   function that just takes an index `index` and a logging directory `logdir`.
   The training script should return a `dict` python type mapping the keys
   declared in the `schema` dictionary above to numpy arrays of the correct
   shape and dtype.

   Note that you probably don't have to use `logdir` unless you want to write
   something other than numpy arrays. Here is a basic example of a `main.py`
   file:
   ```
   def main(*_, index, logdir):
       # Assume 256 training examples and datamodel alpha = 0.5
       mask = np.random.choice(a=[False, True], size=(256,), p=[0.5, 0.5])
       # Assume a function train_model_on_mask that takes in a binary
       # mask and trains on the corresponding subset of the training set
       model = train_model_on_mask(mask)
       margins = evaluate_model(model)
       return {
           'masks': mask,
           'margins': margins
       }
   ```
    The worker will automatically log each key, value in this
    dictionary as a row `<value>` in `logdir/<key>.npy`.
    Note that (as you'll see in the example scripts), you need to manually
    create and save training masks (`datamodels` can log it for you but you will
    need to generate your own masks and then apply them to the training data).

4.  **Running the workers.** The last step is to pass the training script to the
    worker. Each worker run will write into a single index of each contiguous
    array with the datamodel results. Train `num_models` models, each outputting
    matrices corresponding to the (`n`, shape)-shape matrices found in the spec
    file.

    ```
    python -m datamodels.worker \
        --worker.index=0 \
        --worker.main_import=examples.minimal.train_minimal \
        --worker.logdir=$tmp_dir \
        --trainer.multiple=2
    ```
    Note all the arguments not specified in the `datamodels/worker.py` file
    (e.g., `--trainer.multiple` above) will be passed directly to the training
    script that your `main` method is located in.

    You will need to figure out your own way to run thousands of workers. An
    example in GNU parallel with 8 workers at once would be:
    ```zsh
    parallel -j 8 CUDA_VISIBLE_DEVICES='{=1 $_=$arg[1] % 8 =}' \
                        python -m datamodels.worker \
                        --worker.index={} \
                        --worker.main_import=examples.minimal.train_minimal \
                        --worker.logdir=/tmp/ \
                        --trainer.multiple=2 ::: {0..999}
    ```
    For our own lab's projects, we use GNU parallel in conjunction with our SLURM cluster.

### 2. Running sparse linear regression

After training the desired number of models, you can use
`datamodels.regression.compute_datamodels` to fit sparse linear models from
training masks (or any other binary output saved in the last step) to margins
(or any other continuous output saved in the last step). There are only three steps here:

1. **Writing a dataset**: first, we convert the data (stored in memory-mapped
   arrays from the previous steps) to [FFCV](https://ffcv.io) format:

   ```
   python -m datamodels.regression.write_dataset \
                --cfg.data_dir PATH_TO_NPY_DIR \
                --cfg.out_path OUT_DIR/data.beton \
                --cfg.y_name NAME_OF_YVAR \
                --cfg.x_name NAME_OF_XVAR
   ```

2. **Making a config file**: The next step is to make a config file. See, for
   example, below:
    ```yaml
    data:
        data_path: 'OUT_DIR/data.beton' # Path to FFCV dataset
        num_train: 90_000 # Number of models to use for training
        num_val: 10_000 # Number of models to use for validation
        seed: 0 # Random seed for picking the validation set
        target_start_ind: 0 # Select a slice of the data to run on
        target_end_ind: -1 # Select a slice of the data to run on (-1 = end)
        # If target_start_ind and target_end_ind are specified, will slice the
        # output variable accordingly
    cfg:
        k: 100 # Number of lambdas along the regularization path
        batch_size: 2000 # Batch size for regression, must divide num_train and num_val
        lr: 5e-3 # Learning rate
        eps: 1e-6 # Multiplicative difference between the highest and lowest lambda
        out_dir: OUT_DIR # Where to save the results
        num_workers: 8 # Number of workers for dataloader
    early_stopping:
        check_every: 3 # How often to check for inner and outer-loop convergence
        eps: 5e-11 # Tolerance for inner-loop convergence (see GLMNet docs)
    ```

3. **Running the regression**: once the config file has been made, the last step
   is to just run the regression script:
   ```
   python -m datamodels.regression.compute_datamodels -C config_file.yml
   ```


## Citation
If you use this code in your work, please cite using the following BibTeX entry:
```
@inproceedings{ilyas2022datamodels,
  title = {Datamodels: Predicting Predictions from Training Data},
  author = {Andrew Ilyas and Sung Min Park and Logan Engstrom and Guillaume Leclerc and Aleksander Madry},
  booktitle = {ArXiv preprint arXiv:2202.00622},
  year = {2022}
}
```
