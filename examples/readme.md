# <code>datamodels</code> example usage

<b>Simple examples</b> for a <b>toy setting</b> and <b>CIFAR-10</b> that run out of the box!
<ul>
    <li> <b>Toy Setting</b>: See <a href="examples/minimal/">examples/minimal</a> for a dummy example that illustrates how the entire pipeline works together. Simply run <code>example.sh</code>.</li>
    <li> <b>CIFAR10</b>: See <a href="examples/cifar10">examples/cifar</a>:
    <ul>
    <li> To train, run <code>example.sh</code>, which will create necessary data storage, train models, and log their outputs. You must run this script from the root of the <code>datamodels</code> directory for it to work properly. The script also assumes you are running with a machine with access to 8 GPUs (you can modify as appropriate). </li>
    <li> To compute datamodels, run <code>example_reg.sh</code> after modifying the variable <code>tmp_dir</code> to be the directory output by the training script. </li>
    </ul>
    </li>
</ul>