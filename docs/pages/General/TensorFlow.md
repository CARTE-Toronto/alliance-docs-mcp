---
title: "TensorFlow/en"
url: "https://docs.alliancecan.ca/wiki/TensorFlow/en"
category: "General"
last_modified: "2025-08-29T13:09:26Z"
page_id: 3577
display_title: "TensorFlow"
---

TensorFlow is an open-source software library for Machine Intelligence.

If you are porting a TensorFlow program to an Alliance cluster, you should follow our tutorial on machine learning.

==Installing TensorFlow==

These instructions install TensorFlow in your /home directory using Alliance's prebuilt Python wheels. Custom Python wheels are stored in /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/. To install a TensorFlow wheel, we will use the pip command and install it into a  Python virtual environment.

Load modules required by TensorFlow. In some cases, other modules may be required (e.g. CUDA).

Create a new Python virtual environment.

Activate your newly created Python virtual environment.

Install TensorFlow in your newly created virtual environment using the following command.

Load modules required by TensorFlow. TF 1.x requires StdEnv/2018.

Note: TF 1.x is not available on Narval, since StdEnv/2018 is not available on this cluster.

Create a new Python virtual environment.

Activate your newly created Python virtual environment.

Install TensorFlow in your newly created virtual environment using one of the commands below, depending on whether you need to use a GPU.

Do not install the tensorflow package (without the _cpu or _gpu suffixes) as it has compatibility issues with other libraries.

=== CPU-only ===
1.15.0}}

=== GPU ===
1.15.0}}

=== R package ===

To use TensorFlow in R, you will need to first follow the preceding instructions on creating a virtual environment and installing TensorFlow in it. Once this is done, follow these instructions.

Load the required modules.

Activate your Python virtual environment.

Launch R.

In R, install package devtools, then tensorflow:

install.packages('devtools', repos='https://cloud.r-project.org')
devtools::install_github('rstudio/tensorflow')

You are then good to go. Do not call install_tensorflow() in R, as TensorFlow has already been installed in your virtual environment with pip. To use the TensorFlow installed in your virtual environment, enter the following commands in R after the environment has been activated.

library(tensorflow)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))

==Submitting a TensorFlow job with a GPU==
Once you have the above setup completed, you can submit a TensorFlow job.

The job submission script contains

while the Python script has the form

Once the job has completed (should take less than a minute), you should see an output file called something like node_id-job_id.out with contents similar to the following (the logged messages from TensorFlow are only examples, expect different messages and more messages):

TensorFlow can run on all GPU node types. See Using GPUs with SLURM for more information.

==Monitoring==

It is possible to connect to the node running a job and execute processes. This can be used to monitor resources used by TensorFlow and to visualize the progress of the training. See Attaching to a running job for examples.

===TensorBoard===

TensorFlow comes with a suite of visualization tools called TensorBoard. TensorBoard operates by reading TensorFlow events and model files. To know how to create these files, read TensorBoard tutorial on summaries.

TensorBoard requires too much processing power to be run on a login node. Users are strongly encouraged to execute it in the same job as the Tensorflow process. To do so, launch TensorBoard in the background by calling it before your python script, and appending an ampersand (&) to the call:

 # Your SBATCH arguments here

 tensorboard --logdir=/tmp/your_log_dir --host 0.0.0.0 --load_fast false &
 python train.py  # example

Once the job is running, to access TensorBoard with a web browser, you need to create a connection between your computer and the compute node running TensorFlow and TensorBoard. To do this you first need the hostname of the compute node running the Tensorboard server. Show the list of your jobs using the command sq; find the job, and note the value in the "NODELIST" column (this is the hostname).

To create the connection, use the following command on your local computer:

Replace computenode with the node hostname you retrieved from the preceding step, userid by your Alliance username, cluster by the cluster hostname (i.e.: rorqual, fir, nibi, etc.). If port 6006 was already in use, tensorboard will be using another one (e.g. 6007, 6008...).

Once the connection is created, go to http://localhost:6006.

==TensorFlow with multi-GPUs==

TensorFlow offers a number of different strategies to make use of multiple GPUs through the high-level API tf.distribute. In the following sections, we provide code examples of each strategy using Keras for simplicity. For more details, please refer to the official TensorFlow documentation.

====Mirrored strategy====

=====Single node=====

The Python script tensorflow-singleworker.py has the form:

=====Multiple nodes=====

The syntax to use multiple GPUs distributed across multiple nodes is very similar to the single node case, the most notable difference being the use of MultiWorkerMirroredStrategy(). Here, we use SlurmClusterResolver() to tell TensorFlow to acquire all the necessary job information from SLURM, instead of manually assigning master and worker nodes, for example. We also need to add CommunicationImplementation.NCCL to the distribution strategy to specify that we want to use Nvidia's NCCL backend for inter-GPU communications. This was not necessary in the single-node case, as NCCL is the default backend with MirroredStrategy().

Where config_env.sh has the form:

The script launch_training.sh has the form:

And the Python script tensorflow-multiworker.py has the form:

==Creating model checkpoints==
Whether or not you expect your code to run for long time periods, it is a good habit to create Checkpoints during training. A checkpoint is a snapshot of your model at a given point during the training process (after a certain number of iterations or after a number of epochs) that is saved to disk and can be loaded at a later time. It is a handy way of breaking jobs that are expected to run for a very long time, into multiple shorter jobs that may get allocated on the cluster more quickly. It is also a good way of avoiding losing progress in case of unexpected errors in your code or node failures.

===With Keras===

To create a checkpoint when training with keras, we recommend using the callbacks parameter of the model.fit() method. The following example shows how to instruct TensorFlow to create a checkpoint at the end of every training epoch:

 callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./ckpt",save_freq="epoch")] # Make sure the path where you want to create the checkpoint exists

 model.fit(dataset, epochs=10 , callbacks=callbacks)

For more information, please refer to the official TensorFlow documentation.

===With a custom training loop===

Please refer to the official TensorFlow documentation.

==Troubleshooting==

===scikit image===

If you are using the scikit-image library, you may get the following error:
OMP: Error #15: Initializing libiomp5.so, but found libiomp5.so already initialized.

This is because the tensorflow library tries to load a bundled version of OMP which conflicts with the system version. The workaround is as follows:
$(strace python -c 'from skimage.transform import AffineTransform' 2>&1  grep -v ENOENT  grep -ohP -e '(?<")[^"]+libiomp5.so(?")'  xargs realpath)
|find -path '*_solib_local*' -name libiomp5.so -exec ln -sf $LIBIOMP_PATH {} \;
}}
This will patch the tensorflow library installation to use the systemwide libiomp5.so.

===libcupti.so===

Some tracing features of Tensorflow require libcupti.so to be available, and might give the following error if they are not:

I tensorflow/stream_executor/dso_loader.cc:142] Couldn't open CUDA library libcupti.so.9.0. LD_LIBRARY_PATH: /usr/local/cuda-9.0/lib64

The solution is to run the following before executing your script:
$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64/
}}
Where xxx is the appropriate CUDA version, which can be found using module av cuda

===libiomp5.so invalid ELF header===

Sometimes the libiomp5.so shared object file will be erroneously installed as a text file. This might result in errors like the following:

/home/username/venv/lib/python3.6/site-packages/tensorflow/python/../../_solib_local/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib/libiomp5.so: invalid ELF header

The workaround for such errors is to access the directory mentioned in the error (i.e. [...]/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib) and execute the following command:

This will replace the text file with the correct symbolic link.

==Controlling the number of CPUs and threads==

The config parameters intra_op_parallelism_threads and inter_op_parallelism_threads influence the number of threads used by TensorFlow. You can set those parameters with:

 tf.config.threading.set_inter_op_parallelism_threads(num_threads)
 tf.config.threading.set_intra_op_parallelism_threads(num_threads)

==Known issues==
A bug sneaked into the Keras implementation of Tensorflow after version 2.8.3. It affects the performance of the layers used for data augmentation with prefix tf.keras.layers.Random (like tf.keras.layers.RandomRotation, tf.keras.layers.RandomTranslation, etc). It significantly slows down the training process by more than 100 times. The bug is fixed in version 2.12.