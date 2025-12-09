---
title: "Ray"
url: "https://docs.alliancecan.ca/wiki/Ray"
category: "General"
last_modified: "2025-11-14T21:28:51Z"
page_id: 22158
display_title: "Ray"
---

Ray is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries for simplifying running parallel/distributed workloads, in particular Machine Learning jobs.

= Installation =

==Latest available wheels==
To see the latest version of Ray that we have built:

For more information, see Available wheels.

==Installing our wheel==

The preferred option is to install it using the Python wheel as follows:
:1. Load a Python module, thus module load python
:2. Create and start a virtual environment.
:3. Install Ray in the virtual environment with pip install.

:

= Job submission =
In the example that follows, we submit a job that spawns a single-node Ray cluster with 200GB RAM, 48 CPU cores and 4 H100 GPUs per node.

The script can be used for multi-node Ray clusters by changing the default value set by the line #SBATCH --nodes=1, or overriding it with sbatch's --nodes= command-line argument.

Where the script config_env.sh is:

In this simple example, we connect to the Ray cluster launched in the job submission script, then we check that Ray sees the resources allocated to the job.

= Hyperparameter search with Ray Tune =

Tune is a Ray module for experiment execution and hyperparameter tuning at any scale. It supports a wide range of frameworks including Pytorch, Tensorflow and Scikit-Learn. In the example that follows, we use Tune to perform a hyperparameter sweep and find the best combination of learning rate and batch size to train a convolutional neural network with Pytorch. You can find examples using other frameworks on Ray's official documentation

To run this example, you can use one of the job submission templates provided  above depending on whether you require one or multiple nodes. As you will see in the code that follows, the amount of resources required by your job will depend mainly on two factors: the number of samples you wish to draw from the search space and the size of your model in memory. Knowing these two things you can reason about how many trials you will run in total and how many of them can run in parallel using as few resources as possible. For example, how many copies of your model can you fit inside the memory of a single GPU? That is the number of trials you can run in parallel using just one GPU.

In the example, our model takes up about 1GB in memory. We will run 20 trials in total, 10 in parallel at a time on the same GPU, and we will give one CPU to each trial to be used as a DataLoader worker. So we will pick the single node job submission template and we will replace the number of cpus per task with #SBATCH --cpus-per-task=10 and the Python call with python ray-tune-example.py --num_samples=20 --cpus-per-trial=1 gpus-per-trial=0.1. We will also need to install the packages ray[tune] and torchvision in our virtualenv.