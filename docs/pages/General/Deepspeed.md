---
title: "Deepspeed"
url: "https://docs.alliancecan.ca/wiki/Deepspeed"
category: "General"
last_modified: "2025-07-18T15:12:27Z"
page_id: 23434
display_title: "Deepspeed"
---

DeepSpeed is a deep learning training optimization library, providing the means to train massive billion parameter models at scale. Fully compatible with PyTorch, DeepSpeed features implementations of novel memory-efficient distributed training methods, based on the Zero Redundancy Optimizer (ZeRO) concept. Through the use of ZeRO, DeepSpeed enables distributed storage and computing of different elements of a training task - such as optimizer states, model weights, model gradients and model activations - across multiple devices, including GPU, CPU, local hard disk, and/or combinations of these devices. This "pooling" of resources, notably for storage, allows models with massive amounts of parameters to be trained efficiently, across multiple nodes, without explicitly handling Model, Pipeline or Data Parallelism in your code.

==Installing Deepspeed==

Our recommendation is to install it using our provided Python wheel as follows:
:1. Load a Python module, thus module load python
:2. Create and start a virtual environment.
:3. Install both PyTorch and Deepspeed in the virtual environment with pip install.

:

==Multi-GPU and multi-node jobs with Deepspeed==
In the example that follows, we use deepspeed to reproduce our PyTorch tutorial on how to train a model with multiple GPUs distributed over multiple nodes. Notable differences are:

:1. Here we define and configure several common elements of the training task (such as optimizer, learning rate scheduler, batch size and more) in a config file, rather than using code in the main python script.
:2. We also define Deepspeed specific configurations, such as what modality of ZeRO to utilize, in a config file.

Where the script config_env.sh is:

The script launch_training_deepseed.sh is as shown below. Notice that we use torchrun to launch our python script. While Deepspeed has its own launcher, we do not recommend using it at this time:

Next we define and configure our training task in the file ds_config.json. Here we setup ZeRO stage 0, meaning ZerRO is disabled - no model parallelism will take place and this will be a purely data parallel job. We also enable mixed-precision training, where some tensors are computed/stored in half-precision (fp16) to accelerate computations using up less memory space. See Deepspeed's documentation for more details on all configurable parameters.

And finally, pytorch-deepspeed.py is:

== Using PyTorch Lightning ==
In the following tutorial, we use PyTorch Lightning as a wrapper around Deepspeed and demonstrate how to use ZeRO Stage 3 with a pool of GPUs, with offloading to the CPU, and with offloading to the compute node's local storage.

=== ZeRO on GPU ===

In the following example, we use ZeRO Stage 3 to train a model using a "pool" of 4 GPUs. Stage 3 means all three of: optimizer states; model parameters; and model gradients will be split (sharded) between all 4 GPUs. This is more memory-efficient than pure Data Parallelism, where we would have a full replica of the model loaded on each GPU. Using DeepSpeed's optimizer FusedAdam instead of a native PyTorch one, performance is comparable with pure Data Parallelism. DeepSpeed's optimizers are JIT compiled at run-time and you must load the module cuda/ where  must match the version used to build the PyTorch install you are using.

=== ZeRO with offload to CPU ===

In this example, we will again use ZeRO stage 3, but this time we enable offloading model parameters and optimizers states to the CPU. This means that the compute node's memory will be available to store these tensors while they are not required by any GPU computations, and additionally, optimizer steps will be computed on the CPU. For practical purposes, you can think of this as though your GPUs were gaining an extra 32GB of memory. This takes even more pressure off from GPU memory and would allow you to increase your batch size, for example, or increase the size of the model. Using DeepSpeed's optimizer DeepSpeedCPUAdam instead of a native PyTorch one, performance remains at par with pure Data Parallelism. DeepSpeed's optimizers are JIT compiled at run-time and you must load the module cuda/ where  must match the version used to build the PyTorch install you are using.

=== ZeRO with offload to NVMe ===
In this example, we use ZeRO stage 3 yet again, but this time we enable offloading model parameters and optimizers states to the local disk. This means that the compute node's local disk storage will be available to store these tensors while they are not required by any GPU computations. As before, optimizer steps will be computed on the CPU. Again, for practical purposes, you can think of this as extending GPU memory by however much storage is available on the local disk, though this time performance will significantly degrade. This approach works best (i.e., performance degradation is least noticeable) on NVMe-enabled drives, which have higher throughput and faster response times, but it can be used with any type of storage.