---
title: "MXNet/en"
url: "https://docs.alliancecan.ca/wiki/MXNet/en"
category: "General"
last_modified: "2022-11-09T18:31:50Z"
page_id: 11093
display_title: "MXNet"
---

Apache MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scalable to many GPUs and machines.

= Available wheels =
You can list available wheels using the avail_wheels command.

= Installing in a Python virtual environment =
1. Create and activate a Python virtual environment.

2. Install MXNet and its Python dependencies.

3. Validate it.

= Running a job =
A single Convolution layer:

2. Edit the following submission script according to your needs.

3. Submit the job to the scheduler.

= High Performance with MXNet =

== MXNet with Multiple CPUs or a Single GPU ==

Similar to PyTorch and TensorFlow, MXNet contains both CPU and GPU-based parallel implementations of operators commonly used in Deep Learning, such as matrix multiplication and convolution, using OpenMP and MKLDNN (CPU) or CUDA and CUDNN (GPU). Whenever you run MXNet code that performs such operations, they will either automatically leverage multi-threading over as many CPU cores as are available to your job, or run on the GPU if your job requests one.

With the above being said, when training small scale models we strongly recommend using multiple CPUs instead of using a GPU. While training will almost certainly run faster on a GPU (except in cases where the model is very small), if your model and your dataset are not large enough, the speed up relative to CPU will likely not be very significant and your job will end up using only a small portion of the GPU's compute capabilities. This might not be an issue on your own workstation, but in a shared environment like our HPC clusters this means you are unnecessarily blocking a resource that another user may need to run actual large scale computations! Furthermore, you would be unnecessarily using up your group's allocation and affecting the priority of your colleagues' jobs.

Simply put, you should not ask for a GPU if your code is not capable of making a reasonable use of its compute capacity. The following example illustrates how to train a Convolutional Neural Network using MXNet with or without a GPU: