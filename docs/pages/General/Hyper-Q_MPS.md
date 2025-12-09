---
title: "Hyper-Q / MPS/en"
url: "https://docs.alliancecan.ca/wiki/Hyper-Q_/_MPS/en"
category: "General"
last_modified: "2025-08-31T14:16:03Z"
page_id: 24565
display_title: "Hyper-Q / MPS"
---

==Overview==

Hyper-Q (or MPS) is a feature of NVIDIA GPUs.
It is available in GPUs with CUDA compute capability 3.5 and higherFor a table relating NVIDIA GPU model names, architecture names, and CUDA compute capabilities, see https://en.wikipedia.org/wiki/Nvidia_Tesla
.

According to NVIDIA,
::The MPS runtime architecture is designed to transparently enable co-operative multi-process CUDA applications, typically MPI jobs, to utilize Hyper-Q capabilities on the latest NVIDIA (Kepler and later) GPUs. Hyper-Q allows CUDA kernels to be processed concurrently on the same GPU; this can benefit performance when the GPU compute capacity is underutilized by a single application process.

In our tests, MPS may increase the total GPU flop rate even when the GPU is being shared by unrelated CPU processes. That means that MPS is great for CUDA applications with relatively small problem sizes, which on their own cannot efficiently saturate modern GPUs with thousands of cores.

MPS is not enabled by default, but it is straightforward to do.  Execute the following commands before running your CUDA application:

 /tmp/nvidia-mps
|export CUDA_MPS_LOG_DIRECTORY/tmp/nvidia-log
|nvidia-cuda-mps-control -d}}

Then you can use the MPS feature if you have more than one CPU thread accessing the GPU. This will happen if you run a hybrid MPI/CUDA application, a hybrid OpenMP/CUDA application, or multiple instances of a serial CUDA application (GPU farming).

Additional details on MPS can be found here: CUDA Multi Process Service (MPS) - NVIDIA Documentation.

==GPU farming==

One situation when the MPS feature can be very useful is when you need to run multiple instances of a CUDA application, but the application is too small to saturate a modern GPU.  MPS allows you to run multiple instances of the application sharing a single GPU, as long as there is enough of GPU memory for all of the instances of the application.  In many cases this should result in a significantly increased throughput from all of your GPU processes.

Here is an example of a job script to set up GPU farming:

In the above example, we share a single V100 GPU between 8 instances of my_code (which takes a single argument-- the loop index $i). We request 8 CPU cores (#SBATCH -c 8) so there is one CPU core per application instance. The two important elements are
* & on the code execution line, which sends the code processes to the background, and
* the wait command at the end of the script, which ensures that the job runs until all background processes end.