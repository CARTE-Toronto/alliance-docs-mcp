---
title: "CUDA/en"
url: "https://docs.alliancecan.ca/wiki/CUDA/en"
category: "General"
last_modified: "2023-06-01T21:58:01Z"
page_id: 6116
display_title: "CUDA"
---

"CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs)."NVIDIA CUDA Home Page. CUDA is a registered trademark of NVIDIA.

It is reasonable to think of CUDA as a set of libraries and associated C, C++, and Fortran compilers that enable you to write code for GPUs. See OpenACC Tutorial for another set of GPU programming tools.

== Quick start guide ==

===Compiling===
Here we show a simple example of how to use the CUDA C/C++ language compiler, nvcc, and run code created with it. For a longer tutorial in CUDA programming, see CUDA tutorial.

First, load a CUDA module.

$ module purge
$ module load cuda

The following program will add two numbers together on a GPU. Save the file as add.cu. The cu file extension is important!.

nvcc to create an executable named add.

$ nvcc add.cu -o add

=== Submitting jobs===
To run the program, create a Slurm job script as shown below. Be sure to replace def-someuser with your specific account (see Accounts and projects). For options relating to scheduling jobs with GPUs see Using GPUs with Slurm.

Submit your GPU job to the scheduler with

$ sbatch gpu_job.sh
Submitted batch job 3127733
For more information about the sbatch command and running and monitoring jobs, see Running jobs.

Once your job has finished, you should see an output file similar to this:

$ cat slurm-3127733.out
2+7=9

If you run this without a GPU present, you might see output like 2+7=0.

=== Linking libraries ===
If you have a program that needs to link some libraries included with CUDA, for example cuBLAS, compile with the following flags

nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64

To learn more about how the above program works and how to make the use of GPU parallelism, see CUDA tutorial.

== Troubleshooting ==

=== Compute capability ===

NVidia has created this technical term, which they describe as follows:

The compute capability of a device is represented by a version number, also sometimes called its "SM version". This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU."  (CUDA Toolkit Documentation, section 2.6)

The following errors are connected with compute capability:

nvcc fatal : Unsupported gpu architecture 'compute_XX'

no kernel image is available for execution on the device (209)

If you encounter either of these errors, you may be able to fix it by adding the correct flag to the nvcc call:

-gencode arch=compute_XX,code=[sm_XX,compute_XX]

If you are using cmake, provide the following flag:

cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX

where “XX” is the compute capability of the Nvidia GPU that you expect to run the application on.
To find the value to replace “XX“, see the Available GPUs table.

For example, if you will run your code on a Narval A100 node, its compute capability is 80.
The correct flag to use when compiling with nvcc is

-gencode arch=compute_80,code=[sm_80,compute_80]

The flag to supply to cmake is:

cmake .. -DCMAKE_CUDA_ARCHITECTURES=80