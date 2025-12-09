---
title: "Chapel/en"
url: "https://docs.alliancecan.ca/wiki/Chapel/en"
category: "General"
last_modified: "2025-07-25T17:32:57Z"
page_id: 17440
display_title: "Chapel"
---

Chapel is a general-purpose, compiled, high-level parallel programming language with built-in abstractions for shared- and distributed-memory parallelism. There are two styles of parallel programming in Chapel: (1) task parallelism, where parallelism is driven by programmer-specified tasks, and (2) data parallelism, where parallelism is driven by applying the same computation on subsets of data elements, which may be in the shared memory of a single node, or distributed over multiple nodes.

These high-level abstractions make Chapel ideal for learning parallel programming for a novice HPC user. Chapel is incredibly intuitive, striving to merge the ease-of-use of Python and the performance of traditional compiled languages such as C and Fortran. Parallel blocks that typically take tens of lines of MPI code can be expressed in only a few lines of Chapel code. Chapel is open source and can run on any Unix-like operating system, with hardware support from laptops to large HPC systems.

Chapel has a relatively small user base, so many libraries that exist for C, C++, Fortran have not yet been implemented in Chapel. Hopefully, that will change in coming years if Chapel adoption continues to gain momentum in the HPC community.

For more information, please watch our Chapel webinars.

== Single-locale Chapel ==

Single-locale (single node; shared-memory only) Chapel on our general-purpose clusters is provided by the module chapel-multicore. You can use salloc to test Chapel codes either in serial:
0:30:0 --ntasks1 --mem-per-cpu3600 --accountdef-someprof
|chpl test.chpl -o test
|./test
}}
or on multiple cores on the same node:
0:30:0 --ntasks1 --cpus-per-task3 --mem-per-cpu3600 --accountdef-someprof
|chpl test.chpl -o test
|./test
}}
For production jobs, please write a job submission script and submit it with sbatch.

== Multi-locale Chapel ==

Multi-locale (multiple nodes; hybrid shared- and distributed-memory) Chapel on our InfiniBand clusters is provided by the module chapel-ucx.

Consider the following Chapel code printing basic information about the nodes available inside your job:

To run this code on an InfiniBand cluster, you need to load the chapel-ucx module:
0:30:0 --nodes4 --cpus-per-task3 --mem-per-cpu3500 --accountdef-someprof
}}

Once the interactive job starts, you can compile and run your code from the prompt on the first allocated compute node:

For production jobs, please write a Slurm submission script and submit your job with sbatch instead.

== Multi-locale Chapel with NVIDIA GPU support ==

To enable GPU support, please use the module chapel-ucx-cuda. It adds NVIDIA GPU support to multi-locale Chapel on our InfiniBand clusters.

Consider the following basic Chapel GPU code:

To run this code on an InfiniBand cluster, you need to load the chapel-ucx-cuda module:
0:30:0 --mem-per-cpu3500 --gpus-per-node1 --accountdef-someprof
}}

Once the interactive job starts, you can compile and run your code from the prompt on the allocated compute node:

For production jobs, please write a Slurm submission script and submit your job with sbatch instead.