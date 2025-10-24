---
title: "NAMD/en"
url: "https://docs.alliancecan.ca/wiki/NAMD/en"
category: "General"
last_modified: "2025-10-20T16:52:27Z"
page_id: 5596
display_title: "NAMD"
---

`<languages />`{=html}

[NAMD](http://www.ks.uiuc.edu/Research/namd/) is a parallel, object-oriented molecular dynamics code designed for high-performance simulation of large biomolecular systems. Simulation preparation and analysis is integrated into the [VMD](https://docs.alliancecan.ca/VMD "VMD"){.wikilink} visualization package.

# Installation

NAMD is installed by our software team and is available as a module. If a new version is required or if for some reason you need to do your own installation, please contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}. You can also ask for details of how our NAMD modules were compiled.

# Environment modules {#environment_modules}

The latest version of NAMD is 3.0.2 (released August 27, 2025) and it has been installed on all clusters. It has a number of [improvements](https://www.ks.uiuc.edu/Research/namd/3.0.2/announce.html) over the previous version 3.0.1. The previous version 2.14 is also available, but it cannot run on H100 GPUs, so it should be used only on the Narval cluster if the GPU version is used.

To run jobs that span nodes, use UCX.

# Submission scripts {#submission_scripts}

Please refer to the [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} page for help on using the SLURM workload manager.

## Threaded CPU jobs {#threaded_cpu_jobs}

Below is a job script for a threaded simulation. You can increase the number for \--cpus-per-task to use more cores, up to the maximum number of cores available on a cluster node. Check below in \"Performance and benchmarking\" section for advice on how to choose the number of cores to use.

## Multi-node CPU jobs {#multi_node_cpu_jobs}

### UCX jobs {#ucx_jobs}

This example uses 384 processes in total on 2 nodes, each node running 192 processes. This script assumes full nodes are used, thus `ntasks-per-node` should be equal to the number of cores available on the node (192 on fir, rorqual, nibi and trillium). For best performance, NAMD UCX jobs should use full nodes. Only use UCX version if the multicore version of NAMD is not sufficient for your needs and you need to run on multiple nodes.

## Threaded GPU jobs {#threaded_gpu_jobs}

This example uses 16 CPU cores and 1 H100 GPU on a single node. You can increase the number of GPUs and CPU cores up to the maximum available on the node.

**Important**: NAMD 3 added an input flag, which shifts more calculations to the GPU. This can improve performance considerably.

To use it, add this line to your NAMD input file:

`GPUresident on;`

## Multi-node GPU jobs {#multi_node_gpu_jobs}

### UCX GPU jobs {#ucx_gpu_jobs}

Note that a single GPU node provides a great deal of computational power, so using multiple nodes is justified only when your job can use them efficiently.

This example is for Narval and it assumes that full nodes are used, which gives best performance for NAMD jobs. It uses 8 processes in total on 2 nodes, each process(task) using 12 threads and 1 GPU. This fully utilizes Narval GPU nodes which have 48 cores and 4 GPUs per node. Note that 1 core per task has to be reserved for a communications thread, so NAMD will report that only 88 cores are being used but this is normal.

To use this script on other clusters, please look up the specifications of their available nodes and adjust \--cpus-per-task and \--gpus-per-node options accordingly.

**NOTE**: NAMD 2.14 in this example should not be used on clusters with H100 GPUs. UCX version of NAMD 3.0.1 able to run on H100 is not yet installed on Alliance clusters.

# Performance and benchmarking {#performance_and_benchmarking}

A team at [ACENET](https://www.ace-net.ca/) has created a [Molecular Dynamics Performance Guide](https://mdbench.ace-net.ca/mdbench/) for Alliance clusters. It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on NAMD performance.

Here is an example of how you should conduct benchmarking of NAMD. Performance of NAMD will be different for different systems you are simulating, depending especially on the number of atoms in the simulation. Therefore, if you plan to spend a significant amount of time simulating a particular system, it would be very useful to conduct the kind of benchmarking shown below. Collecting and providing this kind of data is also very useful if you are applying for a RAC award.

For a good benchmark, vary the number of steps so that your system runs for a few minutes, and that timing information is collected in reasonable time intervals of at least a few seconds. If your run is too short, you might see fluctuations in your timing results.

The numbers below were obtained for the standard NAMD apoa1 benchmark. The benchmarking was conducted on the Graham cluster, which has CPU nodes with 32 cores and GPU nodes with 32 cores and 2 GPUs. Performing the benchmark on other clusters will have to take account of the different structure of their nodes.

In the results shown in the first table below, we used NAMD 2.12 from the verbs module. Efficiency is computed from (time with 1 core) / (N \* (time with N cores) ).

  \# cores   Wall time (s) per step   Efficiency
  ---------- ------------------------ ------------
  1          0.8313                   100%
  2          0.4151                   100%
  4          0.1945                   107%
  8          0.0987                   105%
  16         0.0501                   104%
  32         0.0257                   101%
  64         0.0133                   98%
  128        0.0074                   88%
  256        0.0036                   90%
  512        0.0021                   77%
                                      

These results show that for this system it is acceptable to use up to 256 cores. Keep in mind that if you ask for more cores, your jobs will wait in the queue for a longer time, affecting your overall throughput.

Now we perform benchmarking with GPUs. NAMD multicore module is used for simulations that fit within 1 node, and NAMD verbs-smp module is used for runs spanning nodes.

  \# cores   #GPUs   Wall time (s) per step   Notes
  ---------- ------- ------------------------ --------------------
  4          1       0.0165                   1 node, multicore
  8          1       0.0088                   1 node, multicore
  16         1       0.0071                   1 node, multicore
  32         2       0.0045                   1 node, multicore
  64         4       0.0058                   2 nodes, verbs-smp
  128        8       0.0051                   2 nodes, verbs-smp
                                              

From this table it is clear that there is no point at all in using more than 1 node for this system, since performance actually becomes worse if we use 2 or more nodes. Using only 1 node, it is best to use 1GPU/16 core as that has the greatest efficiency, but also acceptable to use 2GPU/32core if you need to get your results quickly. Since on Graham GPU nodes your priority is charged the same for any job using up to 16 cores and 1 GPU, there is no benefit from running with 8 cores and 4 cores in this case.

Finally, you have to ask whether to run with or without GPUs for this simulation. From our numbers we can see that using a full GPU node of Graham (32 cores, 2 gpus) the job runs faster than it would on 4 non-GPU nodes of Graham. Since a GPU node on Graham costs about twice what a non-GPU node costs, in this case it is more cost effective to run with GPUs. You should run with GPUs if possible, however, given that there are fewer GPU than CPU nodes, you may need to consider submitting non-GPU jobs if your waiting time for GPU jobs is too long.

# NAMD 3 {#namd_3}

NAMD 3.0.1 is now installed as a module. It might offer better performance than NAMD 2.14 for certain system configurations.

Sometimes a newer version of NAMD will become available on the NAMD website, but it might take us a while to install it as a module. If you want to try it right away, you can download the binary from the NAMD website and modify it so it can run on Alliance systems, like this (change version as needed):

`tar xvfz NAMD_3.0alpha11_Linux-x86_64-multicore-CUDA-SingleNode.tar.gz `\
`cd NAMD_3.0alpha11_Linux-x86_64-multicore-CUDA`\
`setrpaths.sh  --path .`

After this the `namd3` executable located in that directory will be linked to the correct libraries on our systems. You can then submit a job that uses that executable.

For best performance of NAMD 3 on GPUs, we highly recommend adding the following keyword to the configuration file, if the input configuration you are running supports it.

`GPUresident on;`

Please see the [NAMD 3.0 Alpha web page](https://www.ks.uiuc.edu/Research/namd/alpha/3.0alpha/) for more on this parameter and related changes in NAMD 3.

# References

- Downloads: <http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD> \-- Registration is required to download the software.
- [NAMD User\'s guide for version 3.0.1](https://www.ks.uiuc.edu/Research/namd/3.0.1/ug/)
- [NAMD User\'s guide for version 2.14](https://www.ks.uiuc.edu/Research/namd/2.14/ug/)
- [NAMD version 3.0.1 release notes](http://www.ks.uiuc.edu/Research/namd/3.0.1/notes.html)
- [NAMD version 2.14 release notes](http://www.ks.uiuc.edu/Research/namd/2.14/notes.html)
- Tutorials: <http://www.ks.uiuc.edu/Training/Tutorials/>
