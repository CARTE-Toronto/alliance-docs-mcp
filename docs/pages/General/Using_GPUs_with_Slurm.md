---
title: "Using GPUs with Slurm/en"
url: "https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en"
category: "General"
last_modified: "2025-09-10T20:39:53Z"
page_id: 4369
display_title: "Using GPUs with Slurm"
---

= Introduction =

To request one or more GPUs for a Slurm job, use this form:
  --gpus-per-node=[type:]number

The square-bracket notation means that you must specify the number of GPUs, and you may optionally specify the GPU type.  Valid types are listed in the Available GPUs table below, in the column headed "Slurm type specifier".  Here are two examples:
  --gpus-per-node=2
  --gpus-per-node=v100:1

The first line requests two GPUs per node, of any type available on the cluster.  The second line requests one GPU per node, with the GPU being of the V100 type.

The following form can also be used:
  --gres=gpu[[:type]:number]
This is older, and we expect it will no longer be supported in some future release of Slurm.  We recommend that you replace it in your scripts with the above --gpus-per-node form.

There are a variety of other directives that you can use to request GPU resources: --gpus, --gpus-per-socket, --gpus-per-task, --mem-per-gpu, and --ntasks-per-gpu.  Please see the Slurm documentation for sbatch for more about these.  Our staff did not test all the combinations; if you don't get the result you expect, contact technical support.

For general advice on job scheduling, see Running jobs.

= Available GPUs =
These are the GPUs currently available:

Cluster 	Specifications                                                                                       	Slurm typespecifier                                                                                  	GPU model                                                                                            	ComputeCapability(*)                                                                                 	Notes
Fir     	Details                                                                                              	Options                                                                                              	H100-80gb                                                                                            	90                                                                                                   	Two GPUs per CPU socket; all GPUs connected via NVLink
Narval  	Details                                                                                              	Options                                                                                              	A100-40gb                                                                                            	80                                                                                                   	Two GPUs per CPU socket; all GPUs connected via NVLink
Nibi    	Details                                                                                              	Options                                                                                              	H100-80gb                                                                                            	90                                                                                                   	Two GPUs per CPU socket; all GPUs connected via NVLink
Nibi    	Details                                                                                              	Options                                                                                              	MI300A-128gb                                                                                         	N.A.                                                                                                 	Unified memory between CPU and GPU
Rorqual 	Details                                                                                              	Options                                                                                              	H100-80gb                                                                                            	90                                                                                                   	Two GPUs per CPU socket; all GPUs connected via NVLink
Trillium	Details                                                                                              	Options                                                                                              	H100-80gb                                                                                            	90                                                                                                   	Two GPUs per CPU socket; all GPUs connected via NVLink
Arbutus 	Cloud resources are not schedulable via Slurm. See Cloud resources for details of available hardware.	Cloud resources are not schedulable via Slurm. See Cloud resources for details of available hardware.	Cloud resources are not schedulable via Slurm. See Cloud resources for details of available hardware.	Cloud resources are not schedulable via Slurm. See Cloud resources for details of available hardware.

(*) Compute Capability is a technical term created by NVIDIA as a compact way to describe what hardware functions are available on some models of GPU and not on others.
It is not a measure of performance and is relevant only if you are compiling your own GPU programs.  See the page on CUDA programming for more.

== Multi-Instance GPUs (MIGs) ==
MIG, a technology that allows to partition a GPU into multiple instances. Please see Multi-Instance_GPU.

= Selecting the type of GPU to use =

Some clusters have more than one GPU type available, and some clusters only have GPUs on certain nodes.

If you do not supply a type specifier, Slurm may send your job to a node equipped with any type of GPU.
For certain workflows this may be undesirable; for example, molecular dynamics code requires high double-precision performance, for which T4 GPUs are not appropriate.
In such a case, make sure you include a type specifier.

= Requesting CPU cores and system memory =

Along with each GPU instance, your job should have a number of CPU cores (default is 1) and some amount of system memory. The recommended maximum numbers of CPU cores and gigabytes of system memory per GPU instance are listed in the table of bundle characteristics.

= Examples =

== Single-core job ==
If you need only a single CPU core and one GPU:

== Multi-threaded job ==
For a GPU job which needs multiple CPUs in a single node:

For each GPU requested, we recommend
* on Fir, no more than 12 CPU cores;
* on Narval, no more than 12 CPU cores
* on Nibi, no more than 14 CPU cores,
* on Rorqual, no more than 16 CPU cores

== MPI job ==

== Whole nodes ==
If your application can efficiently use an entire node and its associated GPUs, you will probably experience shorter wait times if you ask Slurm for a whole node. Use one of the following job scripts as a template.

===Packing single-GPU jobs within one SLURM job===

If you need to run four single-GPU programs or two 2-GPU programs for longer than 24 hours, GNU Parallel is recommended. A simple example is:

cat params.input | parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python {} &> {#}.out'

In this example, the GPU ID is calculated by subtracting 1 from the slot ID {%} and {#} is the job ID, starting from 1.

A params.input file should include input parameters in each line, like this:

code1.py
code2.py
code3.py
code4.py
...

With this method, you can run multiple tasks in one submission. The -j4 parameter means that GNU Parallel can run a maximum of four concurrent tasks, launching another as soon as one ends. CUDA_VISIBLE_DEVICES is used to ensure that two tasks do not try to use the same GPU at the same time.

== Profiling GPU tasks ==

On Fir and Nibi, GPU profiling is not available since performance counters are not accessible.

On Narval and Rorqual, profiling is possible but requires disabling the
NVIDIA Data Center GPU Manager (DCGM). This must be done during job submission by setting the DISABLE_DCGM environment variable:

1 salloc --accountdef-someuser --gpus-per-node1 --mem4000M --time03:00}}

Then, in your interactive job, wait until DCGM is disabled on the node:
 grep 'Hostengine build info:')" ]; do  sleep 5; done}}

Finally, launch your profiler. For more details on profilers, see Debugging and profiling.