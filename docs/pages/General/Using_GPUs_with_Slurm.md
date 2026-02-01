---
title: "Using GPUs with Slurm/en"
url: "https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en"
category: "General"
last_modified: "2026-01-31T13:12:51Z"
page_id: 4369
display_title: "Using GPUs with Slurm"
---

= Introduction =

To request one or more GPUs for a Slurm job, use this form:
  --gpus-per-node=:

For example:
  --gpus-per-node=a100:1

This requests a single A100 GPU (unless you also use --nodes to specify more than a single node).
See the following section, Available GPUs, for valid models.

The following form can also be used:
  --gres=gpu::
This form may not be supported in the future.  We recommend that you replace it in your scripts with --gpus-per-node.

Slurm supports a variety of other directives that you can use to request GPU resources: --gpus, --gpus-per-socket, --gpus-per-task, --mem-per-gpu, and --ntasks-per-gpu.  Please see the Slurm documentation for sbatch for more about these.  Our staff do not test all of these; if you try one but don't get the result you expect, contact technical support.

For general advice on job scheduling, see Running jobs.

= Available GPUs =
The following table summarizes the available GPU models:

Cluster  	GPU model   	Model specifiersfor Slurm    	Notes
Fir      	H100-80gb   	h100
Fir      	H100-80gb   	nvidia_h100_80gb_hbm3_1g.10gb	MIG
Fir      	H100-80gb   	nvidia_h100_80gb_hbm3_2g.20gb	MIG
Fir      	H100-80gb   	nvidia_h100_80gb_hbm3_3g.40gb	MIG
Narval   	A100-40gb   	a100
Narval   	A100-40gb   	a100_1g.5gb                  	MIG
Narval   	A100-40gb   	a100_2g.10gb                 	MIG
Narval   	A100-40gb   	a100_3g.20gb                 	MIG
Narval   	A100-40gb   	a100_4g.20gb                 	MIG
Nibi     	H100-80gb   	h100
Nibi     	H100-80gb   	nvidia_h100_80gb_hbm3_1g.10gb	MIG
Nibi     	H100-80gb   	nvidia_h100_80gb_hbm3_2g.20gb	MIG
Nibi     	H100-80gb   	nvidia_h100_80gb_hbm3_3g.40gb	MIG
Nibi     	MI300A-128gb	(none; see Nibi)
Rorqual  	H100-80gb   	h100
Rorqual  	H100-80gb   	nvidia_h100_80gb_hbm3_1g.10gb	MIG; synonyms h100_1g.10gb, h100_1.10, h100_10gb
Rorqual  	H100-80gb   	nvidia_h100_80gb_hbm3_2g.20gb	MIG; synonyms h100_2g.20gb, h100_2.20, h100_20gb
Rorqual  	H100-80gb   	nvidia_h100_80gb_hbm3_3g.40gb	MIG; synonyms h100_3g.40gb, h100_3.40, h100_40gb
Trillium 	H100-80gb   	h100
Killarney	H100-80gb   	h100                         	 
Killarney	L40S-48gb   	l40s                         	 
tamIA    	H100-80gb   	h100                         	 
tamIA    	H200        	h200                         	 
Vulcan   	L40S-48gb   	l40s                         	 
Vulcan

GPU models (including MIGs models) available on any given cluster can be obtained from Slurm with the following command.
This may be useful if the table above has not been updated with the latest changes.

grep gpused 's/gpu://g'sed 's/),/\n/g'cut -d: -f1sortuniq}}

There are short synonyms available for some of the MIG models at certain sites; this command will not provide those synonyms.
Also, the presence of a GPU model does not guarantee that you will be able to use that model in your jobs; there may be
further restrictions on what models are available based on (for example) which to research group you belong.
For further information see the site-specific page by clicking on the cluster name in the above table, or contact support.

If you do not supply a model specifier your job may be rejected or it may be sent to an arbitrary GPU model.
There are very few programs which can use an arbitrary GPU efficiently,
so we strongly recommend that you always request a specific GPU model.

There are GPUs available at Arbutus, but like other cloud resources they cannot be scheduled via Slurm.
See Cloud resources for more details.

== Multi-Instance GPUs (MIGs) ==
MIG is a technology that partitions a GPU into multiple instances.
Your jobs might be able to use a MIG instance instead of a whole GPU.
Please see Multi-Instance_GPU for more about this.

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

On Narval and Rorqual, profiling is possible but requires disabling the
NVIDIA Data Center GPU Manager (DCGM). This must be done during job submission by setting the DISABLE_DCGM environment variable:

1 salloc --accountdef-someuser --gpus-per-node1 --mem4000M --time03:00}}

Then, in your interactive job, wait until DCGM is disabled on the node:
 grep 'Hostengine build info:')" ]; do  sleep 5; done}}

Finally, launch your profiler. For more details on profilers, see Debugging and profiling.

On Fir and Nibi, GPU profiling like the above technique is not available yet.

= See also =
CUDA
Multi-Instance GPU
Running jobs