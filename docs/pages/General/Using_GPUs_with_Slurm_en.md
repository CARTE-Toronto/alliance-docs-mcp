---
title: "Using GPUs with Slurm/en"
url: "https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en"
category: "General"
last_modified: "2025-09-10T20:39:53Z"
page_id: 4369
display_title: "Using GPUs with Slurm"
---

`<languages />`{=html}

# Introduction

To request one or more GPUs for a Slurm job, use this form:

` --gpus-per-node=[type:]number`

The square-bracket notation means that you must specify the number of GPUs, and you may optionally specify the GPU type. Valid types are listed in the `<i>`{=html}Available GPUs`</i>`{=html} table below, in the column headed \"Slurm type specifier\". Here are two examples:

` --gpus-per-node=2`\
` --gpus-per-node=v100:1`

The first line requests two GPUs per node, of any type available on the cluster. The second line requests one GPU per node, with the GPU being of the V100 type.

The following form can also be used:

` --gres=gpu[[:type]:number]`

This is older, and we expect it will no longer be supported in some future release of Slurm. We recommend that you replace it in your scripts with the above `--gpus-per-node` form.

There are a variety of other directives that you can use to request GPU resources: `--gpus`, `--gpus-per-socket`, `--gpus-per-task`, `--mem-per-gpu`, and `--ntasks-per-gpu`. Please see the Slurm documentation for [sbatch](https://slurm.schedmd.com/sbatch.html) for more about these. Our staff did not test all the combinations; if you don\'t get the result you expect, [contact technical support](https://docs.alliancecan.ca/Technical_support "contact technical support"){.wikilink}.

For general advice on job scheduling, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

# Available GPUs {#available_gpus}

These are the GPUs currently available:

+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Cluster  | Specifications                                                                           | Slurm type\                                                                                          | GPU model                              | Compute\                               | Notes                                                  |
|          |                                                                                          | specifier                                                                                            |                                        | Capability(\*)                         |                                                        |
+==========+==========================================================================================+======================================================================================================+========================================+========================================+========================================================+
| Fir      | [`<b>`{=html}Details`</b>`{=html}](https://docs.alliancecan.ca/Fir#Node_characteristics "Details"){.wikilink}        | [`<b>`{=html}Options`</b>`{=html}](https://docs.alliancecan.ca/Multi-Instance_GPU#Available_configurations "Options"){.wikilink} | \| H100-80gb                           | 90                                     | Two GPUs per CPU socket; all GPUs connected via NVLink |
+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Narval   | [`<b>`{=html}Details`</b>`{=html}](https://docs.alliancecan.ca/Narval/en#Node_characteristics "Details"){.wikilink}  | \| [`<b>`{=html}Options`</b>`{=html}](https://docs.alliancecan.ca/Narval/en#GPU_instances "Options"){.wikilink}                  | A100-40gb                              | 80                                     | Two GPUs per CPU socket; all GPUs connected via NVLink |
+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Nibi     | [`<b>`{=html}Details`</b>`{=html}](https://docs.alliancecan.ca/Nibi#Node_characteristics "Details"){.wikilink}       | [`<b>`{=html}Options`</b>`{=html}](https://docs.alliancecan.ca/Multi-Instance_GPU#Available_configurations "Options"){.wikilink} | \| H100-80gb                           | 90                                     | Two GPUs per CPU socket; all GPUs connected via NVLink |
|          |                                                                                          |                                                                                                      +----------------------------------------+----------------------------------------+--------------------------------------------------------+
|          |                                                                                          |                                                                                                      | \| MI300A-128gb                        | N.A.                                   | Unified memory between CPU and GPU                     |
+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Rorqual  | [`<b>`{=html}Details`</b>`{=html}](https://docs.alliancecan.ca/Rorqual/en#Node_characteristics "Details"){.wikilink} | \| [`<b>`{=html}Options`</b>`{=html}](https://docs.alliancecan.ca/Rorqual/en#GPU_instances "Options"){.wikilink}                 | H100-80gb                              | 90                                     | Two GPUs per CPU socket; all GPUs connected via NVLink |
+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Trillium | [`<b>`{=html}Details`</b>`{=html}](https://docs.alliancecan.ca/Trillium#Node_characteristics "Details"){.wikilink}   | \| [`<b>`{=html}Options`</b>`{=html}](https://docs.alliancecan.ca/Trillium#GPU_instances "Options"){.wikilink}                   | H100-80gb                              | 90                                     | Two GPUs per CPU socket; all GPUs connected via NVLink |
+----------+------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+----------------------------------------+----------------------------------------+--------------------------------------------------------+
| Arbutus  | Cloud resources are not schedulable via Slurm. See [Cloud resources](https://docs.alliancecan.ca/Cloud_resources "Cloud resources"){.wikilink} for details of available hardware.                                                                                                                             |                                                        |
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------+

(\*) `<b>`{=html}Compute Capability`</b>`{=html} is a technical term created by NVIDIA as a compact way to describe what hardware functions are available on some models of GPU and not on others. It is not a measure of performance and is relevant only if you are compiling your own GPU programs. See the page on [CUDA programming](https://docs.alliancecan.ca/CUDA#.22Compute_Capability.22 "CUDA programming"){.wikilink} for more.

## Multi-Instance GPUs (MIGs) {#multi_instance_gpus_migs}

MIG, a technology that allows to partition a GPU into multiple instances. Please see [Multi-Instance_GPU](https://docs.alliancecan.ca/Multi-Instance_GPU "Multi-Instance_GPU"){.wikilink}.

# Selecting the type of GPU to use {#selecting_the_type_of_gpu_to_use}

Some clusters have more than one GPU type available, and some clusters only have GPUs on certain nodes.

If you do not supply a type specifier, Slurm may send your job to a node equipped with any type of GPU. For certain workflows this may be undesirable; for example, molecular dynamics code requires high double-precision performance, for which T4 GPUs are not appropriate. In such a case, make sure you include a type specifier.

# Requesting CPU cores and system memory {#requesting_cpu_cores_and_system_memory}

Along with each GPU instance, your job should have a number of CPU cores (default is `1`) and some amount of system memory. The recommended maximum numbers of CPU cores and gigabytes of system memory per GPU instance are listed in the [table of bundle characteristics](https://docs.alliancecan.ca/Allocations_and_compute_scheduling#Ratios_in_bundles "table of bundle characteristics"){.wikilink}.

# Examples

## Single-core job {#single_core_job}

If you need only a single CPU core and one GPU:

## Multi-threaded job {#multi_threaded_job}

For a GPU job which needs multiple CPUs in a single node:

For each GPU requested, we recommend

- on Fir, no more than 12 CPU cores;
- on Narval, no more than 12 CPU cores
- on Nibi, no more than 14 CPU cores,
- on Rorqual, no more than 16 CPU cores

## MPI job {#mpi_job}

## Whole nodes {#whole_nodes}

If your application can efficiently use an entire node and its associated GPUs, you will probably experience shorter wait times if you ask Slurm for a whole node. Use one of the following job scripts as a template.

### Packing single-GPU jobs within one SLURM job {#packing_single_gpu_jobs_within_one_slurm_job}

If you need to run four single-GPU programs or two 2-GPU programs for longer than 24 hours, [GNU Parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU Parallel"){.wikilink} is recommended. A simple example is:

    cat params.input | parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python {} &> {#}.out'

In this example, the GPU ID is calculated by subtracting 1 from the slot ID {%} and {#} is the job ID, starting from 1.

A `params.input` file should include input parameters in each line, like this:

    code1.py
    code2.py
    code3.py
    code4.py
    ...

With this method, you can run multiple tasks in one submission. The `-j4` parameter means that GNU Parallel can run a maximum of four concurrent tasks, launching another as soon as one ends. CUDA_VISIBLE_DEVICES is used to ensure that two tasks do not try to use the same GPU at the same time.

## Profiling GPU tasks {#profiling_gpu_tasks}

On Fir and Nibi, GPU profiling is not available since performance counters are not accessible.

On [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink} and [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}, profiling is possible but requires disabling the [NVIDIA Data Center GPU Manager (DCGM)](https://developer.nvidia.com/dcgm). This must be done during job submission by setting the `DISABLE_DCGM` environment variable:

1 salloc \--accountdef-someuser \--gpus-per-node1 \--mem4000M \--time03:00}}

Then, in your interactive job, wait until DCGM is disabled on the node:

`grep 'Hostengine build info:')" ]; do  sleep 5; done}}`

Finally, launch your profiler. For more details on profilers, see [Debugging and profiling](https://docs.alliancecan.ca/Debugging_and_profiling "Debugging and profiling"){.wikilink}.
