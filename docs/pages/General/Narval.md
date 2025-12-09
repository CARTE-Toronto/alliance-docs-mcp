---
title: "Narval/en"
url: "https://docs.alliancecan.ca/wiki/Narval/en"
category: "General"
last_modified: "2025-09-08T18:50:35Z"
page_id: 17288
display_title: "Narval"
---

`<languages />`{=html}

  --------------------------------------------------------------------------------------------------------------------------------------
  Availability: since October 2021
  Login node: **narval.alliancecan.ca**
  Globus collection: **[Compute Canada - Narval](https://app.globus.org/file-manager?origin_id=a1713da6-098f-40e6-b3aa-034efe8b6e5b)**
  Data transfer node (rsync, scp, sftp,\...): **narval.alliancecan.ca**
  Portal : <https://portail.narval.calculquebec.ca/>
  --------------------------------------------------------------------------------------------------------------------------------------

Narval is a general purpose cluster designed for a variety of workloads. Built by Dell Canada and CDW Canada it is located at the [École de technologie supérieure](https://www.etsmtl.ca/en/home) in Montreal. The cluster is named in honour of the [narwhal](https://en.wikipedia.org/wiki/Narwhal), a species of whale which has occasionally been observed in the Gulf of St. Lawrence.

## Site-specific policies {#site_specific_policies}

By policy, Narval\'s compute nodes cannot access the internet. If you need an exception to this rule, contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink} explaining what you need and why.

Crontab is not offered on Narval.

Each job on Narval should have a duration of at least one hour (five minutes for test jobs) and you cannot have more than 1000 jobs, running or queued, at any given moment. The maximum duration for a job on Narval is 7 days (168 hours).

## Storage

+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| HOME\                              | - Location of home directories, each of which has a small fixed quota.                                                 |
| Lustre filesystem, 64 TB of space  | - You should use the `project` space for larger storage needs.                                                         |
|                                    | - Small per user [ quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies " quota"){.wikilink}.             |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is a daily backup of the home directories.                                                                     |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| SCRATCH\                           | - Large space for storing temporary files during computations.                                                         |
| Lustre filesystem, 5.7 PB of space | - No backup system in place.                                                                                           |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - Large [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per user.               |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is an [ automated purge](https://docs.alliancecan.ca/Scratch_purging_policy " automated purge"){.wikilink} of older files in this space.   |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| PROJECT\                           | - This space is designed for sharing data among the members of a research group and for storing large amounts of data. |
| Lustre filesystem, 35 PB of space  |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - Large and adjustable per group [quota](https://docs.alliancecan.ca/Storage_and_file_management/fr#Quotas_et_politiques "quota"){.wikilink}.      |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is a daily backup of the project space.                                                                        |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+

For transferring data via [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink}, you should use the endpoint specified at the top of this page, while for tools like [rsync](https://docs.alliancecan.ca/Transferring_data#Rsync "rsync"){.wikilink} and [scp](https://docs.alliancecan.ca/Transferring_data#SCP "scp"){.wikilink} you can use a login node.

`<span id="Réseautique_haute_performance">`{=html}`</span>`{=html}

## High-performance interconnect {#high_performance_interconnect}

The [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand) [Mellanox HDR](https://www.nvidia.com/en-us/networking/infiniband/qm8700/) network links together all of the nodes of the cluster. Each hub of 40 HDR ports (200 Gb/s) can connect up to 66 nodes with HDR100 (100 Gb/s) with 33 HDR links divided in two (2) by special cables. The seven (7) remaining HDR links allow the hub to be connected to a rack containing the seven (7) central HDR InfiniBand hubs. The islands of nodes are therefore connected by a maximum blocking factor of 33:7 (4.7:1). In contrast, the storage servers are connected by a much lower blocking factor in order to maximize the performance.

In practice the Narval racks contain islands of 48 or 56 regular CPU nodes. It is therefore possible to run parallel jobs using up to 3584 cores with a non-blocking network. For larger jobs or ones which are distributed in a fragmented manner across the network, the blocking factor is 4.7:1. The interconnect remains a high-performance one nonetheless.

`<span id="Caractéristiques_des_nœuds">`{=html}`</span>`{=html}

## Node characteristics {#node_characteristics}

+-------+-------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+----------------------------------------------------------+
| nodes | cores | available memory  | CPU                                                                                                                                                              | storage            | GPU                                                      |
+=======+=======+===================+==================================================================================================================================================================+====================+==========================================================+
| 1145  | 64    | 249G or 255000M   | 2 x [AMD EPYC 7532 (Zen 2)](https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7532.html) @ 2.40 GHz, 256M cache L3 | 1 x 960G SSD       |                                                          |
+-------+       +-------------------+                                                                                                                                                                  |                    |                                                          |
| 33    |       | 2009G or 2057500M |                                                                                                                                                                  |                    |                                                          |
+-------+       +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+                    |                                                          |
| 3     |       | 4000G or 4096000M | 2 x [AMD EPYC 7502 (Zen 2)](https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7502.html) @ 2.50 GHz, 128M cache L3 |                    |                                                          |
+-------+-------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+----------------------------------------------------------+
| 159   | 48    | 498G or 510000M   | 2 x [AMD EPYC 7413 (Zen 3)](https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7003-series/amd-epyc-7413.html) @ 2.65 GHz, 128M cache L3 | 1 x SSD of 3.84 TB | 4 x NVidia A100SXM4 (40 GB memory), connected via NVLink |
+-------+-------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+----------------------------------------------------------+

### AMD processors {#amd_processors}

#### Supported instructions sets {#supported_instructions_sets}

Narval is equipped with 2nd and 3rd generation AMD EPYC processors which support the [AVX2 instruction set](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2).

Narval does not however support the [AVX512](https://en.wikipedia.org/wiki/AVX-512) instruction set, in contrast to the nodes of more recent clusters.

#### Intel compilers {#intel_compilers}

Intel compilers can compile applications for Narval\'s AMD processors with AVX2 and earlier instruction sets. Use the `-march=core-avx2` option to produce executables which are compatible with both Intel and AMD processors.

However, if you have compiled a program on a system which uses Intel processors and you have used one or more options like `-xXXXX`, such as `-xCORE-AVX2`, the compiled program will not work on Narval because the Intel compilers add additional instructions in order to verify that processor used is an Intel product. On Narval, the options `-xHOST` and `-march=native` are equivalent to `-march=pentium` (the old 1993 Pentium) and should `<b>`{=html}not`</b>`{=html} be used.

#### Software environments {#software_environments}

[StdEnv/2023](https://docs.alliancecan.ca/Standard_software_environments "StdEnv/2023"){.wikilink} is the standard software environment on Narval; previous versions (2016 and 2018) have been blocked intentionally. If you need an application only available with an older standard environment, please write to [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.

#### BLAS and LAPACK libraries {#blas_and_lapack_libraries}

The Intel MKL library works with AMD processors, although not in an optimal way. We now favour the use of the FlexiBLAS library. For more details, please consult the page on [BLAS and LAPACK](https://docs.alliancecan.ca/BLAS_and_LAPACK "BLAS and LAPACK"){.wikilink}.

`<span id="Instances_GPU">`{=html}`</span>`{=html}

### GPU instances {#gpu_instances}

To request one or more full A100 GPUs, you need to use one of the following Slurm options:

- `<b>`{=html}One A100-40gb`</b>`{=html} : `--gpus=a100:1`
- `<b>`{=html}Multiple A100-40gb`</b>`{=html} per node:
  - `--gpus-per-node=a100:2`
  - `--gpus-per-node=a100:3`
  - `--gpus-per-node=a100:4`
- `<b>`{=html} For multiple full A100 GPUs`</b>`{=html} spread anywhere: `--gpus=a100:n` (replace `n` with the number of GPUs you want)

Several GPU nodes are configured with [Multi-Instance GPU technology](https://docs.alliancecan.ca/Multi-Instance_GPU "Multi-Instance GPU technology"){.wikilink}. Four sizes are available:

- `<b>`{=html}1g.5gb`</b>`{=html} : 1/8 compute capacity, GPU memory 5 GB
- `<b>`{=html}2g.10gb`</b>`{=html} : 2/8 compute capacity, GPU memory 10 GB
- `<b>`{=html}3g.20gb`</b>`{=html} : 3/8 compute capacity, GPU memory 20 GB
- `<b>`{=html}4g.20gb`</b>`{=html} : 4/8 compute capacity, GPU memory 20 GB (fewer of this version are available)

To request `<b>`{=html}one and only one`</b>`{=html} GPU instance for your compute job, options are :

- `<b>`{=html}1g.5gb`</b>`{=html} : `--gpus=a100_1g.5gb:1`
- `<b>`{=html}2g.10gb`</b>`{=html} : `--gpus=a100_2g.10gb:1`
- `<b>`{=html}3g.20gb`</b>`{=html} : `--gpus=a100_3g.20gb:1`
- `<b>`{=html}4g.20gb`</b>`{=html} : `--gpus=a100_4g.20gb:1`

The maximum recommended number of `<b>`{=html}CPU cores and system memory`</b>`{=html} per GPU instance is listed in [this table](https://docs.alliancecan.ca/Allocations_and_compute_scheduling#Ratios_in_bundles "this table"){.wikilink}.

`<span id="Suivi_de_vos_tâches">`{=html}`</span>`{=html}

## Monitoring jobs {#monitoring_jobs}

From the [Narval portal](https://portail.narval.calculquebec.ca/), you can monitor your jobs using CPUs and GPUs `<b>`{=html}in real time`</b>`{=html} or examine jobs that have run in the past. This can help you to optimize resource usage and shorten wait time in the queue.

You can monitor your usage of

- compute nodes,
- memory,
- GPU.

It is important that you use the allocated resources and to correct your requests when compute resources are less used or not used at all. For example, if you request 4 cores (CPUs) but use only one, you should adjust the script file accordingly.
