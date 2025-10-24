---
title: "Rorqual/en"
url: "https://docs.alliancecan.ca/wiki/Rorqual/en"
category: "General"
last_modified: "2025-09-09T21:49:06Z"
page_id: 26842
display_title: "Rorqual"
---

`<languages />`{=html}

  --------------------------------------------------------------------------------------------------------------------------------------------------------------
  Availability: June 19, 2025
  Login node: **rorqual.alliancecan.ca**
  Data transfer node (rsync, scp, sftp \...): **rorqual.alliancecan.ca**
  [Automation node](https://docs.alliancecan.ca/Automation_in_the_context_of_multifactor_authentication "Automation node"){.wikilink}: robot.rorqual.alliancecan.ca
  Globus collection: **[alliancecan#rorqual](https://app.globus.org/file-manager?origin_id=f19f13f5-5553-40e3-ba30-6c151b9d35d4)**
  JupyterHub: [jupyterhub.rorqual.alliancecan.ca](https://jupyterhub.rorqual.alliancecan.ca/)
  Portal: [metrix.rorqual.alliancecan.ca](https://metrix.rorqual.alliancecan.ca/)
  Webinar: [slides](https://docs.google.com/presentation/d/1Ah61BBKZIJcn_AeosgUspxRCX_amubPUetZY68SfyXU), [video](https://www.youtube.com/watch?v=lXetzrViI8Q)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------

Rorqual is a heterogeneous and versatile cluster designed for a wide variety of scientific calculations. Built by Dell Canada and CDW Canada, the cluster is located at the [École de technologie supérieure](https://www.etsmtl.ca/en/) in Montreal. Its name recalls the [rorqual](https://en.wikipedia.org/wiki/Rorqual), a marine mammal of which several species can be observed in the St. Lawrence River.

## Access

Each researcher must [request access in CCDB](https://ccdb.alliancecan.ca/me/access_systems), via `<i>`{=html}Resources\--\> Access Systems`</i>`{=html}.

1.  Select *Rorqual* from the list on the left.
2.  Select *I request access*.
3.  Click on the button to accept each of the following agreements
    1.  Calcul Québec Consent for the collection and use of personal information
    2.  Rorqual Service Level Agreement
    3.  Calcul Québec Terms of Use

It can take `<b>`{=html}up to one hour`</b>`{=html} for your access to be enabled.

`<span id="Particularités">`{=html}`</span>`{=html}

## Site-specific policies {#site_specific_policies}

Rorqual\'s compute nodes cannot access the internet. If you need an exception to this rule, contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink} explaining what you need and why.

The `crontab` tool is not offered.

Each job should have a duration of at least one hour (at least five minutes for test jobs) and you cannot have more than 1000 jobs, running or queued, at any given moment. The maximum duration is 7 days (168 hours).

## Storage

+---------------------------+-----------------------------------------------------------------------------------------------------------------+
| HOME\                     | - This small space cannot be increased; for larger storage needs, use the `/project` space                      |
| Lustre filesystem, 116 TB | - Small per user [quotas](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quotas"){.wikilink}       |
|                           | - Daily automatic backup                                                                                        |
+---------------------------+-----------------------------------------------------------------------------------------------------------------+
| SCRATCH\                  | - Accessible via symbolic link `$HOME/links/scratch`                                                            |
| Lustre filesystem, 6.5 PB | - Large space for storing temporary files during computations                                                   |
|                           | - No backup system in place                                                                                     |
|                           | - Large per user [quotas](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quotas"){.wikilink}       |
|                           | - Older files are [ automatically purged](https://docs.alliancecan.ca/Scratch_purging_policy " automatically purged"){.wikilink}            |
+---------------------------+-----------------------------------------------------------------------------------------------------------------+
| PROJECT\                  | - Accessible via symbolic link `$HOME/links/projects/nom-du-projet`                                             |
| Lustre filesystem, 62 PB  | - Designed for sharing data among the members of a research group and for storing large amounts of data         |
|                           | - Large and adjustable per group [quotas](https://docs.alliancecan.ca/Storage_and_file_management#Quotas_et_politiques "quotas"){.wikilink} |
|                           | - Daily backup                                                                                                  |
+---------------------------+-----------------------------------------------------------------------------------------------------------------+

For transferring data via [Globus](https://docs.alliancecan.ca/Globus/fr "Globus"){.wikilink}, use the endpoint specified at the top of this page; for tools like [rsync](https://docs.alliancecan.ca/Transferring_data#Rsync "rsync"){.wikilink} and [scp](https://docs.alliancecan.ca/Transferring_data#SCP "scp"){.wikilink}, please use the login node.

## High-performance interconnect {#high_performance_interconnect}

- InfiniBand interconnect
  - HDR 200Gb/s
  - Maximum blocking factor 34:6 or 5.667:1
  - CPU node island size, up to 31 nodes of 192 cores, fully non-blocking.

`<span id="Caractéristiques_des_nœuds">`{=html}`</span>`{=html}

## Node characteristics {#node_characteristics}

+-------+-------+-------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------+
| nodes | cores | available memory  | storage                      | CPU                                                                                                                                                                        | GPU                         |
+=======+=======+===================+==============================+============================================================================================================================================================================+=============================+
| 670   | 192   | 750G or 768000M   | 1 x SATA SSD, 480G (6Gbit/s) | 2 x [AMD EPYC 9654 (Zen 4)](https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-9004-series/amd-epyc-9654.html) @ 2.40 GHz, 384MB cache L3          |                             |
+-------+       |                   +------------------------------+                                                                                                                                                                            |                             |
| 8     |       |                   | 1 x NVMe SSD, 3.84TB         |                                                                                                                                                                            |                             |
+-------+       +-------------------+------------------------------+                                                                                                                                                                            |                             |
| 8     |       | 3013G or 3086250M | 1 x SATA SSD, 480G (6Gbit/s) |                                                                                                                                                                            |                             |
+-------+-------+-------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------+
| 81    | 64    | 498G or 510000M   | 1 x NVMe SSD, 3.84TB         | 2 x [Intel Xeon Gold 6448Y](https://ark.intel.com/content/www/us/en/ark/products/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz.html) @ 2.10 GHz, 60MB cache L3 | 4 x NVidia H100 SXM5 (80GB) |
+-------+-------+-------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------+

- To get a larger `$SLURM_TMPDIR` space, a job can be submitted with `--tmp=xG`, where `x` is a value between 370 and 3360.

### CPU nodes {#cpu_nodes}

The 192 cores and the different memory spaces are not equidistant, which causes variable delays (of the order of nanoseconds) to access data. In each node, there are

- 2 sockets, each with 12 system memory channels
  - 4 [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes per socket, each connected to 3 system memory channels
    - 3 chiplets per NUMA node, each with its own 32 MiB [L3 cache memory](https://en.wikipedia.org/wiki/CPU_cache)
      - 8 cores per chiplet, each with its own 1 MiB L2 cache memory and 32+32 KiB L1 cache memory

In other words, we have

- groups of 8 closely spaced cores sharing a single L3 cache, which is ideal for [multithreaded parallel programs](https://docs.alliancecan.ca/Running_jobs#Threaded_or_OpenMP_job "multithreaded parallel programs"){.wikilink} (for example, with the `--cpus-per-task=8` option)
- NUMA nodes of 3x8 = 24 cores sharing a trio of system memory channels
- a total of 2x4x3x8 = 192 cores per node

To fully benefit from this topology, full nodes must be reserved (e.g., with `--ntasks-per-node=24 --cpus-per-task=8`) and the place of processes and threads must be explicitly controlled. Depending on the parallel program and the number of cores used, gains can be marginal or significant.

### GPU nodes {#gpu_nodes}

The architecture is not as hierarchical.

- 2 sockets, each with
  - 8 system memory channels
  - 60 MiB L3 cache memory
  - 32 equidistant cores, each each with its own 2 MiB L2 cache memory and 32+48 KiB L1 cache memory
  - 2 NVidia H100 accelerators

The 4 node accelerators are interconnected by [SXM5](https://en.wikipedia.org/wiki/SXM_(socket)).

`<span id="Instances_GPU">`{=html}`</span>`{=html}

### GPU instances {#gpu_instances}

Available GPU instance names are:

+-------------------------------------------------------------------------+-------------------+----------------+----------------+------------------------------------+
| Model or instance                                                       | Short name        | Without unit   | By memory      | Long name                          |
+==============================+==========================================+===================+================+================+====================================+
| `<b>`{=html}GPU`</b>`{=html} | \| `<b>`{=html}H100-80gb`</b>`{=html}    | \| `h100`         | \| `h100`      | \| `h100_80gb` | \| `nvidia_h100_80gb_hbm3`         |
+------------------------------+------------------------------------------+-------------------+----------------+----------------+------------------------------------+
| `<b>`{=html}MIG`</b>`{=html} | \| `<b>`{=html}H100-1g.10gb`</b>`{=html} | \| `h100_1g.10gb` | \| `h100_1.10` | \| `h100_10gb` | \| `nvidia_h100_80gb_hbm3_1g.10gb` |
|                              +------------------------------------------+-------------------+----------------+----------------+------------------------------------+
|                              | `<b>`{=html}H100-2g.20gb`</b>`{=html}    | \| `h100_2g.20gb` | \| `h100_2.20` | \| `h100_20gb` | \| `nvidia_h100_80gb_hbm3_2g.20gb` |
|                              +------------------------------------------+-------------------+----------------+----------------+------------------------------------+
|                              | `<b>`{=html}H100-3g.40gb`</b>`{=html}    | \| `h100_3g.40gb` | \| `h100_3.40` | \| `h100_40gb` | \| `nvidia_h100_80gb_hbm3_3g.40gb` |
+------------------------------+------------------------------------------+-------------------+----------------+----------------+------------------------------------+

To request one or more full H100 GPUs, you need to use one of the following Slurm options:

- `<b>`{=html}One H100-80gb`</b>`{=html} : `--gpus=h100:1` or `--gpus=h100_80gb:1`
- `<b>`{=html}Multiple H100-80gb`</b>`{=html} per node :
  - `--gpus-per-node=h100:2`
  - `--gpus-per-node=h100:3`
  - `--gpus-per-node=h100:4`
- `<b>`{=html} For multiple full H100 GPUs`</b>`{=html} spread anywhere: `--gpus=h100:n` (replace `n` with the number of GPUs you want)

Approximately half of the GPU nodes are configured with [MIG technology](https://docs.alliancecan.ca/Multi-Instance_GPU "MIG technology"){.wikilink}, and only 3 GPU instance sizes are available:

- `<b>`{=html}H100-1g.10gb`</b>`{=html}: 1/8^th^ of the computing power with 10GB GPU memory
- `<b>`{=html}H100-2g.20gb`</b>`{=html}: 2/8^th^ of the computing power with 20GB GPU memory
- `<b>`{=html}H100-3g.40gb`</b>`{=html}: 3/8^th^ of the computing power with 40GB GPU memory

To request `<b>`{=html}one and only one GPU instance`</b>`{=html} for your compute job, use the corresponding option:

- `<b>`{=html}H100-1g.10gb`</b>`{=html} : `--gpus=h100_1g.10gb:1`
- `<b>`{=html}H100-2g.20gb`</b>`{=html} : `--gpus=h100_2g.20gb:1`
- `<b>`{=html}H100-3g.40gb`</b>`{=html} : `--gpus=h100_3g.40gb:1`

The maximum recommended number of `<b>`{=html}CPU cores and system memory`</b>`{=html} per GPU instance is listed in [this table](https://docs.alliancecan.ca/Allocations_and_compute_scheduling#Ratios_in_bundles "this table"){.wikilink}.
