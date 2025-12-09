---
title: "Narval/en"
url: "https://docs.alliancecan.ca/wiki/Narval/en"
category: "General"
last_modified: "2025-09-08T18:50:35Z"
page_id: 17288
display_title: "Narval"
---

Availability: since October 2021
Login node: narval.alliancecan.ca
Globus collection: Compute Canada - Narval
Data transfer node (rsync, scp, sftp,...): narval.alliancecan.ca
Portal : https://portail.narval.calculquebec.ca/

Narval is a general purpose cluster designed for a variety of workloads. Built by Dell Canada and CDW Canada it is located at the École de technologie supérieure in Montreal. The cluster is named in honour of the narwhal, a species of whale which has occasionally been observed in the Gulf of St. Lawrence.

==Site-specific policies==

By policy, Narval's compute nodes cannot access the internet. If you need an exception to this rule, contact technical support explaining what you need and why.

Crontab is not offered on Narval.

Each job on Narval should have a duration of at least one hour (five minutes for test jobs) and you cannot have more than 1000 jobs, running or queued, at any given moment. The maximum duration for a job on Narval is 7 days (168 hours).

==Storage==

HOME  Lustre filesystem, 64 TB of space

*Location of home directories, each of which has a small fixed quota.
*You should use the project space for larger storage needs.
*Small per user  quota.

*There is a daily backup of the home directories.
SCRATCH  Lustre filesystem, 5.7 PB of space

*Large space for storing temporary files during computations.
*No backup system in place.

*Large quota per user.

*There is an  automated purge of older files in this space.
PROJECT  Lustre filesystem, 35 PB of space

*This space is designed for sharing data among the members of a research group and for storing large amounts of data.

*Large and adjustable per group quota.

*There is a daily backup of the project space.

For transferring data via Globus, you should use the endpoint specified at the top of this page, while for tools like rsync and scp you can use a login node.

==High-performance interconnect==

The InfiniBand Mellanox HDR network links together all of the nodes of the cluster. Each hub of 40 HDR ports (200 Gb/s) can connect up to 66 nodes with HDR100 (100 Gb/s) with 33 HDR links divided in two (2) by special cables. The seven (7) remaining HDR links allow the hub to be connected to a rack containing the seven (7) central HDR InfiniBand hubs. The islands of nodes are therefore connected by a maximum blocking factor of 33:7 (4.7:1). In contrast, the storage servers are connected by a much lower blocking factor in order to maximize the performance.

In practice the Narval racks contain islands of 48 or 56 regular CPU nodes. It is therefore possible to run parallel jobs using up to 3584 cores with a non-blocking network. For larger jobs or ones which are distributed in a fragmented manner across the network, the blocking factor is 4.7:1. The interconnect remains a high-performance one nonetheless.

==Node characteristics==

nodes	cores	available memory 	CPU                                                	storage           	GPU
1145 	64   	249G or 255000M  	2 x AMD EPYC 7532 (Zen 2) @ 2.40 GHz, 256M cache L3	1 x 960G SSD
33   	64   	2009G or 2057500M	2 x AMD EPYC 7532 (Zen 2) @ 2.40 GHz, 256M cache L3	1 x 960G SSD
3    	64   	4000G or 4096000M	2 x AMD EPYC 7502 (Zen 2) @ 2.50 GHz, 128M cache L3	1 x 960G SSD
159  	48   	498G or 510000M  	2 x AMD EPYC 7413 (Zen 3) @ 2.65 GHz, 128M cache L3	1 x SSD of 3.84 TB	4 x NVidia A100SXM4 (40 GB memory), connected via NVLink

===AMD processors===
==== Supported instructions sets ====
Narval is equipped with 2nd and 3rd generation AMD EPYC processors which support the AVX2 instruction set.

Narval does not however support the AVX512 instruction set, in contrast to the nodes of more recent clusters.

====Intel compilers====
Intel compilers can compile applications for Narval's AMD processors with AVX2 and earlier instruction sets. Use the -march=core-avx2 option to produce executables which are compatible with both Intel and AMD processors.

However, if you have compiled a program on a system which uses Intel processors and you have used one or more options like -xXXXX, such as -xCORE-AVX2, the compiled program will not work on Narval because the Intel compilers add additional instructions in order to verify that processor used is an Intel product. On Narval, the options -xHOST and -march=native are equivalent to -march=pentium (the old 1993 Pentium) and should not be used.

====Software environments====
StdEnv/2023 is the standard software environment on Narval; previous versions (2016 and 2018) have been blocked intentionally. If you need an application only available with an older standard environment, please write to Technical support.

====BLAS and LAPACK libraries====
The Intel MKL library works with AMD processors, although not in an optimal way. We now favour the use of the FlexiBLAS library. For more details, please consult the page on BLAS and LAPACK.

===GPU instances===

To request one or more full A100 GPUs, you need to use one of the following Slurm options:
* One A100-40gb : --gpus=a100:1
* Multiple A100-40gb per node:
** --gpus-per-node=a100:2
** --gpus-per-node=a100:3
** --gpus-per-node=a100:4
*  For multiple full A100 GPUs spread anywhere: --gpus=a100:n (replace n with the number of GPUs you want)

Several GPU nodes are configured with Multi-Instance GPU technology. Four sizes are available:

* 1g.5gb : 1/8 compute capacity, GPU memory 5 GB
* 2g.10gb : 2/8 compute capacity, GPU memory 10 GB
* 3g.20gb : 3/8 compute capacity, GPU memory 20 GB
* 4g.20gb : 4/8 compute capacity, GPU memory 20 GB (fewer of this version are available)

To request one and only one GPU instance for your compute job, options are :

* 1g.5gb : --gpus=a100_1g.5gb:1
* 2g.10gb : --gpus=a100_2g.10gb:1
* 3g.20gb : --gpus=a100_3g.20gb:1
* 4g.20gb : --gpus=a100_4g.20gb:1

The maximum recommended number of CPU cores and system memory per GPU instance is listed in this table.

==Monitoring jobs==

From the Narval portal, you can monitor your jobs using CPUs and GPUs in real time or examine jobs that have run in the past. This can help you to optimize resource usage and shorten wait time in the queue.

You can monitor your usage of
* compute nodes,
* memory,
* GPU.

It is important that you use the allocated resources and to correct your requests when compute resources are less used or not used at all. For example, if you request 4 cores (CPUs) but use only one, you should adjust the script file accordingly.