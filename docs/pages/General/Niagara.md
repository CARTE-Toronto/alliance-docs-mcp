---
title: "Niagara/en"
url: "https://docs.alliancecan.ca/wiki/Niagara/en"
category: "General"
last_modified: "2025-09-10T18:38:17Z"
page_id: 4990
display_title: "Niagara"
---

Availability: In production since April 2018
Login node: niagara.alliancecan.ca
Globus collection: alliancecan#niagara
Data mover nodes (rsync, scp, ...): nia-dm2, nia-dm2, see Moving data
System Status Page: https://docs.scinet.utoronto.ca
Portal : https://my.scinet.utoronto.ca

Niagara is a homogeneous cluster, owned by the University of Toronto and operated by SciNet, intended to enable large parallel jobs of 1040 cores and more. It was designed to optimize throughput of a range of
scientific codes running at scale, energy efficiency, and network and storage performance and capacity.

The Niagara Quickstart has specific instructions for Niagara, where the user experience on Niagara is similar to that on Graham
and Cedar, but slightly different.

Preliminary documentation about the GPU expansion to Niagara called "Mist" can be found on  the SciNet documentation site.

Niagara is an allocatable resource in the Resource Allocation Competition (RAC).

Niagara installation update at the SciNet User Group Meeting on February 14th, 2018

 Niagara installation time-lag video

==Niagara hardware specifications==

* 2024 nodes, each with 40 Intel "Skylake" cores at 2.4 GHz or 40 Intel "CascadeLake" cores at 2.5 GHz, for a total of 80,640 cores.
* 202 GB (188 GiB) of RAM per node.
* EDR Infiniband network in a 'Dragonfly+' topology.
* 12.5PB of scratch, 3.5PB of project space (parallel filesystem: IBM Spectrum Scale, formerly known as GPFS).
* 256 TB burst buffer (Excelero + IBM Spectrum Scale).
* No local disks.
* No GPUs.
* Theoretical peak performance ("Rpeak") of 6.25 PF.
* Measured delivered performance ("Rmax") of 3.6 PF.
* 920 kW power consumption.

==Attached storage systems==

Home 200TBParallel high-performance filesystem (IBM Spectrum Scale)
* Backed up to tape
* Persistent
Scratch12.5PB (~100GB/s Write, ~120GB/s Read)Parallel high-performance filesystem (IBM Spectrum Scale)
* Inactive data is purged.
Burst buffer232TB (~90GB/s Write , ~154 GB/s Read)Parallel extra high-performance filesystem (Excelero+IBM Spectrum Scale)
* Inactive data is purged.
Project3.5PB (~100GB/s Write, ~120GB/s Read)Parallel high-performance filesystem (IBM Spectrum Scale
* Backed up to tape
* Allocated through RAC
* Persistent
Archive20PBHigh Performance Storage System (IBM HPSS)
* tape-backed HSM
* Allocated through RAC
* Persistent

==High-performance interconnect==

The Niagara cluster has an EDR Infiniband network in a 'Dragonfly+' topology, with five wings.
Each wing of maximually 432 nodes (i.e., 17280 cores) has
1-to-1 connections.  Network traffic between wings is done through
adaptive routing, which alleviates network congestion and yields an effective blocking of 2:1 between nodes of different wings.

==Node characteristics==

* CPU: 2 sockets with 20 Intel Skylake cores (2.4GHz, AVX512), for a total of 40 cores per node
* Computational performance: 3.07 TFlops theoretical peak.
* Network connection: 100Gb/s EDR Dragonfly+
* Memory: 202 GB (188 GiB) of RAM, i.e., a bit over 4GiB per core.
* Local disk: none. GPUs/Accelerators: none.
* Operating system: Linux CentOS 7

==Scheduling==

The Niagara cluster uses the Slurm scheduler to run jobs.  The basic scheduling commands are therefore similar to those for Cedar and Graham, with a few differences:

* Scheduling is by node only. This means jobs always need to use multiples of 40 cores per job.
* Asking for specific amounts of memory is not be necessary and is discouraged; all nodes have the same amount of memory (202GB/188GiB minus some operating system overhead).

Details, such as how to request burst buffer usage in jobs, are still being worked out.

==Software==

* Module-based software stack.
* Both the standard Alliance software stack as well as cluster-specific software tuned for Niagara are available.
* In contrast with Cedar and Graham, no modules are loaded by default to prevent accidental conflicts in versions. To load the software stack that a user would see on Graham and Cedar, one can load the "CCEnv" module (see Niagara Quickstart).

==Access to Niagara==
Access to Niagara is not enabled automatically for everyone with an Alliance account, but anyone with an active Alliance account can get their access enabled.

If you have an active Alliance account but you do not have access to Niagara yet (e.g. because you are a new user and belong to a group whose primary PI does not have an allocation as granted in the annual Resource Allocation Competition), go to the opt-in page on the CCDB site.  After clicking the "Join" button on that page, it usually takes only one or two business days for access to be granted.

If at any time you require assistance, please do not hesitate to contact us.

===Getting started===

Please read the Niagara Quickstart carefully.