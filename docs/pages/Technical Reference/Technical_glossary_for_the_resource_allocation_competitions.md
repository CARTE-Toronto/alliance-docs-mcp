---
title: "Technical glossary for the resource allocation competitions/en"
url: "https://docs.alliancecan.ca/wiki/Technical_glossary_for_the_resource_allocation_competitions/en"
category: "Technical Reference"
last_modified: "2025-11-24T22:06:46Z"
page_id: 14456
display_title: "Technical glossary for the resource allocation competitions"
---

== Computational resources ==
CPU: Is the abbreviation for central processing unit. Sometimes referred to simply as the central processor, but more commonly called processor, the CPU is the brains of the computer where most calculations take place.

GPU: GPU computing is the use of a graphics processing unit (GPU) to accelerate deep learning,  analytics,  and engineering applications, for example. GPU accelerators now power energy-efficient data centres in government labs, universities, enterprises, and small-and-medium businesses around the world. They play a huge role in accelerating applications in platforms ranging from artificial intelligence to cars, drones, and robots.

VCPU: Stands for virtual central processing unit. One or more VCPUs are assigned to every Virtual Machine (VM) within a cloud environment. Each VCPU is seen as a single physical CPU core by the VM’s operating system.

VGPU: Stands for virtual graphics processing unit (VGPU). One or more VGPUs can be assigned to Virtual Machines (VM) within a cloud environment. Each VGPU is seen as a single physical GPU device by the VM's operating system.

Reference GPU Unit (RGU): RGU is a unit measuring the amount of GPU resources that are used. It represents the "cost" of utilizing a particular GPU model, whose RGU value varies based on performance. For example: 1 GPU A100-40GB = 4.0 RGU; 1 GPU V100-16GB = 2.2 RGU; 1 GPU P100-12GB = 1.0 RGU.

== Resource allocations ==
Allocations of storage space and cloud resources are handled differently from allocations of CPUs and GPUs.

* Storage allocations are straightforward: A research group will get a maximum amount of storage that they can use exclusively throughout the allocation period. Likewise, a cloud resource allocation is a maximum number of VCPUs, amount of storage, etc., which cannot be exceeded during the allocation period.
* CPU and GPU allocations are not maxima but targets for average usage, which are then translated into scheduling priorities. An allocation of N CPUs means the group may expect to have access to N CPUs throughout the allocation period. Periods of greater usage may be possible as competing demand allows, but are not guaranteed; periods of lesser usage will not be compensated, whether due to the group’s actions or resource unavailability. To learn more about compute allocations, click here.

== Batch computing ==
Cluster: A group of interconnected compute nodes managed as a unit by a scheduling program.

Compute node: A computational unit of a cluster, one or more of which can be allocated to a job. A node has its own operating system image, one or more CPU cores and some memory (RAM). Nodes can be used by the jobs in either exclusive or shared manner depending on the cluster.

Core-year: The equivalent of using 1 CPU core continuously for a full year. Using 12 cores for a month, or 365 cores for a single day are both equivalent to 1 core-year. Compute allocations are based on core-years allocations.

Core-equivalent: A core-equivalent is a bundle made up of a single core and some amount of associated memory. In other words, a core-equivalent is a core plus the amount of memory considered to be associated with each core on a given system. See detailed explanation here.

GPU-year: a GPU-year is the equivalent of using 1 GPU continuously for a full year or 12 GPU for a month.

RGU-year: A value in RGU-years is calculated from multiplying GPU-years times the RGU of a given GPU model. For example, 10 GPU-years of an A100-40GB (which costs 4 RGU) equals 40 RGU-years.

Head or Login node: Typically when you access a cluster you are accessing a head node, or gateway/login node. A head node is configured to be the launching point for jobs running on the cluster. When you are told or asked to login or access a cluster, invariably you are being directed to log into the head node, often nothing more than a node configured to act as a middle point between the actual cluster and the outside network.

Fair share allocation: Generally speaking, batch processing priority is allocated based on a fair-share algorithm. Each user is allocated a share of the total system resources, which effectively translates into priority access to the system. If you have used a large fraction of the system recently (ie. larger than your fair-share), your priority drops. However, the scheduling system has a limited time window over which it calculates priority. After some time (e.g., weeks) of reduced usage, it gradually “forgets” that you overused in the past. This is designed to ensure full system usage and not to penalize users who take advantage of idle compute resources. A consequence is that your total allocation is not a limit on how many compute resources you can consume. Rather, your total allocation represents what you should be able to get over the course of the year if you submit a constant workload to the system and it is fully busy. In other words, once your “total allocation” is used, just keep working.

Job: A job is the basic execution object managed by the batch system. It is a collection of one or more related computing processes that is managed as a whole. Users define resource requirements for the job when they submit it to the batch system. A job description includes a resource request, such as the amount of required memory, the duration of the job, and how many compute cores this job will require. Jobs can be either serial (running on one compute core) or parallel (running on multiple compute cores).

Parallel job: A job that runs simultaneously on multiple CPU cores. Parallel jobs can be roughly classified as threaded/SMP jobs running on a single compute node and sharing the same memory space, and distributed memory jobs that can run across several compute nodes.

Serial job: A job that requires one compute CPU core to run.

Uneven usage: Most schedulers are tuned to deliver a certain number of core-years over a fixed period of time, assuming relatively consistent usage of the system. Users may have very inconsistent workloads, with significant peaks and valleys in their usage. They therefore may need a “burst” of compute resources in order to use their RAC allocation effectively. Normally we expect allocations to be used in a relatively even way throughout the award period. If you anticipate having bursty workloads or variable usage, please indicate that in your RAC application. If you are having problems running jobs, contact Technical support.

== Memory ==
Memory per core: The amount of memory (RAM) per CPU core. If a compute node has 2 CPUs, each having 6 cores and 24GB (gigabytes) of installed RAM, then this compute node will have 2GB of memory per core.

Memory per node: The total amount of installed RAM in a compute node.

System memory per core: Amount of system memory requested by a job in GB divided by the number of CPU cores requested by the job.

System memory per GPU: Amount of system memory requested by a job in GB divided by the number of GPUs requested by the job.

== Storage ==
Disk: A disk, hard drive or solid-state drive is permanent storage (compared to a computer’s main memory or RAM) that holds programs, input files, output results, etc.

Filesystems: A directory structure made available for use by systems in a cluster. Each filesystem may have different performance characteristics, space available, and intended use. Some filesystems may be available to only head nodes in a cluster, while others may be shared with compute nodes for working storage during job execution. Filesystems typically available on clustered systems include:

Scratch:  This filesystem, available on compute nodes, is composed of high-performance storage used during computational jobs. Intended primarily for temporary or transient files, bulk results of your computations and simulations, or any material that can be easily recreated or reacquired. Data may be copied to scratch, then removed from scratch once job execution is complete. Scratch storage is subject to periodic “cleaning” (or purging) according to local system policies, and is not formally allocated, however limited by generous quotas.

Home:  The home filesystem is commonly used for storage of user’s personal files, executable programs, job execution scripts, and relatively small input datasets. Each user has a folder in the home filesystem called a “home directory”. The home directory is persistent, smaller than scratch and, in most systems, backed up regularly. The home directory is visible to all nodes in a given cluster.
Project: The project filesystem is of medium to high performance disk and also available on compute nodes. This filesystem is larger in available storage than home, and in most systems is backed up regularly. This filesystem is generally used to store frequently-used project data, however with minimum data churning, and is allocated through the RAC process.

Nearline: The nearline filesystem is a disk-tape hybrid storage system, in which data with size above a certain threshold is  automatically migrated from disk to tape, and then back again upon read operations. Access to this storage resource requires deliberate actions by users (i.e., via the Linux command line: cp, mv, rsync, etc …) of placing files into this designated nearline location, or by file transfers from another filesystem (scratch, project, home, etc).  The tape subsystem has very high capacity, but adds latency when files need to be accessed again. This storage system should be used for datasets that are infrequently accessed, and needs to be retained for long periods of time. This is not true “archival” storage in that the datasets must be part of an “active” project. Nearline capacity is managed by quotas, and  allocations are via the RAC process.

dCache: dCache is a storage filesystem developed originally for high-energy physics projects for very large datasets (petabytes). dCache storage is essentially an object file storage layer on top of classical storage providing a single namespace and various authorized access and transfer protocols to the underlying immutable data. Allocations here tend to be for large projects with many Principal Investigators and researchers. dCache capacity is allocated via the RAC process. If you wish to use this storage, contact the Subatomic Physics National Team by writing to our Technical support.

Local storage: This refers to the hard drive or solid-state drive in a compute node that can be used to temporarily store programs, input files, or their results. Files in local storage on one node can not be accessed on any other node. The local storage may not be persistent, so the files created on the local storage should be moved to non-local storage to avoid data loss.

Site: A member of one of the Alliance's partners providing advanced research computing (ARC) resources (such as high-performance computing clusters, Clouds, storage, and/or technical support).

Tape: Tape is a storage technology used to store long-term data that are infrequently accessed. It is considerably lower in cost than disk and is a viable option for many use cases.

Terabytes (TB): Terabytes are most often used to measure the storage capacity of large storage devices. One terabyte (abbreviated “TB”) is equal to 1,000 gigabytes and precedes the petabyte unit of measurement.

== Cloud ==
Alliance Cloud: is a pool of hardware supporting virtualization. This can be thought of as Infrastructure as a Service (IaaS).

VCPU year: same as CPU year, but for cloud.

VGPU year: same as GPU year, but for cloud.

Compute instances: These are instances that have a limited life-time and typically have constant high-CPU requirements for the instances life-time. They have also been referred to as ‘batch’ instances. These will be granted higher vCPU/Memory quotas since they are time-limited instances.

Persistent instances: These are instances that are meant to run indefinitely (e.g., based on the clouds availability) and would include web servers, database servers, etc. In general, these are thought to be lower CPU or bursty CPU instances. These will have lower vCPU/Memory quotas since they are meant to consume the resources for long periods of time.
Cloud storage: Persistent cloud storage provides virtual disk functionality to virtual machines running in the cloud. Persistent cloud storage is very reliable and scalable, made possible by specialized software (Ceph).

Floating IP: A public IP address that a project can associate with a VM so that the instance has the same public IP address each time that it boots. You create a pool of floating IP addresses and assign them to instances as they are launched to maintain a consistent IP address for maintaining DNS assignment.

Instance: A running Virtual Machine (VM), or a VM in a known state such as suspended, that can be used like a hardware server.

Memory per core: See definition in the Memory section above.

Ephemeral local disk: Ephemeral disks are often used in cloud native application use cases where VMs are expected to be short-lived and the data does not need to persist beyond the life of the VM. This may include a VM with data that is cached for short usage; a VM hosting applications that replicate their data across multiple VMs; or a VM whose persistent data is saved on a volume or on any external storage support. Ephemeral local disks use the storage directly attached to a virtualisation host and are not expected to survive if the hardware fails, unlike volumes which are backed by a resilient storage cluster. An ephemeral local disk is purged and deleted when an instance is itself deleted by the cloud user or by the admin.

Service Portal: We host many research web portals that serve datasets or tools to a broad research community. These portals generally do not require large computing or storage resources, but may require support effort from our technical team. Groups applying for a service portal often use our cloud, generally require a public IP address, and may (or may not) have more stringent up-time requirements than most research projects. This option is shown as “Portal” in the online form.

Virtual Machine (VM): See Instance above.

Volume and Snapshot Storage: Total size of persistent storage space for all volumes and snapshots. Measured in GB.

Volume Snapshot: A point-in-time copy of an OpenStack storage volume. Used for backups or as a base to instantiate (launch) other VMs.

Object Storage: persistent object storage space used to store large amounts of data that is mostly read access (such as images and data sets). This is considered 'bulk' storage. Accessible from anywhere in the world. Offered as S3 and Swift protocols. Measured in TB.

Shared Filesystem Storage: persistent storage space offered as a Unix-compliant file system that can be mounted across multiple hosts in a tenant. This is useful for sharing data across multiple hosts. Service runs on CephFS and requires either a Fuse driver (Windows/Linux) or the CephFS kernel driver (Linux) for access. Measured in TB.