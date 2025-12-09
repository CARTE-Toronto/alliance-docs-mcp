---
title: "Mp2/en"
url: "https://docs.alliancecan.ca/wiki/Mp2/en"
category: "General"
last_modified: "2024-11-13T17:24:12Z"
page_id: 13562
display_title: "Mp2"
---

`<languages />`{=html}

  ----------------------------------------------------------------------
  Availability: February 2012 - April 1, 2020
  Login Node: **mp2.calculcanada.ca**
  Globus Endpoint: **computecanada#mammouth**
  Data Transfer Node (rsync, scp, sftp,\...) : **mp2.calculcanada.ca**
  ----------------------------------------------------------------------

**Mp2 is now exclusive to researchers from the Université de Sherbrooke.**

Mammouth-Mp2 is a heterogeneous and polyvalent cluster designed for ordinary computations; it is located at the [Université de Sherbrooke](http://www.usherbrooke.ca/).

# Site-specific policies {#site_specific_policies}

- Each job must have a duration of at least one hour (at least five minutes for test jobs) and a user cannot have more than 1000 jobs (running and queued) at any given time. The maximum duration of a job is 168 hours (seven days).

<!-- -->

- No GPUs.

# Storage

+----------------------------+------------------------------------------------------------------------------------------------------------------------+
| HOME\                      | - This space is small and cannot be expanded; you should use your `project` space for substantial storage needs.       |
| Lustre filesystem\         |                                                                                                                        |
| 79.6 TB of space in total  | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - 50 GB of space and 500K files per user.                                                                              |
|                            |                                                                                                                        |
|                            | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - There is a daily backup.                                                                                             |
+----------------------------+------------------------------------------------------------------------------------------------------------------------+
| SCRATCH\                   | - Large space for storing temporary files during computations.                                                         |
| Lustre filesystem\         |                                                                                                                        |
| 358.3 TB of space in total | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - 20 TB of space and 1M files per user.                                                                                |
|                            |                                                                                                                        |
|                            | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - No backup system in place.                                                                                           |
+----------------------------+------------------------------------------------------------------------------------------------------------------------+
| PROJECT\                   | - This space is designed for sharing data among the members of a research group and for storing large amounts of data. |
| Lustre filesystem\         |                                                                                                                        |
| 716.6 TB of space in total | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - 1 TB of space and 500K files per group.                                                                              |
|                            |                                                                                                                        |
|                            | <!-- -->                                                                                                               |
|                            |                                                                                                                        |
|                            | - No backup system in place.                                                                                           |
+----------------------------+------------------------------------------------------------------------------------------------------------------------+

For transferring data by Globus, you should use the endpoint `computecanada#mammouth`, whereas tools like rsync and scp can simply use an ordinary login node.

# High-performance interconnect {#high_performance_interconnect}

The Mellanox QDR (40 Gb/s) Infiniband network links together all of the cluster\'s nodes and is non-blocking for groups of 216 nodes and 5:1 for the rest of the cluster.

# Node characteristics {#node_characteristics}

  Quantity   Cores   Available Memory         CPU Type                                                                Storage          GPU Type
  ---------- ------- ------------------------ ----------------------------------------------------------------------- ---------------- ----------
  1588       24      \| 31 GB or 31744 MB     12 cores/socket, 2 sockets/node. AMD Opteron Processor 6172 @ 2.1 GHz   1TB SATA disk.   \-
  20         48      \| 251 GB or 257024 MB   12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz   1TB SATA disk.   \-
  2          48      \| 503 GB or 515072 MB   12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz   1TB SATA disk.   \-
