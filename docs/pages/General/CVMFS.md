---
title: "CVMFS/en"
url: "https://docs.alliancecan.ca/wiki/CVMFS/en"
category: "General"
last_modified: "2024-03-11T16:10:27Z"
page_id: 16159
display_title: "CVMFS"
---

`<languages />`{=html}

This page describes the CERN Virtual Machine File System (CVMFS). We use CVMFS to distribute software, data and other content. Refer to [accessing CVMFS](https://docs.alliancecan.ca/accessing_CVMFS "accessing CVMFS"){.wikilink} for instructions on configuring a CVMFS client to access content, and to the official [documentation](https://cvmfs.readthedocs.io/) and [webpage](https://cernvm.cern.ch/fs/) for further information.

## Introduction

CVMFS is a distributed read-only content distribution system, implemented as a POSIX filesystem in user space (FUSE) using HTTP transport. It was originally developed for the LHC (Large Hadron Collider) experiments at CERN to deliver software to virtual machines and to replace diverse shared software installation areas and package management systems at numerous computing sites. It is designed to deliver software in a fast, scalable and reliable fashion, and is now also used to distribute data. The scale of usage across dozens of projects involves \~10^10^ files and directories, \~10^2^ compute sites, and \~10^5^ clients around the world. The [CernVM Monitor](https://cvmfs-monitor-frontend.web.cern.ch/) shows many research groups which use CVMFS and the stratum sites which replicate their repositories.

### Features

- Only one copy of the software needs to be maintained, and can be propagated to and used at multiple sites. Commonly used software can be installed on CVMFS in order to reduce remote software administration.
- Software applications and their prerequisites can be run from CVMFS, eliminating any requirement on the Linux distribution type or release level of a client node.
- The project software stack and OS can be decoupled. For the cloud use case in particular, this allows software to be accessed in a VM without being embedded in the VM image, enabling VM images and software to be updated and distributed separately.
- Content versioning is provided via repository catalog revisions. Updates are committed in transactions and can be rolled back to a previous state.
- Updates are propagated to clients automatically and atomically.
- Clients can view historical versions of repository content.
- Files are fetched using the standard HTTP protocol. Client nodes do not require ports or firewalls to be opened.
- Fault tolerance and reliability are achieved by using multiple redundant proxy and stratum servers. Clients transparently fail over to the next available proxy or server.
- Hierarchical caching makes the CVMFS model highly scalable and robust and minimizes network traffic. There can be several levels in the content delivery and caching hierarchy:
  - The stratum 0 holds the master copy of the repository;
  - Multiple stratum 1 servers replicate the repository contents from the stratum 0;
  - HTTP proxy servers cache requests from clients to stratum 1 servers;
  - The CVMFS client downloads files on demand into the local client cache(s).
    - Two tiers of local cache can be used, e.g. a fast SSD cache and a large HDD cache. A cluster filesystem can also be used as a shared cache for all nodes in a cluster.
- CVMFS clients have read-only access to the filesystem.
- By using Merkle trees and content-addressable storage, and encoding metadata in catalogs, all metadata is treated as data, and practically all data is immutable and highly amenable to caching.
- Metadata storage and operations scale by using nested catalogs, allowing resolution of metadata queries to be performed locally by the client.
- File integrity and authenticity are verified using signed cryptographic hashes, avoiding data corruption or tampering.
- Automatic de-duplication and compression minimize storage usage on the server side. File chunking and on-demand access minimize storage usage on the client side.
- Versatile configurations can be deployed by writing authorization helpers or cache plugins to interact with external authorization or storage providers.

## Reference Material {#reference_material}

- [2018-01-31 Compute Canada Software Installation and Distribution](https://indico.cern.ch/event/608592/contributions/2858287/) 2018 CernVM Workshop
- [2019-06-03 CVMFS at Compute Canada](https://indico.cern.ch/event/757415/contributions/3433887/) 2019 CernVM Workshop
- [2019-06-20 Providing A Unified User Environment for Canada's National Advanced Computing Centers](https://guidebook.com/g/canheitarc2019/#/session/23411098) CANHEIT 2019
- [2019-07-28 Providing a Unified Software Environment for Canada's National Advanced Computing Centers](https://dl.acm.org/doi/10.1145/3332186.3332210) Proceedings of the Practice and Experience in Advanced Research Computing \'19
  - PDF also available [here](https://ssl.linklings.net/conferences/pearc/pearc19_program/views/includes/files/pap139s3-file1.pdf)
- [2020-09-24 Distributing software across campuses and the world with CVMFS](https://bc.net/distributing-software-across-campuses-and-world-cernvm-fs-0) BCNET Connect 2020
- [2021-01-26 CVMFS Tutorial](https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/) Easybuild User Meeting 2021
  - [tutorial slides](https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/eum21-cvmfs-tutorial-slides.pdf)
- [2021-09-27 Unlimited scientific libraries and applications in Kubernetes, instantly!](https://towardsdatascience.com/unlimited-scientific-libraries-and-applications-in-kubernetes-instantly-b69b192ec5e5) Towards Data Science article
  - Illustrates the Compute Canada approach to distributing research applications for users (although the deployment described in the article is only used for a single demo cluster, and uses CephFS instead of CVMFS).
- [2022-02-16 EESSI: A cross-platform ready-to-use optimized scientific software stack](https://onlinelibrary.wiley.com/doi/10.1002/spe.3075) Journal of Software: Practice and Experience, 2022
  - Illustrates an extension to the Compute Canada approach to distributing software, for a broader research community and with wider hardware support.
- [2022-09-13 CVMFS in Canadian Advanced Research Computing](https://indico.cern.ch/event/1079490/contributions/4939532/) 2022 CernVM Workshop
