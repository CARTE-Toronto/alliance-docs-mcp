---
title: "Storage and file management/en"
url: "https://docs.alliancecan.ca/wiki/Storage_and_file_management/en"
category: "General"
last_modified: "2025-10-16T21:52:46Z"
page_id: 3195
display_title: "Storage and file management"
---

`<languages />`{=html}

## Overview

We provide a wide range of storage options to cover the needs of our very diverse users. These storage solutions range from high-speed temporary local storage to different kinds of long-term storage, so you can choose the storage medium that best corresponds to your needs and usage patterns. In most cases the [filesystems](https://en.wikipedia.org/wiki/File_system) on our systems are a `<i>`{=html}shared`</i>`{=html} resource and for this reason should be used responsibly because unwise behaviour can negatively affect dozens or hundreds of other users. These filesystems are also designed to store a limited number of very large files, which are typically binary since very large (hundreds of MB or more) text files lose most of their interest in being readable by humans. You should therefore avoid storing tens of thousands of small files, where small means less than a few megabytes, particularly in the same directory. A better approach is to use commands like [`tar`](https://docs.alliancecan.ca/Archiving_and_compressing_files "tar"){.wikilink} or `zip` to convert a directory containing many small files into a single very large archive file.

It is also your responsibility to manage the age of your stored data: most of the filesystems are not intended to provide an indefinite archiving service so when a given file or directory is no longer needed, you need to move it to a more appropriate filesystem which may well mean your personal workstation or some other storage system under your control. Moving significant amounts of data between your workstation and one of our systems or between two of our systems should generally be done using [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink}.

Note that our storage systems are not for personal use and should only be used to store research data.

When your account is created on a cluster, your home directory will not be entirely empty. It will contain references to your scratch and [project](https://docs.alliancecan.ca/Project_layout "project"){.wikilink} spaces through the mechanism of a [symbolic link](https://en.wikipedia.org/wiki/Symbolic_link), a kind of shortcut that allows easy access to these other filesystems from your home directory. Note that these symbolic links may appear up to a few hours after you first connect to the cluster. While your home and scratch spaces are unique to you as an individual user, the project space is shared by a research group. This group may consist of those individuals with an account sponsored by a particular faculty member or members of a [RAC allocation](https://docs.alliancecan.ca/Resource_Allocation_Competition "RAC allocation"){.wikilink}. A given individual may thus have access to several different project spaces, associated with one or more faculty members, with symbolic links to these different project spaces in the directory projects of your home. Every account has one or many projects. In the folder `projects` within their home directory, each user has a link to each of the projects they have access to. For users with a single active sponsored role, it is the default project of your sponsor while users with more than one active sponsored role will have a default project that corresponds to the default project of the faculty member with the most sponsored accounts.

All users can check the available disk space and the current disk utilization for the `<i>`{=html}project`</i>`{=html}, `<i>`{=html}home`</i>`{=html} and `<i>`{=html}scratch`</i>`{=html} filesystems with the command line utility `<b>`{=html}`<i>`{=html}diskusage_report`</b>`{=html}`</i>`{=html}, available on our clusters. To use this utility, log into the cluster using SSH, at the command prompt type `<i>`{=html}diskusage_report`</i>`{=html}, and press the Enter key. Below is a typical output of this utility:

    # diskusage_report
                       Description                Space           # of files
                     Home (username)         280 kB/47 GB              25/500k
                  Scratch (username)         4096 B/18 TB              1/1000k
           Project (def-username-ab)       4096 B/9536 GB              2/500k
              Project (def-username)       4096 B/9536 GB              2/500k

More detailed output is available using the [Diskusage Explorer](https://docs.alliancecan.ca/Diskusage_Explorer "Diskusage Explorer"){.wikilink} tool.

## Storage types {#storage_types}

Unlike your personal computer, our systems will typically have several storage spaces or filesystems and you should ensure that you are using the right space for the right task. In this section we will discuss the principal filesystems available on most of our systems and the intended use of each one along with some of its characteristics.

- `<b>`{=html}HOME:`</b>`{=html} While your home directory may seem like the logical place to store all your files and do all your work, in general this isn\'t the case; your home normally has a relatively small quota and doesn\'t have especially good performance for writing and reading large amounts of data. The most logical use of your home directory is typically source code, small parameter files and job submission scripts.
- `<b>`{=html}PROJECT:`</b>`{=html} The project space has a significantly larger quota and is well adapted to [ sharing data](https://docs.alliancecan.ca/Sharing_data " sharing data"){.wikilink} among members of a research group since it, unlike the home or scratch, is linked to a professor\'s account rather than an individual user. The data stored in the project space should be fairly static, that is to say the data are not likely to be changed many times in a month. Otherwise, frequently changing data, including just moving and renaming directories, in project can become a heavy burden on the tape-based backup system.
- `<b>`{=html}SCRATCH`</b>`{=html}: For intensive read/write operations on large files (\> 100 MB per file), scratch is the best choice. However, remember that important files must be copied off scratch since they are not backed up there, and older files are subject to [purging](https://docs.alliancecan.ca/Scratch_purging_policy "purging"){.wikilink}. The scratch storage should therefore be used for temporary files: checkpoint files, output from jobs and other data that can easily be recreated. `<b>`{=html}Do not regard SCRATCH as your normal storage! It is for transient files that you can afford to lose.`</b>`{=html}
- `<b>`{=html}NEARLINE`</b>`{=html}: Nearline is a tape-based filesystem intended for inactive data. Datasets which you do not expect to access for months are good candidates to be stored in `/nearline`. For more information, see [Using nearline storage](https://docs.alliancecan.ca/Using_nearline_storage "Using nearline storage"){.wikilink}.
- `<b>`{=html}SLURM_TMPDIR`</b>`{=html}: While a job is running, the environment variable `$SLURM_TMPDIR` holds a unique path to a temporary folder on a fast, local filesystem on each compute node allocated to the job. When the job ends, the directory and its contents are deleted, so `$SLURM_TMPDIR` should be used for temporary files that are only needed for the duration of the job. Its advantage, compared to the other networked filesystem types above, is increased performance due to the filesystem being local to the compute node. It is especially well-suited for large collections of small files (for example, smaller than a few megabytes per file). Note that this filesystem is shared between all jobs running on the node, and that the available space depends on the compute node type. A more detailed discussion of using `$SLURM_TMPDIR` is available at [ this page](https://docs.alliancecan.ca/Using_$SLURM_TMPDIR " this page"){.wikilink}.

## Project space consumption per user {#project_space_consumption_per_user}

While the command `<b>`{=html}diskusage_report`</b>`{=html} gives the space and file count usage per user on `<i>`{=html}home`</i>`{=html} and `<i>`{=html}scratch`</i>`{=html}, it shows the total quota of the group on project. It includes all the files from each member of the group. Since the files that belong to a user could however be anywhere in the project space, it is difficult to obtain correct figures per user and per given project in case a user has access to more than one project. However, users can obtain an estimate of their space and file count use on the entire project space by running the command

`lfs quota -u $USER /project`

In addition to that, users can obtain an estimate for the number of files in a given directory (and its subdirectories) using the command `lfs find`, e.g.

``` console
lfs find <path to the directory> -type f | wc -l
```

## Best practices {#best_practices}

- Regularly clean up your data in the scratch and project spaces, because those filesystems are used for huge data collections.
  - Document your files with [README files](https://docs.alliancecan.ca/README_files "README files"){.wikilink}.
  - For any set of files that could be deleted:
    1.  Create a temporary directory `toDelete`
    2.  Move the files to be deleted to this directory
    3.  `<b>`{=html}Verify`</b>`{=html} the contents of `toDelete`
    4.  Delete `toDelete` recursively.
  - If possible, avoid using `*` and `/` characters in your `rm` commands.
    - Navigate to the parent directory that contains the item(s) to be deleted. Double-check you are in the correct directory.
    - If the directory has a [Makefile](https://docs.alliancecan.ca/Make "Makefile"){.wikilink}, it may use `*` and `/` in `rm` commands, but these commands have to be well tested.
  - In shell scripts, if environment variables are used in `rm` commands, each variable has to be tested before use: empty or undefined variables can cause catastrophic errors, and any input value has to be checked against malicious or erroneous use of the script.
- Only use text format for files that are smaller than a few megabytes.
- As far as possible, use scratch and local storage for temporary files. For local storage you can use the temporary directory created by the [job scheduler](https://docs.alliancecan.ca/Running_jobs "job scheduler"){.wikilink} for this, named `$SLURM_TMPDIR`.
- If your program must search within a file, it is fastest to do it by first reading it completely before searching.
- If you no longer use certain files but they must be retained, [archive and compress](https://docs.alliancecan.ca/Archiving_and_compressing_files "archive and compress"){.wikilink} them, and if possible move them to an alternative location like [nearline](https://docs.alliancecan.ca/Using_nearline_storage "nearline"){.wikilink}.
- For more on managing many files, see [Handling large collections of files](https://docs.alliancecan.ca/Handling_large_collections_of_files "Handling large collections of files"){.wikilink}, especially if you are limited by a quota on the number of files.
- Having any sort of parallel write access to a file stored on a shared filesystem like home, scratch and project is likely to create problems unless you are using a specialized tool such as [MPI-IO](https://en.wikipedia.org/wiki/Message_Passing_Interface#I/O).
- If your needs are not well served by the available storage options please contact [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

## Filesystem quotas and policies {#filesystem_quotas_and_policies}

In order to ensure that there is adequate space for all users, there are a variety of quotas and policy restrictions concerning backups and automatic purging of certain filesystems. By default on our clusters, each user has access to the home and scratch spaces, and each group has access to 1 TB of project space. Small increases in project and scratch spaces are available through our [Rapid Access Service](https://docs.alliancecan.ca/Rapid_Access_Service "Rapid Access Service"){.wikilink}. Larger increases in project spaces are available through the annual [Resource Allocation Competition](https://docs.alliancecan.ca/Resource_Allocation_Competition "Resource Allocation Competition"){.wikilink}. You can see your current quota usage for various filesystems on our clusters using the command [`diskusage_report`](https://docs.alliancecan.ca/Storage_and_file_management#Overview "diskusage_report"){.wikilink}.

`<tabs>`{=html} `<tab name="Fir">`{=html}

  Filesystem       Default Quota                       Lustre-based   Backed up   Purged                                     Available by Default   Mounted on Compute Nodes
  ---------------- ----------------------------------- -------------- ----------- ------------------------------------------ ---------------------- --------------------------
  Home Space       50 GB and 500K files per user[^1]   Yes            Yes         No                                         Yes                    Yes
  Scratch Space    20 TB and 1M files per user         Yes            No          Files older than 60 days are purged.[^2]   Yes                    Yes
  Project Space    1 TB and 500K files per group[^3]   Yes            Yes         No                                         Yes                    Yes
  Nearline Space   2 TB and 5000 files per group       Yes            Yes         No                                         Yes                    No

  : Filesystem Characteristics

<references />

Since April 1, 2024, new Rapid Access Service (RAS) policies allow larger quotas for the project and nearline spaces. For more details, see the \"Storage\" section at [Rapid Access Service](https://docs.alliancecan.ca/Rapid_Access_Service "Rapid Access Service"){.wikilink}. Quota changes larger than those permitted by RAS will require an application to the annual [Resource Allocation Competition](https://docs.alliancecan.ca/Resource_Allocation_Competition "Resource Allocation Competition"){.wikilink}. `</tab>`{=html} `<tab name="Nibi">`{=html}

  Filesystem       Default Quota                                     Lustre-based   Backed up   Purged   Available by Default   Mounted on Compute Nodes
  ---------------- ------------------------------------------------- -------------- ----------- -------- ---------------------- --------------------------
  Home Space       50 GB and 500K files per user[^4]                 No             Yes         No       Yes                    Yes
  Scratch Space    20 TB hard / 1TB soft and 1M files per user[^5]   No             No          No       Yes                    Yes
  Project Space    1 TB and 500K files per group[^6]                 No             Yes         No       Yes                    Yes
  Nearline Space   10 TB and 5000 files per group                    Yes            Yes         No       Yes                    No

  : Filesystem Characteristics

<references />

Since April 1, 2024, new Rapid Access Service (RAS) policies allow larger quotas for project and nearline spaces. For more details, see the \"Storage\" section at [Rapid Access Service](https://docs.alliancecan.ca/Rapid_Access_Service "Rapid Access Service"){.wikilink}. Quota changes larger than those permitted by RAS will require an application to the annual [Resource Allocation Competition](https://docs.alliancecan.ca/Resource_Allocation_Competition "Resource Allocation Competition"){.wikilink}. `</tab>`{=html} `<tab name="Narval and Rorqual">`{=html}

  Filesystem       Default Quota                       Lustre-based   Backed up   Purged                                     Available by Default   Mounted on Compute Nodes
  ---------------- ----------------------------------- -------------- ----------- ------------------------------------------ ---------------------- --------------------------
  Home Space       50 GB and 500K files per user[^7]   Yes            Yes         No                                         Yes                    Yes
  Scratch Space    20 TB and 1M files per user         Yes            No          Files older than 60 days are purged.[^8]   Yes                    Yes
  Project Space    1 TB and 500K files per group[^9]   Yes            Yes         No                                         Yes                    Yes
  Nearline Space   1 TB and 5000 files per group       Yes            Yes         No                                         Yes                    No

  : Filesystem Characteristics

<references />

Since April 1, 2024, new Rapid Access Service (RAS) policies allow larger quotas for project and nearline spaces. For more details, see the \"Storage\" section at [Rapid Access Service](https://docs.alliancecan.ca/Rapid_Access_Service "Rapid Access Service"){.wikilink}. Quota changes larger than those permitted by RAS will require an application to the annual [Resource Allocation Competition](https://docs.alliancecan.ca/Resource_Allocation_Competition "Resource Allocation Competition"){.wikilink}. `</tab>`{=html} `<tab name="Niagara">`{=html}

+-----------+----------------------------------------------+------------+-----------------+-----------+----------------+------------------+
| location  | quota                                        | block size | expiration time | backed up | on login nodes | on compute nodes |
+===========+==========================+===================+===========:+=================+===========+================+==================+
| \$HOME    | 100 GB per user                              | 1 MB       |                 | yes       | yes            | read-only        |
+-----------+----------------------------------------------+------------+-----------------+-----------+----------------+------------------+
| \$SCRATCH | 25 TB per user (dynamic per group)           | 16 MB      | 2 months        | no        | yes            | yes              |
|           +--------------------------+-------------------+            |                 |           |                |                  |
|           | up to 4 users per group  | 50TB              |            |                 |           |                |                  |
|           +--------------------------+-------------------+            |                 |           |                |                  |
|           | up to 11 users per group | 125TB             |            |                 |           |                |                  |
|           +--------------------------+-------------------+            |                 |           |                |                  |
|           | up to 28 users per group | 250TB             |            |                 |           |                |                  |
|           +--------------------------+-------------------+            |                 |           |                |                  |
|           | up to 60 users per group | 400TB             |            |                 |           |                |                  |
|           +--------------------------+-------------------+            |                 |           |                |                  |
|           | above 60 users per group | 500TB             |            |                 |           |                |                  |
+-----------+--------------------------+-------------------+------------+-----------------+-----------+----------------+------------------+
| \$PROJECT | by group allocation (RRG or RPP)             | 16 MB      |                 | yes       | yes            | yes              |
+-----------+----------------------------------------------+------------+-----------------+-----------+----------------+------------------+
| \$ARCHIVE | by group allocation                          |            |                 | dual-copy | no             | no               |
+-----------+----------------------------------------------+------------+-----------------+-----------+----------------+------------------+
| \$BBUFFER | 10 TB per user                               | 1 MB       | very short      | no        | yes            | yes              |
+-----------+----------------------------------------------+------------+-----------------+-----------+----------------+------------------+

- [Inode vs. Space quota (PROJECT and SCRATCH)](https://docs.scinet.utoronto.ca/images/9/9a/Inode_vs._Space_quota_-_v2x.pdf)
- [dynamic quota per group (SCRATCH)](https://docs.scinet.utoronto.ca/images/0/0e/Scratch-quota.pdf)
- Compute nodes do not have local storage.
- Archive (a.k.a. nearline) space is on [HPSS](https://docs.scinet.utoronto.ca/index.php/HPSS)
- Backup means a recent snapshot, not an archive of all data that ever was.
- `$BBUFFER` stands for [Burst Buffer](https://docs.scinet.utoronto.ca/index.php/Burst_Buffer), a faster parallel storage tier for temporary data.

`</tab>`{=html} `<tab name="Killarney">`{=html}

+-----------+-----------------------------------------------------------------+-----------------+-----------+----------------+------------------+
| Location  | Quota                                                           | Expiration Time | Backed Up | On Login Nodes | On Compute Nodes |
+===========+===============================================+=================+=================+===========+================+==================+
| \$HOME    | 50 GB per user                                                  | none            | yes       | yes            | yes              |
+-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+
| \$SCRATCH | CIFAR AI Chairs                               | 2 TB per user   | 2 months        | no        | yes            | yes              |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | AI Institute Faculty Affiliates               | 1 TB per user   |                 |           |                |                  |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | Faculty members, within an AI program         | 500 GB per user |                 |           |                |                  |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | Faculty members, applying AI to other domains | 250 GB per user |                 |           |                |                  |
+-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+
| \$PROJECT | CIFAR AI Chairs                               | 5 TB            | none            | yes       | yes            | yes              |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | AI Institute Faculty Affiliates               | 2 TB            |                 |           |                |                  |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | Faculty members, within an AI program         | 1 TB            |                 |           |                |                  |
|           +-----------------------------------------------+-----------------+                 |           |                |                  |
|           | Faculty members, applying AI to other domains | 500 GB          |                 |           |                |                  |
+-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+

: Filesystem Characteristics

- All filesystems are served from VastData storage
- Backup means a recent snapshot, not an archive of all data that ever was.
- See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information on `<b>`{=html}expiration time`</b>`{=html}.
  `</tab>`{=html}

  <references />

  `<tab name="TamIA">`{=html}

  +-----------+-----------------------------------------------------------------+-----------------+-----------+----------------+------------------+
  | Location  | Quota                                                           | Expiration Time | Backed Up | On Login Nodes | On Compute Nodes |
  +===========+===============================================+=================+=================+===========+================+==================+
  | \$HOME    | 25 GB per user                                                  | none            | no        | yes            | yes              |
  +-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+
  | \$SCRATCH | CIFAR AI Chairs                               | 2 TB per user   | 2 months        | no        | yes            | yes              |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | AI Institute Faculty Affiliates               | 1 TB per user   |                 |           |                |                  |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | Faculty members, within an AI program         | 500 GB per user |                 |           |                |                  |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | Faculty members, applying AI to other domains | 500 GB per user |                 |           |                |                  |
  +-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+
  | \$PROJECT | CIFAR AI Chairs                               | 5 TB            | none            | no        | yes            | yes              |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | AI Institute Faculty Affiliates               | 2 TB            |                 |           |                |                  |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | Faculty members, within an AI program         | 500 GB          |                 |           |                |                  |
  |           +-----------------------------------------------+-----------------+                 |           |                |                  |
  |           | Faculty members, applying AI to other domains | 500 GB          |                 |           |                |                  |
  +-----------+-----------------------------------------------+-----------------+-----------------+-----------+----------------+------------------+

  : Filesystem Characteristics

  - See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information on `<b>`{=html}expiration time`</b>`{=html}.
    `</tab>`{=html}

    <references />

    `<tab name="Vulcan">`{=html}

    +-----------+---------------------------------------------------------------+-----------------+-----------+----------------+------------------+
    | Location  | Quota                                                         | Expiration Time | Backed Up | On Login Nodes | On Compute Nodes |
    +===========+===============================================+===============+=================+===========+================+==================+
    | \$HOME    | 50 GB per user                                                | none            | yes       | yes            | yes              |
    +-----------+-----------------------------------------------+---------------+-----------------+-----------+----------------+------------------+
    | \$SCRATCH | CIFAR AI Chairs                               | 5 TB per user | 2 months        | no        | yes            | yes              |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | AI Institute Faculty Affiliates               | 5 TB per user |                 |           |                |                  |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | Faculty members, within an AI program         | 5 TB per user |                 |           |                |                  |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | Faculty members, applying AI to other domains | 5 TB per user |                 |           |                |                  |
    +-----------+-----------------------------------------------+---------------+-----------------+-----------+----------------+------------------+
    | \$PROJECT | CIFAR AI Chairs                               | 12.5 TB       | none            | yes       | yes            | yes              |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | AI Institute Faculty Affiliates               | 10 TB         |                 |           |                |                  |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | Faculty members, within an AI program         | 7.5 TB        |                 |           |                |                  |
    |           +-----------------------------------------------+---------------+                 |           |                |                  |
    |           | Faculty members, applying AI to other domains | 5 TB          |                 |           |                |                  |
    +-----------+-----------------------------------------------+---------------+-----------------+-----------+----------------+------------------+

    : Filesystem Characteristics

<li>

See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information on `<b>`{=html}expiration time`</b>`{=html}.

`</tab>`{=html}

<references />

`</tabs>`{=html}

The backup policy on the home and project space is nightly backups which are retained for 30 days, while deleted files are retained for a further 60 days; note that is entirely distinct from the age limit for purging files from the scratch space. If you wish to recover a previous version of a file or directory, you should contact [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} with the full path for the file(s) and desired version (by date).

## See also {#see_also}

- [Diskusage Explorer](https://docs.alliancecan.ca/Diskusage_Explorer "Diskusage Explorer"){.wikilink}
- [Project layout](https://docs.alliancecan.ca/Project_layout "Project layout"){.wikilink}
- [Sharing data](https://docs.alliancecan.ca/Sharing_data "Sharing data"){.wikilink}
- [Transferring data](https://docs.alliancecan.ca/Transferring_data "Transferring data"){.wikilink}
- [Tuning Lustre](https://docs.alliancecan.ca/Tuning_Lustre "Tuning Lustre"){.wikilink}
- [Archiving and compressing files](https://docs.alliancecan.ca/Archiving_and_compressing_files "Archiving and compressing files"){.wikilink}
- [Handling large collections of files](https://docs.alliancecan.ca/Handling_large_collections_of_files "Handling large collections of files"){.wikilink}
- [Parallel I/O introductory tutorial](https://docs.alliancecan.ca/Parallel_I/O_introductory_tutorial "Parallel I/O introductory tutorial"){.wikilink}

[^1]: This quota is fixed and cannot be changed.

[^2]: See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information.

[^3]: Project space can be increased to 40 TB per group by a RAS request, subject to the limitations that the minimum project space per quota cannot be less than 1 TB and the sum over all four general-purpose clusters cannot exceed 43 TB. The group\'s sponsoring PI should write to [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} to make the request.

[^4]: This quota is fixed and cannot be changed.

[^5]: An 1 TB soft quota on scratch applies to each researcher. This soft quota can be exceeded for up to 60 days after which no additional files may be written to scratch. Files may be written again once the researcher has removed or deleted enough files to bring their total scratch use under 1 TB. See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information.

[^6]: Project space can be increased to 40 TB per group by a RAS request, subject to the limitations that the minimum project space per quota cannot be less than 1 TB and the sum over all four general-purpose clusters cannot exceed 43 TB. The group\'s sponsoring PI should write to [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} to make the request.

[^7]: This quota is fixed and cannot be changed.

[^8]: See [Scratch purging policy](https://docs.alliancecan.ca/Scratch_purging_policy "Scratch purging policy"){.wikilink} for more information.

[^9]: Project space can be increased to 40 TB per group by a RAS request, subject to the limitations that the minimum project space per quota cannot be less than 1 TB and the sum over all four general-purpose clusters cannot exceed 43 TB. The group\'s sponsoring PI should write to [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} to make the request.
