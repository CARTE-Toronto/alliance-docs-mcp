---
title: "Handling large collections of files/en"
url: "https://docs.alliancecan.ca/wiki/Handling_large_collections_of_files/en"
category: "General"
last_modified: "2024-08-29T22:26:37Z"
page_id: 11765
display_title: "Handling large collections of files"
---

In certain domains, notably AI and Machine Learning, it is common to have to manage very large collections of files, meaning hundreds of thousands or more. The individual files may be fairly small, e.g. less than a few hundred kilobytes. In these cases, a problem arises due to filesystem quotas on our clusters that limit the number of filesystem objects. Very large numbers of files, particularly small ones, create significant problems for the performance of these shared filesystems as well as the automated backup of the home and project spaces.

So how can a user or group of users store these necessary datasets on the cluster?  In this page we will present a variety of different solutions, each with its own pros and cons, so you may judge for yourself which is appropriate for you.

=Finding folders with lots of files=

As always in optimization, it is better to start looking for where some cleanup is worth doing. You may consider the following code which will recursively count all files in folders in the current directory:

for FOLDER in $(find . -maxdepth 1 -type d | tail -n +2); do
  echo -ne "$FOLDER:\t"
  find $FOLDER -type f | wc -l
done

=Finding folders using the most disk space=
The following code will output the 10 directories using the most disk space from your current directory.
 sort -hr  head -10}}

=Solutions=

==Local disk==
Local disks attached to compute nodes are at least SATA SSD or better, and, in general, will have a performance that is considerably better than the project or scratch filesystems. Note that a local disk is shared by all running jobs on that node without being allocated by the scheduler. The actual amount of local disk space varies from one cluster to another (and might also vary within a given cluster). For example,

*  Béluga offers roughly 370GB of local disk for the CPU nodes, the GPU nodes have a 1.6TB NVMe disk (to help with the AI image datasets with their millions of small files).
* Niagara does not have local storage on the compute nodes (but see  Data management at Niagara)
* For other clusters you can assume the available disk size to be at least 190GB

You can access this local disk inside of a job using the environment variable $SLURM_TMPDIR. One approach therefore would be to keep your dataset archived as a single tar file in the project space and then copy it to the local disk at the beginning of your job, extract it and use the dataset during the job. If any changes were made, at the job's end you could again archive the contents to a tar file and copy it back to the project space.

Here is an example of a submission script that allocates an entire node

== RAM disk ==
The /tmp file system can be used as a RAM disk on the compute nodes. It is implemented using tmpfs. Here is more information
* /tmp is tmpfs on all clusters
* /tmp is cleared at job end
* like all of a job's other memory use, falls under the cgroup limit corresponding to the sbatch request
* we set the tmpfs size via mount options at 100%, which could potentially confuse some scripts, since it means /tmp's size is shown as the node's MemTotal. For example, df reports /tmp size as the physical RAM size, which does not correspond to the sbatch request

==Archiving ==

===dar===
Disk archive utility, conceived of as a significant modernization of the venerable tar tool. For more information, see Dar.

===HDF5===
This is a high-performance binary file format that can be used to store a variety of different kinds of data, including extended objects such as matrices but also image data. There exist tools for manipulating HDF5 files in several common programming languages including Python (e.g. h5py). For more information, see HDF5.

===SQLite===

The SQLite software allows for the use of a relational database which resides entirely in a single file stored on disk, without the need for a database server. The data located in the file can be accessed using standard SQL (Structured Query Language) commands such as SELECT and there are APIs for several common programming languages. Using these APIs you can then interact with your SQLite database inside of a program written in C/C++, Python, R, Java and Perl. Modern relational databases contain datatypes for handling the storage of binary blobs, such as the contents of an image file, so storing a collection of 5 or 10 million small PNG or JPEG images inside of a single SQLite file may be much more practical than storing them as individual files. There is the overhead of creating the SQLite database and this approach assumes that you are familiar with SQL and designing a simple relational database with a small number of tables. Note as well that the performance of SQLite can start to degrade for very large database files, several gigabytes or more, in which case you may need to contemplate the use of a more traditional   database server using MySQL or PostgreSQL.

The SQLite executable is called sqlite3.  It is available via the nixpkgs module, which is loaded by default on our systems.

===Parallel compression===
When creating an archive from a significant number of files, it may be useful to use pigz instead of the traditional gzip to compress the archive.
"pigz -p 4" -f dir.tar.gz dir_to_tar}}
Here the archive will be compressed using 4 cores.

===Partial extraction from an archive===
Sometimes, it is not necessary to extract all the content of an archive but only part of it. For example, if the current simulation or job only needs files from a specific folder, this particular folder can be extracted from the archive and saved on the local disk using:

==Cleaning up hidden files==

==git==
When working with Git, over time the number of files in the hidden .git repository subdirectory can grow significantly. Using git repack will pack many of the files together into a few large database files and greatly speed up Git's operations.