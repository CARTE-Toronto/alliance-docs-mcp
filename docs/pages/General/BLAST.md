---
title: "BLAST/en"
url: "https://docs.alliancecan.ca/wiki/BLAST/en"
category: "General"
last_modified: "2024-01-25T20:21:02Z"
page_id: 9806
display_title: "BLAST"
---

BLAST ("Basic Local Alignment Search Tool") finds regions of similarity between biological sequences. The program compares nucleotide or protein sequences to sequence databases and calculates the statistical significance.

== User manual ==
You can find more information on its arguments in the user manual
or with

== Databases ==
Some frequently used sequence databases are installed on the clusters in /cvmfs/bio.data.computecanada.ca/content/databases/Core/blast_dbs/2022_03_23/.
Examine that directory and its subdirectories, e.g. with

== Accelerating the search ==
For the examples below, the file ref.fa will be used as the reference database in FASTA format, and seq.fa as the queries.

=== makeblastdb ===
Before running a search, we must build the database. This can be a preprocessing job, where the other jobs are dependent on the completion of the makeblastdb job.
Here is an example of a submission script:

=== Task array ===
BLAST search can greatly benefit from data parallelism by splitting the query file into multiples queries and running these queries against the database.

==== Preprocessing ====
In order to accelerate the search, the seq.fa file must be split into smaller chunks. These should be at least 1MB or greater, but not smaller as it may hurt the parallel filesystem.

Using the faSplit utility:

will create 10 files named seqN.fa where N is in the range of [0..9] for 10 queries (sequences).

==== Job submission ====
Once our queries are split, we can create a task for each seq.fa.N file using a job array. The task id from the array will map to the file name containing the query to run.

This solution allows the scheduler to fit the smaller jobs from the array where there are resources available in the cluster.

With the above submission script, we can submit our search and it will run after the database has been created.
afterok:$(sbatch makeblastdb.sh) blastn_array.sh}}

Once all the tasks from the array are done, the results can be concatenated using

where the 10 files will be concatenated into seq.ref file.
This could be done from the login node or as a dependent job upon completion of all the tasks from the array.

=== GNU Parallel ===
GNU Parallel is a great tool to pack many small jobs into a single job, and parallelize it.
This solution helps alleviate the issue of too many small files in a parallel filesystem by querying fixed size chunks from seq.fa and running on one node and multiple cores.

As an example, if your seq.fa file is 3MB, you could read blocks of 1MB and GNU Parallel will create 3 jobs, thus using 3 cores. If we would have requested 10 cores in our task, we would have wasted 7 cores. Therefore, the block size is important. We can also let GNU Parallel decide, as done below.

See also Handling large files in the GNU Parallel page.

==== Running with multiple cores on one node====

Note: The file must not be compressed.

===== Job submission =====
With the above submission script, we can submit our search and it will run after the database has been created.
afterok:$(sbatch makeblastdb.sh) blastn_gnu.sh}}

=== Additional tips ===
* If it fits into the node's local storage, copy your FASTA database to the local scratch space ($SLURM_TMPDIR).
* Reduce the number of hits returned (-max_target_seqs, -max_hsps can help), if it is reasonable for your research.
* Limit your hit list to nearly identical hits using -evalue filters, if it is reasonable for your research.