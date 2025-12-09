---
title: "Samtools/en"
url: "https://docs.alliancecan.ca/wiki/Samtools/en"
category: "General"
last_modified: "2025-05-02T19:22:01Z"
page_id: 28329
display_title: "Samtools"
---

== Description ==
Samtools is a suite of programs for interacting with high-throughput sequencing data.
It is closely related to BCFtools and to HTSlib.  Primary documentation for all three
of these packages can be found at https://www.htslib.org/

* Samtools is for reading, writing, editing, indexing, and viewing files in SAM, BAM, or CRAM format
* BCFtools is for reading and writing files in BCF2, VCF, and gVCF format, and for calling, filtering, and summarizing SNP and short indel sequence variants
* HTSlib is a C-language library for reading and writing high-throughput sequencing data.  It is used by both Samtools and BCFtools.

This page does not cover all features of Samtools.  Please refer to Samtools for the complete list of all subtools.

To load the default version of samtools use module load samtools, e.g.:

For more on the module command, including how to find other versions of samtools, see Using modules

== General usage ==

SAMtools provides tools for manipulating alignments in SAM and BAM formats.
A common task is to convert SAM files ("Sequence Alignment/Map") to BAM files.
BAM files are compressed versions of SAM files and are much smaller in size; the "B" stands for "binary".
BAM files are easy to manipulate and are ideal for storing large nucleotide sequence alignments.

CRAM is a more recent format for the same type of data, and offers still greater compression.

=== Converting a SAM file to a BAM file ===

Prior to converting, verify if your SAM file carries a header section with character “@”. You can inspect the header section using the view command:

If the SAM file contains a header, either of these forms can be used to convert the data to BAM format:

If headers are absent, you can use the reference FASTA file to map the reads:

=== Sorting and indexing BAM files ===

You may also have to sort and index BAM files for many downstream applications

You can also convert a SAM file directly to a sorted BAM file using the shell pipe:

A sorted BAM file, together with its index file with extension .bai, is a common prerequisite for many other processes such as variant calling, feature counting, etc.

=== Processing multiple files with multithreading and/or GNU parallel ===

You will typically have more than one SAM file to process at one time.
A job script with a loop is a good way to handle multiple files, as in the following example:

Samtools typically runs on a single core by default but in some cases it may improve your efficiency to use multithreading or GNU parallel.

Samtools can take advantage of multiple cores ("multithreading") if given the -@ flag:

A different way to take advantage of multiple cores is to use GNU parallel to process multiple files concurrently:

 parallel -j ${SLURM_CPUS_PER_TASK} "time samtools view -bS {}  samtools sort -o {.}_mt_sorted.bam"
}}

The above script will execute view and sort on four SAM files concurrently.
If you have more input files, modify the --cpous-per-task request.