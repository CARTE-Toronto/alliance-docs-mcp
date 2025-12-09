---
title: "GATK/en"
url: "https://docs.alliancecan.ca/wiki/GATK/en"
category: "General"
last_modified: "2023-07-24T18:04:02Z"
page_id: 14578
display_title: "GATK"
---

The Genome Analysis Toolkit (GATK) is a set of bioinformatic tools for
analyzing high-throughput sequencing (HTS) and variant call format (VCF)
data. The toolkit is well established for germline short variant
discovery from whole genome and exome sequencing data.
It is a leading tool in variant discovery and best practices
for genomics research.

==Availability and module loading ==
We provide several versions of GATK. To access the version information, use
the module command

which gives you some information about GATK and versions

gatk/3.7
gatk/3.8
gatk/4.0.0.0
gatk/4.0.8.1
gatk/4.0.12.0
gatk/4.1.0.0
gatk/4.1.2.0
gatk/4.1.7.0
gatk/4.1.8.0
gatk/4.1.8.1
gatk/4.2.2.0
gatk/4.2.4.0
gatk/4.2.5.0

More specific information on any given version can be accessed with

As you can see, this module only has the StdEnv/2020 module as prerequisite
so it can be loaded with

or, given that StdEnv/2020 is loaded by default, simply with

==General usage ==
The later versions of GATK (>=4.0.0.0) provide a wrapper over the Java executables (.jar). Loading the GATK modules will automatically set most of the environmental variables you will need to successfully run GATK.

The module spider command also provides information on usage and some examples of the wrapper:

      Usage
      =====
      gatk [--java-options "-Xmx4G"] ToolName [GATK args]

      Examples
      ========
      gatk --java-options "-Xmx8G" HaplotypeCaller -R reference.fasta -I input.bam -O output.vcf

As you probably notice, there are some arguments to be passed directly to Java through the --java-options such as the maximum heap memory (-Xmx8G in the example, reserving 8 Gb of memory for the virtual machine). We recommend that you always use -DGATK_STACKTRACE_ON_USER_EXCEPTION=true since it will give you more information in case the program fails. This information can help you or us (if you need support) to solve the issue.
Note that all options passed to --java-options have to be within quotation marks.

=== Considerations regarding our systems ===

To use GATK on our systems, we recommend that you use the --tmp-dir option and set it to ${SLURM_TMPDIR} when in an sbatch job, so that the temporary files are redirected to the local storage.

Also, when using GenomicsDBImport, make sure to have the option --genomicsdb-shared-posixfs-optimizations enabled as it will Allow for optimizations to improve the usability and performance for shared Posix Filesystems(e.g. NFS, Lustre). If not possible or if you are using GNU parallel to run multiple intervals at the same time, please copy your database to ${SLURM_TMPDIR} and run it from there as your IO operations might disrupt the filesystem. ${SLURM_TMPDIR} is a local storage and therefore is not only faster, but the IO operations would not affect other users.

===Earlier versions than GATK 4 ===
Earlier versions of GATK do not have the gatk command. Instead, you have to call the jar file:

java -jar GenomeAnalysisTK.jar PROGRAM OPTIONS

However, GenomeAnalysisTK.jar must be in PATH. On our systems, the environmental variables $EBROOTPICARD for Picard (included in GATK >= 4) and $EBROOTGATK for GATK contain the path to the jar file, so the appropriate way to call GATK <= 3 is

module load nixpkgs/16.09 gatk/3.8
java -jar "${EBROOTGATK}"/GenomeAnalysisTK.jar PROGRAM OPTIONS

You can find the specific usage of GATK <= 3 in the GATK3 guide.

===Multicore usage ===
Most  GATK (>=4) tools are not multicore by default. This means that you should request only one core when calling them. Some tools use threads in some of the computations (e.g. Mutect2 has the --native-pair-hmm-threads) and therefore you can require more CPUs (most of them with up to 4 threads) for these computations. GATK4, however, does provide some Spark commands https://gatk.broadinstitute.org/hc/en-us/articles/360035890591-

Not all GATK tools use Spark

Tools that can use Spark generally have a note to that effect in their respective Tool Doc.

* Some GATK tools exist in distinct Spark-capable and non-Spark-capable versions. The "sparkified" versions have the suffix "Spark" at the end of their names. Many of these are still experimental; down the road we plan to consolidate them so that there will be only one version per tool.

* Some GATK tools only exist in a Spark-capable version. Those tools don't have the "Spark" suffix.

For the commands that do use Spark, you can request multiple CPUs. NOTE: Please provide the exact number of CPUs to the spark command.  For example if you requested 10 CPUs, use --spark-master local[10] instead of --spark-master local[*]. If you want to use multiple nodes to scale the Spark cluster, you have to first deploy a SPARK cluster and then set the appropriate variables in the GATK command.

==Running GATK via Apptainer==

If you encounter errors like IllegalArgumentException while using the installed modules on our clusters, we recommend that you try another workflow by using the program via Apptainer.

A Docker image of GATK can be found here and other versions are available on this page. You will need first to build an Apptainer image from the Docker image;
to get the latest version for example, you can run the following commands on the cluster

module load apptainer
apptainer build gatk.sif docker://broadinstitute/gatk

or to get a particular version:

module load apptainer
apptainer build gatk_VERSION.sif docker://broadinstitute/gatk:VERSION

In your SBATCH script, you should use something like this:

module load apptainer
apptainer exec -B /home -B /project -B /scratch -B /localscratch \
    /gatk.sif gatk [--java-options "-Xmx4G"] ToolName [GATK args]

For more information about Apptainer, watch the recorded Apptainer webinar.

==Frequently asked questions ==
===How do I add a read group (RG) tag in my bam file? ===
Assuming that you want to add a read group called tag to the file called input.bam, you can use the GATK/PICARD command AddOrReplaceReadGroups:

gatk  AddOrReplaceReadGroups \
    -I input.bam \
    -O output.bam \
    --RGLB tag \
    --RGPL ILLUMINA
    --RGPU tag \
    --RGSM tag \
    --SORT_ORDER 'coordinate' \
    --CREATE_INDEX true

This assumes that your input file is sorted by coordinates and will generate an index along with the annotated output (--CREATE_INDEX true)

===How do I deal with java.lang.OutOfMemoryError: Java heap space ===
Subprograms of GATK often require more memory to process your files. If you were not using the -Xms command, add it to the --java-options. For example, let's imagine that you run the following command:

gatk MarkDuplicates \
    -I input.bam \
    -O marked_duplicates.bam \
    -M marked_dup_metrics.txt

but it gives you the java.lang.OutOfMemoryError: Java heap space error. Try:

gatk MarkDuplicates \
    --java-options "-Xmx8G DGATK_STACKTRACE_ON_USER_EXCEPTION=true"
    -I input.bam \
    -O marked_duplicates.bam \
    -M marked_dup_metrics.txt

If it fails again, keep increasing the memory until you find the required memory for your particular dataset. If you are using any of our systems, remember to request enough memory for this.

If you are interested in knowing more about java heap space, you can start here.

===Increasing the heap memory does not fix java.lang.OutOfMemoryError: Java heap space ===
There are cases in which the memory issue cannot be fixed with increasing the heap memory. This often happens with non-model organisms, and you are using too many scaffolds in your reference. In this case it is recommended to remove small scaffolds and create subsets of your reference. This implies that you have to map multiple times and run the pipelines in each of the subsets. This approach does not work for all pipelines so review your results carefully. GATK is designed with the human genome in mind, and therefore other organisms will require adjustments in many parameters and pipelines.

===Using more resources than asked for ===
Sometimes GATK/JAVA applications will use more memory or CPUs/threads than the numbers requested. This is often generated by the JAVA garbage collection. To control this, add -XX:ConcGCThreads=1 to the --java-options argument.

===FAQ on GATK ===
You can find the GATK FAQs on their website.

=References =
GATK Home

GATK SPARK

Make GATK tools run faster