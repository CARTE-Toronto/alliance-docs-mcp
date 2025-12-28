---
title: "R/en"
url: "https://docs.alliancecan.ca/wiki/R/en"
category: "General"
last_modified: "2025-12-16T18:27:00Z"
page_id: 2629
display_title: "R"
---

R is a system for statistical computation and graphics. It consists of a language plus a runtime environment with graphics, a debugger, access to certain system functions, and the ability to run programs stored in script files.

Even though R was not developed for high-performance computing (HPC), its popularity with scientists from a variety of disciplines, including engineering, mathematics, statistics, bioinformatics, etc. makes it an essential tool on HPC installations dedicated to academic research. Features such as C extensions, byte-compiled code and parallelization allow for reasonable performance in single-node jobs. Thanks to R’s modular nature, users can customize the R functions available to them by installing packages from the Comprehensive R Archive Network (CRAN) into their home directories.

User Julie Fortin has written a blog post, "How to run your R script with Compute Canada" which you might find useful.

== The R interpreter ==
You need to begin by loading an R module; there will typically be several versions available and you can see a list of all of them using the command

You can load a particular R module using a command like

For more on this, see Using modules.

Now you can start the R interpreter and type R code inside that environment:

To execute an R script non-interactively, use Rscript with the file containing the R commands as an argument:

Rscript will automatically pass scripting-appropriate options --slave and --no-restore to the R interpreter. These also imply the --no-save option, preventing the creation of useless workspace files on exit.

Note that any calculations lasting more than two or three minutes should not be run on the login node.
They should be run via the job scheduler.

A simple job script looks like this:

See Running jobs for more information.

== Installing R packages ==

=== install.packages() ===

To install packages from CRAN, you can use install.packages in an interactive R session on a cluster login node. Since the compute nodes on most clusters do not have access to the Internet, installing R packages in a batch or interactive job is not possible. Many R packages are developed using the GNU family of compilers so we recommend that you load a gcc module before trying to install any R packages. Use the same version of the gcc for all packages you install.

==== Installing for a specific R version ====
For example, to install the sp package that provides classes and methods for spatial data, use the following command on a login node:

If the argument repos is not specified, you will be asked to select an appropriate mirror for download. Ideally, it will be geographically close to the cluster you're working on.

Some packages require defining the environment variable TMPDIR before installing.

==== Installing for one or many R versions ====
Specify the local installation directory according to the R module that is currently loaded.
~/.local/R/$EBVERSIONR/
}}
Install the package.
"https://cloud.r-project.org/")'}}

In your submission script, you then have to load the desired R module and set the local library directory with export R_LIBS=~/.local/R/$EBVERSIONR/.

=== Dependencies ===
Some packages depend on external libraries which are already installed on our clusters. If the library you need is listed at Available software, then load the appropriate module before installing the package that requires it.

For example, the package rgdal requires a library called gdal. Running module spider gdal/3.9.1 shows how to load this module.

If any package fails to install, be sure to read the error message carefully as it might give you details concerning additional modules you need to load.  See Using modules for more on the module family of commands.

=== Downloaded packages ===
To install a package that you downloaded (i.e. not using install.packages()), you can install it as follows. Assuming the package is named archive_package.tgz, run the following command in a shell:

==Using system calls in R==

Using the R command system() you can execute commands in the ambient environment from inside R. On our clusters, this can lead to problems because R will give an incorrect value to the environment variable LD_LIBRARY_PATH. You can avoid this problem by using the syntax system("LD_LIBRARY_PATH=$RSNT_LD_LIBRARY_PATH ") in your R system calls.

== Passing arguments to R scripts ==
Sometimes it can be useful to pass parameters as arguments to R scripts, to avoid having to either change the R script for every job or having to manage multiple copies of otherwise identical scripts. This can be useful for specifying the names for input- or output files, as well as specifying numerical parameters. For example, instead of specifying the name of an input file and/or a numerical parameter like this

and changing the code every time either of these changes, parameters can be passed to the R-script when starting it:

and the next

The following example expects exactly two arguments. The first one should be a string which will be used for the variable "name" and the second one should be an integer for the variable "number".

This script can be used like this:

==Exploiting parallelism in R==

The processors on our clusters are quite ordinary.
What makes these supercomputers super is that you have access to thousands of CPU cores with a high-performance network.
In order to take advantage of this hardware, you must run code "in parallel." However, note that prior to investing a lot of time and effort
in parallelizing your R code, you should first ensure that your serial implementation is as efficient as possible. As an interpreted
language, the use of loops in R, and especially nested loops, constitutes a significant performance bottleneck. Whenever possible you
should try to use vectorized forms of R functions and more functional elements of the R programming language like the family of
apply functions and the ifelse function. This will frequently offer you a far better performance gain by eliminating
a loop altogether instead of simply parallelizing the (slow) execution of this loop across several CPU cores.

The CRAN Task View on High-Performance and Parallel Computing with R
describes a bewildering collection of interrelated R packages for parallel computing.
For an excellent overview and advice, see the October 2023 Compute Ontario colloquium
"High-Performance R"
(slides).

The following subsections contain some further notes and examples.

A note on terminology: In most of our documentation the term 'node' refers
to an individual machine, also called a 'host', and a collection of such nodes makes up a 'cluster'.
In a lot of R documentation however, the term 'node' refers to a worker process and a 'cluster' is a
collection of such processes. As an example, consider the following quote, "Following snow, a pool
of worker processes listening via sockets for commands from the master is called a 'cluster' of
nodes."Core package "parallel" vignette, https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf.

=== doParallel and foreach ===
====Usage====
Foreach can be considered as a unified interface for all backends (i.e. doMC, doMPI, doParallel, doRedis, etc.). It works on all platforms, assuming that the backend works. doParallel acts as an interface between foreach and the parallel package and can be loaded alone. There are some known efficiency issues when using foreach to run a very large number of very small tasks. Therefore, keep in mind that the following code is not the best example of an optimized use of the foreach() call but rather that the function chosen was kept at a minimum for demonstration purposes.

You must register the backend by feeding it the number of cores available. If the backend is not registered, foreach will assume that the number of cores is 1 and will proceed to go through the iterations serially.

The general method to use foreach is:
# to load both foreach and the backend package;
# to register the backend;
# to call foreach() by keeping it on the same line as the %do% (serial) or %dopar% operator.

====Running====

1. Place your R code in a script file, in this case the file is called test_foreach.R.

2. Copy the following content in a job submission script called job_foreach.sh:

3. Submit the job with:

For more on submitting jobs, see Running jobs.

=== doParallel and makeCluster ===
====Usage====
You must register the backend by feeding it the nodes name multiplied by the desired number of processes. For instance, with two nodes (node1 and node2) and two processes, we would create a cluster composed of : node1 node1 node2 node2 hosts. The PSOCK cluster type will run commands through SSH connections into the nodes.

====Running====
1. Place your R code in a script file, in this case the file is called test_makecluster.R.

2. Copy the following content in a job submission script called job_makecluster.sh:
 cut -f 1 -d '.'))
R -f test_makecluster.R
}}

In the above example the scheduler might place all four processes on just one node.
This is okay, but if you wish to prove that the same job works even if the processes happen
to be placed on different nodes, then add the line #SBATCH --ntasks-per-node=2

3. Submit the job with:

For more information on submitting jobs, see Running jobs.

=== Rmpi ===

====Installing====
This next procedure installs Rmpi, an interface (wrapper) to MPI routines, which allow R to run in parallel.

1. See the available R modules by running:

module spider r

2.  Select the R version and load the required OpenMPI module.

module load gcc/12.3
module load openmpi/4.1.5
module load r/4.5.0

3. Download the latest Rmpi version; change the version number to whatever is desired.

wget https://cran.r-project.org/src/contrib/Rmpi_0.7-3.3.tar.gz

4. Specify the directory where you want to install the package files; you must have write permission for this directory. The directory name can be changed if desired.

mkdir -p ~/local/R_libs/
export R_LIBS=~/local/R_libs/

5. Run the install command.

R CMD INSTALL --configure-args="--with-Rmpi-include=$EBROOTOPENMPI/include   --with-Rmpi-libpath=$EBROOTOPENMPI/lib --with-Rmpi-type='OPENMPI' " Rmpi_0.7-3.3.tar.gz

Again, carefully read any error message that comes up when packages fail to install and load the required modules to ensure that all your packages are successfully installed.

====Running====

1. Place your R code in a script file, in this case the file is called test.R.

2. Copy the following content in a job submission script called job.sh:

3. Submit the job with:

sbatch job.sh

For more on submitting jobs, see Running jobs.