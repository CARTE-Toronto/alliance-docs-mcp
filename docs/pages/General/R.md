---
title: "R/en"
url: "https://docs.alliancecan.ca/wiki/R/en"
category: "General"
last_modified: "2025-11-13T14:20:02Z"
page_id: 2629
display_title: "R"
---

`<languages/>`{=html}

R is a system for statistical computation and graphics. It consists of a language plus a runtime environment with graphics, a debugger, access to certain system functions, and the ability to run programs stored in script files.

Even though R was not developed for high-performance computing (HPC), its popularity with scientists from a variety of disciplines, including engineering, mathematics, statistics, bioinformatics, etc. makes it an essential tool on HPC installations dedicated to academic research. Features such as C extensions, byte-compiled code and parallelization allow for reasonable performance in single-node jobs. Thanks to R's modular nature, users can customize the R functions available to them by installing packages from the Comprehensive R Archive Network ([CRAN](https://cran.r-project.org/)) into their home directories.

User Julie Fortin has written a blog post, [\"How to run your R script with Compute Canada\"](https://medium.com/the-nature-of-food/how-to-run-your-r-script-with-compute-canada-c325c0ab2973) which you might find useful.

## The R interpreter {#the_r_interpreter}

You need to begin by loading an R module; there will typically be several versions available and you can see a list of all of them using the command

You can load a particular R module using a command like

For more on this, see [Using modules.](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules."){.wikilink}

Now you can start the R interpreter and type R code inside that environment:

To execute an R script non-interactively, use `Rscript` with the file containing the R commands as an argument:

`Rscript` will automatically pass scripting-appropriate options `--slave` and `--no-restore` to the R interpreter. These also imply the `--no-save` option, preventing the creation of useless workspace files on exit.

Note that **any calculations lasting more than two or three minutes should not be run on the login node**. They should be run via the job scheduler.

A simple job script looks like this:

See [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} for more information.

## Installing R packages {#installing_r_packages}

### install.packages()

To install packages from [CRAN](https://cran.r-project.org/), you can use `install.packages` in an interactive R session on a cluster login node. Since the compute nodes on most clusters do not have access to the Internet, installing R packages in a batch or interactive job is not possible. Many R packages are developed using the GNU family of compilers so we recommend that you load a `gcc` [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink} before trying to install any R packages. Use the same version of the `gcc` for all packages you install.

#### Installing for a specific R version {#installing_for_a_specific_r_version}

For example, to install the `sp` package that provides classes and methods for spatial data, use the following command on a login node:

If the argument `repos` is not specified, you will be asked to select an appropriate mirror for download. Ideally, it will be geographically close to the cluster you\'re working on.

Some packages require defining the environment variable `TMPDIR` before installing.

#### Installing for one or many R versions {#installing_for_one_or_many_r_versions}

Specify the local installation directory according to the R module that is currently loaded. \~/.local/R/\$EBVERSIONR/ }} Install the package. \"<https://cloud.r-project.org/>\")\'}}

In your submission script, you then have to load the desired R module and set the local library directory with `export R_LIBS=~/.local/R/$EBVERSIONR/`.

### Dependencies

Some packages depend on external libraries which are already installed on our clusters. If the library you need is listed at [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink}, then load the appropriate [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink} before installing the package that requires it.

For example, the package `rgdal` requires a library called `gdal`. Running `module spider gdal/3.9.1` shows how to load this module.

If any package fails to install, be sure to read the error message carefully as it might give you details concerning additional modules you need to load. See [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules"){.wikilink} for more on the `module` family of commands.

### Downloaded packages {#downloaded_packages}

To install a package that you downloaded (i.e. not using `install.packages()`), you can install it as follows. Assuming the package is named `archive_package.tgz`, run the following command in a shell:

## Using system calls in R {#using_system_calls_in_r}

Using the R command `system()` you can execute commands in the ambient environment from inside R. On our clusters, this can lead to problems because R will give an incorrect value to the environment variable `LD_LIBRARY_PATH`. You can avoid this problem by using the syntax `system("LD_LIBRARY_PATH=$RSNT_LD_LIBRARY_PATH ``<my system call>`{=html}`")` in your R system calls.

## Passing arguments to R scripts {#passing_arguments_to_r_scripts}

Sometimes it can be useful to pass parameters as arguments to R scripts, to avoid having to either change the R script for every job or having to manage multiple copies of otherwise identical scripts. This can be useful for specifying the names for input- or output files, as well as specifying numerical parameters. For example, instead of specifying the name of an input file and/or a numerical parameter like this

and changing the code every time either of these changes, parameters can be passed to the R-script when starting it:

and the next

The following example expects exactly two arguments. The first one should be a string which will be used for the variable \"name\" and the second one should be an integer for the variable \"number\". `{{File
  |name=arguments_test.R
  |lang="R"
  |contents=
args = commandArgs(trailingOnly=TRUE)

# test if there is at least two arguments: if not, return an error
if (length(args)<2) {
  stop("At least two arguments must be supplied ('name' (text) and 'numer' (integer) )", call.=FALSE)
}

name      <- args[1]                # read first argument as string
number    <- as.integer( args[2] )  # read second argument as integer

print(paste("Processing with name:'", name, "' and number:'", number,"'", sep = ''))
}}`{=mediawiki}

This script can be used like this:

## Exploiting parallelism in R {#exploiting_parallelism_in_r}

The processors on our clusters are quite ordinary. What makes these supercomputers `<i>`{=html}super`</i>`{=html} is that you have access to thousands of CPU cores with a high-performance network. In order to take advantage of this hardware, you must run code \"in parallel.\" However, note that prior to investing a lot of time and effort in parallelizing your R code, you should first ensure that your serial implementation is as efficient as possible. As an interpreted language, the use of loops in R, and especially nested loops, constitutes a significant performance bottleneck. Whenever possible you should try to use vectorized forms of R functions and more functional elements of the R programming language like the family of `apply` functions and the `ifelse` function. This will frequently offer you a far better performance gain by eliminating a loop altogether instead of simply parallelizing the (slow) execution of this loop across several CPU cores.

The [CRAN Task View on High-Performance and Parallel Computing with R](https://cran.r-project.org/web/views/HighPerformanceComputing.html) describes a bewildering collection of interrelated R packages for parallel computing. For an excellent overview and advice, see the October 2023 Compute Ontario colloquium [\"High-Performance R\"](https://education.scinet.utoronto.ca/course/view.php?id=1333) ([slides](https://education.scinet.utoronto.ca/mod/resource/view.php?id=2887)).

The following subsections contain some further notes and examples.

`<b>`{=html}A note on terminology:`</b>`{=html} In most of our documentation the term \'node\' refers to an individual machine, also called a \'host\', and a collection of such nodes makes up a \'cluster\'. In a lot of R documentation however, the term \'node\' refers to a worker process and a \'cluster\' is a collection of such processes. As an example, consider the following quote, \"Following `<b>`{=html}snow`</b>`{=html}, a pool of worker processes listening *via* sockets for commands from the master is called a \'cluster\' of nodes.\"[^1].

### doParallel and foreach {#doparallel_and_foreach}

#### Usage

Foreach can be considered as a unified interface for all backends (i.e. doMC, doMPI, doParallel, doRedis, etc.). It works on all platforms, assuming that the backend works. doParallel acts as an interface between foreach and the parallel package and can be loaded alone. There are some [known efficiency issues](https://docs.alliancecan.ca/Scalability "known efficiency issues"){.wikilink} when using foreach to run a very large number of very small tasks. Therefore, keep in mind that the following code is not the best example of an optimized use of the foreach() call but rather that the function chosen was kept at a minimum for demonstration purposes.

You must register the backend by feeding it the number of cores available. If the backend is not registered, foreach will assume that the number of cores is 1 and will proceed to go through the iterations serially.

The general method to use foreach is:

1.  to load both foreach and the backend package;
2.  to register the backend;
3.  to call foreach() by keeping it on the same line as the %do% (serial) or %dopar% operator.

#### Running

1\. Place your R code in a script file, in this case the file is called *test_foreach.R*.

```{=mediawiki}
{{File
  |name=test_foreach.R
  |lang="r"
  |contents=
# library(foreach) # optional if using doParallel
library(doParallel) #

# a very simple function
test_func <- function(var1, var2) {
    # some heavy workload
    sum <- 0
    for (i in c(1:3141593)) {
        sum <- sum + var1 * sin(var2 / i)
    }
    return(sqrt(sum))
}

# we will iterate over two sets of values, you can modify this to explore the mechanism of foreach
var1.v = c(1:8)
var2.v = seq(0.1, 1, length.out = 8)

# Use the environment variable SLURM_CPUS_PER_TASK to set the number of cores.
# This is for SLURM. Replace SLURM_CPUS_PER_TASK by the proper variable for your system.
# Avoid manually setting a number of cores.
ncores = Sys.getenv("SLURM_CPUS_PER_TASK") 

registerDoParallel(cores=ncores)# Shows the number of Parallel Workers to be used
print(ncores) # this how many cores are available, and how many you have requested.
getDoParWorkers()# you can compare with the number of actual workers

# be careful! foreach() and %dopar% must be on the same line!
foreach(var1=var1.v, .combine=rbind) %:% foreach(var2=var2.v, .combine=rbind) %dopar% {test_func(var1=var1, var2=var2)}
}}
```
2\. Copy the following content in a job submission script called *job_foreach.sh*:

3\. Submit the job with:

For more on submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

### doParallel and makeCluster {#doparallel_and_makecluster}

#### Usage {#usage_1}

You must register the backend by feeding it the nodes name multiplied by the desired number of processes. For instance, with two nodes (node1 and node2) and two processes, we would create a cluster composed of : `node1 node1 node2 node2` hosts. The *PSOCK* cluster type will run commands through SSH connections into the nodes.

#### Running {#running_1}

1\. Place your R code in a script file, in this case the file is called `test_makecluster.R`. `{{File
  |name=test_makecluster.R
  |lang="r"
  |contents=
library(doParallel)

# Create an array from the NODESLIST environnement variable
nodeslist = unlist(strsplit(Sys.getenv("NODESLIST"), split=" "))

# Create the cluster with the nodes name. One process per count of node name.
# nodeslist = node1 node1 node2 node2, means we are starting 2 processes on node1, likewise on node2.
cl = makeCluster(nodeslist, type = "PSOCK") 
registerDoParallel(cl)

# Compute (Source : https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf)
x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10000

foreach(icount(trials), .combine=cbind) %dopar%
    {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
    }

# Don't forget to release resources
stopCluster(cl)
}}`{=mediawiki}

2\. Copy the following content in a job submission script called `job_makecluster.sh`:

`cut -f 1 -d '.'))`

R -f test_makecluster.R }}

In the above example the scheduler might place all four processes on just one node. This is okay, but if you wish to prove that the same job works even if the processes happen to be placed on different nodes, then add the line `#SBATCH --ntasks-per-node=2`

3\. Submit the job with:

For more information on submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

### Rmpi

#### Installing

This next procedure installs [Rmpi](https://cran.r-project.org/web/packages/Rmpi/index.html), an interface (wrapper) to MPI routines, which allow R to run in parallel.

1\. See the available R modules by running:

``` bash
module spider r
```

2\. Select the R version and load the required OpenMPI module.

``` bash
module load gcc/12.3
module load openmpi/4.1.5
module load r/4.5.0
```

3\. Download [the latest Rmpi version](https://cran.r-project.org/web/packages/Rmpi/index.html); change the version number to whatever is desired.

``` bash
wget https://cran.r-project.org/src/contrib/Rmpi_0.7-3.3.tar.gz
```

4\. Specify the directory where you want to install the package files; you must have write permission for this directory. The directory name can be changed if desired.

``` bash
mkdir -p ~/local/R_libs/
export R_LIBS=~/local/R_libs/
```

5\. Run the install command.

``` bash
R CMD INSTALL --configure-args="--with-Rmpi-include=$EBROOTOPENMPI/include   --with-Rmpi-libpath=$EBROOTOPENMPI/lib --with-Rmpi-type='OPENMPI' " Rmpi_0.7-3.3.tar.gz
```

Again, carefully read any error message that comes up when packages fail to install and load the required modules to ensure that all your packages are successfully installed.

#### Running {#running_2}

1\. Place your R code in a script file, in this case the file is called *test.R*.

2\. Copy the following content in a job submission script called *job.sh*:

3\. Submit the job with:

``` bash
sbatch job.sh
```

For more on submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

[^1]: Core package \"parallel\" vignette, <https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf>
