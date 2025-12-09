---
title: "CPLEX/en"
url: "https://docs.alliancecan.ca/wiki/CPLEX/en"
category: "General"
last_modified: "2023-03-24T16:27:20Z"
page_id: 6706
display_title: "CPLEX"
---

`<languages />`{=html}

[CPLEX](https://www.ibm.com/analytics/data-science/prescriptive-analytics/cplex-optimizer) is software for optimization, developed by IBM and available for academic users through [IBM\'s Academic Initiative](https://www.ibm.com/academic/home).

## Download

To use CPLEX on Compute Canada clusters, you must first [register with IBM](https://www.ibm.com/academic/home) and then download a personal version of the software. If you are presented with a choice of architectures, choose *Linux x86-64*.

## Installation

The file is an executable archive which will perform the installation after you have answered a few questions. To execute the archive, you need to type the command `bash ./cplex_studioXYZ.linux-x86.bin`.

To access the software, you can create a personal module. Modules are normally created and placed in a directory hierarchy. In order for your modules to be found, you need to modify the configuration file `$HOME/.bashrc` so that it points to the root of this hierarchy, by adding the following line:

\$HOME/modulefiles:\$MODULEPATH}}

Next, you need to create a directory structure in which to put your new cplex module:

In this directory, you should create a file (e.g. `$HOME/modulefiles/mycplex/12.8.0`) with the version number corresponding to the version you downloaded (e.g. 12.8.0) and with the following content: `{{Fichier
  |name=cplex_module.sh
  |lang="sh"
  |contents=

<!--T:9-->
#%Module1.0####
##
## cplex
##
proc ModulesHelp { } {
        global cplexversion

        <!--T:10-->
puts stderr "\tIBM ILOG cplex "
        puts stderr "\tThis module provides configuration for cplex, concert, cpoptimizer and opl"
}

<!--T:11-->
module-whatis   "IBM ILOG cplex (cplex, concert, cpoptimizer, opl). This version doesn't ask for a licence file."

<!--T:12-->
# for Tcl script use only
set     cplexversion        XYZ
set     studio_root          <root path of cplex>
set     cplexroot             $studio_root/cplex
set     concertroot           $studio_root/concert
set     oplroot               $studio_root/opl
set     cpoptimizerroot       $studio_root/cpoptimizer

<!--T:13-->
set cplexbin x86-64_linux
set cplexlib $cplexbin/static_pic
set concertbin x86-64_linux
set concertlib $concertbin/static_pic
set oplbin x86-64_linux
set opllib $oplbin/static_pic
set cpoptimizerbin x86-64_linux
set cpoptimizerlib $cpoptimizerbin/static_pic

<!--T:14-->
prepend-path    PATH         $cplexroot/bin/$cplexbin
prepend-path    PATH         $oplroot/bin/$oplbin
prepend-path    PATH         $cpoptimizerroot/bin/$cpoptimizerbin

<!--T:15-->
prepend-path    CPATH        $cplexroot/include
prepend-path    CPATH        $concertroot/include
prepend-path    CPATH        $oplroot/include
prepend-path    CPATH        $cpoptimizerroot/include

<!--T:16-->
prepend-path -d " "   CPATH_EXPANDED        -I$cplexroot/include
prepend-path -d " "  CPATH_EXPANDED        -I$concertroot/include
prepend-path -d " "   CPATH_EXPANDED        -I$oplroot/include
prepend-path -d " "   CPATH_EXPANDED        -I$cpoptimizerroot/include

<!--T:17-->
prepend-path    LIBRARY_PATH $cplexroot/lib/$cplexlib
prepend-path    LIBRARY_PATH $concertroot/lib/$concertlib
prepend-path    LIBRARY_PATH $oplroot/lib/$opllib
prepend-path    LIBRARY_PATH $oplroot/bin/x86-64_linux/
prepend-path    LIBRARY_PATH $cpoptimizerroot/lib/$cpoptimizerlib

<!--T:18-->
prepend-path -d " "   LIBRARY_PATH_EXPANDED -L$cplexroot/lib/$cplexlib
prepend-path -d " "  LIBRARY_PATH_EXPANDED -L$concertroot/lib/$concertlib
prepend-path -d " "   LIBRARY_PATH_EXPANDED -L$oplroot/lib/$opllib
prepend-path -d " "   LIBRARY_PATH_EXPANDED -L$oplroot/bin/x86-64_linux/
prepend-path -d " "   LIBRARY_PATH_EXPANDED  -L$cpoptimizerroot/lib/$cpoptimizerlib

<!--T:19-->
prepend-path    LD_LIBRARY_PATH      $cplexroot/bin/$cplexbin
prepend-path    LD_LIBRARY_PATH      $oplroot/bin/$oplbin

<!--T:20-->
prepend-path     CLASSPATH $cplexroot/lib/cplex.jar
prepend-path     MATLABPATH $cplexroot/matlab
prepend-path     STUDIO_ROOT $studio_root
<!--T:21-->
}}`{=mediawiki} Adjust the lines which correspond to the variables `cplexversion` and `studio_root` so that they have the correct values for your situation, i.e. the version you downloaded and the path you specified when extracting the archive.

## Java

If you use Java, you will have some further steps to carry out. Firstly, in your `.bashrc` file, you can add the line .}} which will allow your code to be found during execution.

Next, you should modify the dynamic library of CPLEX. Look for this library in the directory hierarchy of the installation directory, make a copy of it and then execute the command

It\'s possible that during your compilation, you receive an error message because of a lack of memory. In this case, you should request a compute node using an interactive job to do the compilation. For example, 1:0:0 \--ntasks1 \--mem-per-cpu8G}}

## Python

After you have installed [CPLEX](https://www.ibm.com/analytics/data-science/prescriptive-analytics/cplex-optimizer) as documented above, you must first install the module that you created:

To install the CPLEX Python modules like `docplex`, we suggest that you use a [ virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment " virtual environment"){.wikilink}.

Once the virtual environment has been activated, you should go in the directory `$STUDIO_ROOT/python` after which you can install the module using the command:

The installation of these Python modules must be done on the login node because they are not available in our [ software stack](https://docs.alliancecan.ca/Available_Python_wheels " software stack"){.wikilink}.
