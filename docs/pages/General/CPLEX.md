---
title: "CPLEX/en"
url: "https://docs.alliancecan.ca/wiki/CPLEX/en"
category: "General"
last_modified: "2023-03-24T16:27:20Z"
page_id: 6706
display_title: "CPLEX"
---

CPLEX is software for optimization, developed by IBM and available for academic users through IBM's Academic Initiative.

==Download==
To use CPLEX on Compute Canada clusters, you must first register with IBM and then download a personal version of the software. If you are presented with a choice of architectures, choose Linux x86-64.

==Installation==
The file is an executable archive which will perform the installation after you have answered a few questions. To execute the archive, you need to type the command  bash ./cplex_studioXYZ.linux-x86.bin.

To access the software, you can create a personal module. Modules are normally created and placed in a directory hierarchy. In order for your modules to be found, you need to modify the configuration file $HOME/.bashrc so that it points to the root of this hierarchy, by adding the following line:

$HOME/modulefiles:$MODULEPATH}}

Next, you need to create a directory structure in which to put your new cplex module:

In this directory, you should create a file (e.g. $HOME/modulefiles/mycplex/12.8.0) with the version number corresponding to the version you downloaded (e.g. 12.8.0) and with the following content:

Adjust the lines which correspond to the variables cplexversion and studio_root so that they have the correct values for your situation, i.e. the version you downloaded and the path you specified when extracting the archive.

==Java==
If you use Java, you will have some further steps to carry out. Firstly, in your .bashrc file, you can add the line
.}}
which will allow your code to be found during execution.

Next, you should modify the dynamic library of CPLEX. Look for this library in the directory hierarchy of the installation directory, make a copy of it and then execute the command

It's possible that during your compilation, you receive an error message because of a lack of memory. In this case, you should request a compute node using an interactive job to do the compilation. For example,
1:0:0 --ntasks1 --mem-per-cpu8G}}

==Python==

After you have installed CPLEX as documented above, you must first install the module that you created:

To install the CPLEX Python modules like docplex, we suggest that you use a  virtual environment.

Once the virtual environment has been activated, you should go in the directory $STUDIO_ROOT/python after which you can install the module using the command:

The installation of these Python modules must be done on the login node because they are not available in our  software stack.