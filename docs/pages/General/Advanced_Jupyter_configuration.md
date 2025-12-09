---
title: "Advanced Jupyter configuration/en"
url: "https://docs.alliancecan.ca/wiki/Advanced_Jupyter_configuration/en"
category: "General"
last_modified: "2025-09-23T21:59:54Z"
page_id: 18533
display_title: "Advanced Jupyter configuration"
---

= Introduction =

* Project Jupyter: "a non-profit, open-source project, born out of the IPython Project in 2014 as it evolved to support interactive data science and scientific computing across all programming languages."https://jupyter.org/about.html
* JupyterLab: "a web-based interactive development environment for notebooks, code, and data. Its flexible interface allows users to configure and arrange workflows in data science, scientific computing, computational journalism, and machine learning. A modular design allows for extensions that expand and enrich functionality."https://jupyter.org/

A JupyterLab server should only run on a compute node or on a cloud instance; cluster login nodes are not a good choice because they impose various limits which can stop applications if they consume too much CPU time or memory. In the case of using a compute node, users can reserve compute resources by submitting a job that requests a specific number of CPUs (and optionally GPUs), an amount of memory and the run time. In this page, we give detailed instructions on how to configure and submit a JupyterLab job on any national cluster.

If you are instead looking for a preconfigured Jupyter environment, please check the Jupyter page.

= Installing JupyterLab =

These instructions install JupyterLab with the pip command in a
Python virtual environment:

If you do not have an existing Python virtual environment, create one. Then, activate it:

Load a Python module, either the default one (as shown below) or
a specific version (see available versions with module avail python):
If you intend to use RStudio Server, make sure to load rstudio-server first:
Create a new Python virtual environment:
Activate your newly created Python virtual environment:

Install JupyterLab in your new virtual environment (note: it takes a few minutes):
In the virtual environment, create a wrapper script that launches JupyterLab:
Finally, make the script executable:

= Installing extensions =

Extensions allow you to add functionalities and modify the JupyterLab’s user interface.

=== Jupyter Lmod ===

Jupyter Lmod is an extension that allows you to interact with environment modules before launching kernels. The extension uses the Lmod's Python interface to accomplish module-related tasks like loading, unloading, saving a collection, etc.

The following commands will install and enable the Jupyter Lmod extension in your environment (note: the third command takes a few minutes to complete):

Instructions on how to manage loaded software modules in the JupyterLab interface are provided in the JupyterHub page.

=== RStudio Server ===

The RStudio Server allows you to develop R codes in an RStudio environment that appears in your web browser in a separate tab. Based on the above Installing JupyterLab procedure, there are a few differences:

Load the rstudio-server module before the python module and before creating a new virtual environment:
Once Jupyter Lab is installed in the new virtual environment, install the Jupyter RSession proxy:

All other configuration and usage steps are the same. In JupyterLab, you should see an RStudio application in the Launcher tab.

= Using your installation =

== Activating the environment ==

Make sure the Python virtual environment in which you have installed JupyterLab is activated.
For example, when you log onto the cluster, you have to activate it again with:
To verify that your environment is ready, you can get a list of installed jupyter* packages with the following command: grep jupyter
|result=
jupyter-client==7.1.0+computecanada
jupyter-core==4.9.1+computecanada
jupyter-server==1.9.0+computecanada
jupyterlab==3.1.7+computecanada
jupyterlab-pygments==0.1.2+computecanada
jupyterlab-server==2.3.0+computecanada
}}
== Starting JupyterLab ==

To start a JupyterLab server, submit an interactive job with salloc. Adjust the parameters based on your needs. See Running jobs for more information.
1:0:0 --ntasks1 --cpus-per-task2 --mem-per-cpu1024M --accountdef-yourpi srun $VIRTUAL_ENV/bin/jupyterlab.sh
|result=
...
[I 2021-12-06 10:37:14.262 ServerApp] jupyterlab  extension was successfully linked.
...
[I 2021-12-06 10:37:39.259 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2021-12-06 10:37:39.356 ServerApp]

    To access the server, open this file in a browser:
        file:///home/name/.local/share/jupyter/runtime/jpserver-198146-open.html
    Or copy and paste one of these URLs:
        http://node_name.int.cluster.computecanada.ca:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb
     or http://127.0.0.1:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb
}}
== Connecting to JupyterLab ==

To access JupyterLab running on a compute node from your web browser, you will need to create an SSH tunnel from your computer through the cluster since the compute nodes are not directly accessible from the internet.

=== From Linux or macOS ===

On a Linux or macOS system, we recommend using the sshuttle Python package.

On your computer, open a new terminal window and create the SSH tunnel with the following sshuttle command where  must be replaced with your Alliance account username, and  by the cluster on which you have launched JupyterLab:

Then, copy and paste the first provided HTTP address into your web browser. In the above salloc example, this would be:

http://node_name.int.cluster.alliancecan.ca:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb

=== From Windows ===

An SSH tunnel can be created from Windows using  MobaXTerm as follows. Note: this procedure also works from any terminal that supports the ssh command.

Once JupyterLab is launched on a compute node (see Starting JupyterLab), you can extract the hostname:port and the token from the first provided HTTP address. For example:
http://node_name.int.cluster.alliancecan.ca:8888/lab?token=101c368829...2728fad4eb
       └────────────────────┬────────────────────┘           └──────────┬──────────┘
                      hostname:port                                   token

Open a new Terminal tab in MobaXTerm. In the following command, replace  with its corresponding value (refer to the above figure), replace  with your Alliance account username, and replace  with the cluster on which you have launched JupyterLab:
Open your web browser and go to the following address where  must be replaced with the alphanumerical value extracted from the above figure:
http://localhost:8888/?token=

== Shutting down JupyterLab ==

You can shut down the JupyterLab server before the walltime limit by pressing Ctrl-C twice in the terminal that launched the interactive job.

If you have used MobaXterm to create an SSH tunnel, press Ctrl-D to shut down the tunnel.

= Adding kernels =

It is possible to add kernels for other programming languages, for a different Python version or for a persistent virtual environment that has all required packages and libraries for your project. Refer to Making kernels for Jupyter to learn more.

The installation of a new kernel is done in two steps:
# Installation of the packages that will allow the language interpreter to communicate with the Jupyter interface.
# Creation of a file that will indicate to JupyterLab how to initiate a communication channel with the language interpreter. This file is called a kernel spec file, and it will be saved in a subfolder of ~/.local/share/jupyter/kernels.

In the following sections, we provide a few examples of the kernel installation procedure.

== Julia Kernel ==

Prerequisites:
# The configuration of a Julia kernel depends on a Python virtual environment and a kernels folder. If you do not have these dependencies, make sure to follow the first few instructions listed in Python kernel (note: no Python kernel required).
# Since the installation of Julia packages requires an access to the internet, the configuration of a Julia kernel must be done in a remote shell session on a login node.

Once you have a Python virtual environment available and activated, you may configure the Julia kernel:

Load the Julia module:
Install IJulia: julia
}}
Important: start or restart a new JupyterLab session before using the Julia kernel.

For more information, see the IJulia documentation.

=== Installing more Julia packages ===

As in the above installation procedure, it is required to install Julia packages from a login node, but the Python virtual environment could be deactivated:

Make sure the same Julia module is loaded:
Install any required package. For example with Glob: julia
}}
The newly installed Julia packages should already be usable in a notebook executed by the Julia kernel.

== Python kernel ==

In a terminal with an active session on the remote server,
you may configure a Python virtual environment with all the required Python modules
and a custom Python kernel for JupyterLab.
Here are the initial steps for the simplest Jupyter configuration
in a new Python virtual environment:

If you do not have a Python virtual environment, create one. Then, activate it:

Start from a clean Bash environment (this is only required if you are using the Jupyter Terminal via JupyterHub for the creation and configuration of the Python kernel):$HOME bash -l
}}
Load a Python module:
Create a new Python virtual environment:
Activate your newly created Python virtual environment:

Create the common kernels folder, which is used by all kernels you want to install:
Finally, install the Python kernel:

Install the ipykernel library:
Generate the kernel spec file. Replace  with a name that will uniquely identify your kernel:

Important: start or restart a new JupyterLab session before using the Python kernel.

For more information, see the ipykernel documentation.

=== Installing more Python libraries ===

Based on the Python virtual environment configured in the previous section:

If you are using the Jupyter Terminal via JupyterHub, make sure the activated Python virtual environment is running in a clean Bash environment. See the above section for details.
Install any required library. For example, numpy:
The newly installed Python libraries can now be imported in any notebook using the Python 3.x Kernel.

== R Kernel ==

Prerequisites:
# The configuration of an R kernel depends on a Python virtual environment and a kernels folder. If you do not have these dependencies, make sure to follow the first few instructions listed in Python kernel (note: no Python kernel required).
# Since the installation of R packages requires an access to CRAN, the configuration of an R kernel must be done in a remote shell session on a login node.

Once you have a Python virtual environment available and activated, you may configure the R kernel:

Load an R module:
Install the R kernel dependencies (crayon, pbdZMQ, devtools) - this will take up to 10 minutes, and packages should be installed in a local directory like ~/R/x86_64-pc-linux-gnu-library/4.1:'http://cran.us.r-project.org')
}}
Install the R kernel.
Install the R kernel spec file.
Important: Start or restart a new JupyterLab session before using the R kernel.

For more information, see the IRkernel documentation.

=== Installing more R packages ===

The installation of R packages cannot be done from notebooks because there is no access to CRAN.
As in the above installation procedure, it is required to install R packages from a login node, but the Python virtual environment could be deactivated:

Make sure the same R module is loaded:
Start the R shell and install any required package. For example with doParallel:'http://cran.us.r-project.org')
}}
The newly installed R packages should already be usable in a notebook executed by the R kernel.

= Running notebooks as Python scripts =
For longer run or analysis, we need to submit a non-interactive job. We then need to convert our notebook to a Python script, create a submission script and submit it.

1. From the login node, create and activate a virtual environment, then install nbconvert if not already available.

2. Convert the notebook (or all notebooks) to Python scripts.

3. Create your submission script, and submit your job.

In your submission script, run your converted notebook with:
python mynotebook.py

and submit your non-interactive job:

= References =