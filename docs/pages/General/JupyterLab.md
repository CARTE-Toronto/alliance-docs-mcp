---
title: "JupyterLab/en"
url: "https://docs.alliancecan.ca/wiki/JupyterLab/en"
category: "General"
last_modified: "2026-01-08T16:15:02Z"
page_id: 15666
display_title: "JupyterLab"
---

= JupyterLab =
JupyterLab is the recommended general-purpose user interface to use on a JupyterHub.
From a JupyterLab server, you can manage your remote files and folders,
and you can launch Jupyter applications like a terminal, (Python 3) notebooks, RStudio and a Linux desktop.

You can add your own "kernels", which appear as application tiles described below.  To configure such kernels, please see Adding kernels.

== Launching JupyterLab ==
There are a few ways to launch JupyterLab.

The traditional way would be to use JupyterHub, but more recently, sites have deployed Open OnDemand which sometimes can launch the interface below. In the table below, the column "Fully-featured" indicates whether the JupyterLab interface available has all of the features described below. If there is a link, it is to that cluster's JupyterHub or Open OnDemand server.

Cluster  	JupyterHub	JupyterHub    	Open OnDemand                                            	Open OnDemand                                            	Open OnDemand
         	Available 	Fully-featured	Available                                                	JupyterLab                                               	Fully-featured
Fir      	Yes       	Yes           	No                                                       	No                                                       	No
Killarney	No        	No            	No                                                       	No                                                       	No
Narval   	Yes       	Yes           	No                                                       	No                                                       	No
Nibi     	No        	No            	Yes                                                      	Yes                                                      	Yes
Rorqual  	Yes       	Yes           	No                                                       	No                                                       	No
tamIA    	No        	No            	No                                                       	No                                                       	No
Trillium 	No        	No            	Yes, request "Jupyter Lab + Alliance software extensions"	Yes, request "Jupyter Lab + Alliance software extensions"	Yes, request "Jupyter Lab + Alliance software extensions"
Vulcan   	No        	No            	Yes                                                      	Yes                                                      	Yes

It is also possible to launch JupyterLab by installing it yourself in a virtual environment, but this is not recommended. You will also not benefit from any of the pre-configured applications described below.

== The JupyterLab interface ==

When you open JupyterLab in one of our most recent clusters, you will be presented with a dashboard pre-populated with a few launchers. Default launchers include Python 3.11, LibreQDA, Mate Desktop (VNC), OpenRefine, RStudio, VS Code and XFCE4 Desktop (VNC). In addition, you may find links to the cluster's Globus collection, to the cluster's job portal, as well as links to relevant documentation pages. By loading modules, you will see new launchers appear in the dashboard (see below).

In the menu bar on the top, please note that in order to close your session, you may do so through the File menu:
* Hub Control Panel: if you want to manually stop the JupyterLab server and the corresponding job on the cluster. This is useful when you want to start a new JupyterLab server with more or less resources.
* Log Out: the session will end, which will also stop the JupyterLab server and the corresponding job on the cluster.
Most other menu items are related to notebooks and Jupyter applications.

=== Tool selector on left ===
On the left side of the interface, you will find the tool selector. This changes the content of the frame on the right. The most relevant ones are:

==== File Browser (folder icon) ====
This is where you can browse in your home, project and scratch spaces. It is also possible to use it to upload files.

==== Running Terminals and Kernels (stop icon)  ====
This is to stop kernel sessions and terminal sessions

==== GPU Dashboards (GPU card icon)  ====
If your job uses GPUs, this will give you access to some resource monitoring options.

==== Software Modules ====

This is where you can load or unload software modules available in our environment. Depending on the modules loaded, icons directing to the corresponding Jupyter applications will appear in the Launcher tab. By default, we load a number of modules to provide you access to basic tools.

The search box can search for any available module and show the result in the Available Modules subpanel. Note: Some modules are hidden until their dependency is loaded: we recommend that you first look for a specific module with module spider module_name from a terminal.

The next subpanel is the list of Loaded Modules in the whole JupyterLab session.

The last subpanel is the list of Available modules, similar to the output of module avail. By clicking on a module's name, detailed information about the module is displayed. By clicking on the Load link, the module will be loaded and added to the Loaded Modules list.

=== Status bar at the bottom ===

* By clicking on the icons, this brings you to the Running Terminals and Kernels tool.

== Prebuilt applications ==

JupyterLab offers access to a terminal, an IDE (Desktop), a Python console and different options to create text and markdown files. This section presents only the main supported Jupyter applications that work with our software stack.

=== Applications that are available by default ===
A number of software modules are loaded by default, to give you access to those applications without any further actions.

==== Python ====

A Python kernel, with the default version, is automatically loaded. This allows you to start python notebooks automatically using the icon.

We load a default version of the Python software, but you may use a different one by loading another version of the ipython-kernel modules.

This python environment does not come with most pre-installed packages. However, you can load some modules, such as scipy-stack in order to get additional features.

You can also install python packages directly in the notebook's environment, by running

pip install --no-index package-name

in a cell of your notebook and then restarting your kernel.

==== VS Code ====

VS Code (Visual Studio Code) is a code editor originally developed by Microsoft, but which is an open standard on which code-server is based to make the application available through any browser.

The version which we have installed comes with a large number of extensions pre-installed. For more details, see our page on Visual Studio Code.

For a new session, the VS Code session can take up to 3 minutes to complete its startup.

It is possible to reopen an active VS Code session after the web browser tab was closed.

The VS Code session will end when the JupyterLab session ends.

==== LibreQDA ====

LibreQDA is an application for qualitative analysis, forked from Taguette.

This icon will launch a single-user version of the software, which can be used for text analysis.

For a new session, the LibreQDA session can take up to 3 minutes to complete its startup.

It is possible to reopen an active LibreQDA session after the web browser tab was closed.

The LibreQDA session will end when the JupyterLab session ends.

==== RStudio ====

RStudio is an integrated development environment primarily use for the R language.

We load a default version of the R software, but you may use a different one by loading another version of the rstudio-server modules. Please do so before launching RStudio, otherwise you may have to restart your JupyterLab session.

This RStudio launcher will open or reopen an RStudio interface in a new web browser tab.

It is possible to reopen an active RStudio session after the web browser tab was closed.

The RStudio session will end when the JupyterLab session ends.

Note that simply quitting RStudio or closing the RStudio and JupyterHub tabs in your browser will not release the resources (CPU, memory, GPU) nor end the underlying Slurm job.  Please end your session with the menu item File > Log Out on the JupyterLab browser tab.

==== MLflow ====
MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible. We load a default version of MLflow by default, but you may use a different version of it by loading a mlflow module. Please see our MLflow page for more information on how to use MLflow to track your AI experiments.

==== OpenRefine ====

OpenRefine is a powerful, free and open-source tool to clean up messy data, to transform it, and to extend it in order to add value to it.

It is commonly used to correct typos in manually collected survey data.

For a new session, the OpenRefine session can take up to 3 minutes to complete its startup.

It is possible to reopen an active OpenRefine session after the web browser tab was closed.

The OpenRefine session will end when the JupyterLab session ends.

==== Tensorboard ====
Tensorboard provides the visualization and tooling needed for machine learning experimentation. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more. We load a default version of tensorboard, but if a different module is available, you can change the version. See our page on Tensorboard for more details on using this software package.

==== Desktop ====

Two different Desktop environments are available by default. Mate Desktop, and XFCE Desktop. You may choose whichever you prefer. XFCE yields a more modern UI, while Mate is lighter to use.
These launchers will open or reopen a remote Linux desktop interface in a new web browser tab.

This is equivalent to running a VNC server on a compute node, then creating an SSH tunnel and finally using a VNC client, but you need nothing of all this with JupyterLab!

For a new session, the Desktop session can take up to 3 minutes to complete its startup.

It is possible to reopen an active desktop session after the web browser tab was closed.

The desktop session will end when the JupyterLab session ends.

==== Terminal ====

JupyterLab also natively allows you to open a terminal session. This may be useful to run bash commands, submit jobs, or edit files.

The terminal runs a (Bash) shell on the remote compute node without the need of an SSH connection.

Gives access to the remote filesystems (/home, /project, /scratch).

Allows running compute tasks.

The terminal allows copy-and-paste operations of text:

Copy operation: select the text, then press Ctrl+C.

Note: Usually, Ctrl+C is used to send a SIGINT signal to a running process, or to cancel the current command. To get this behaviour in JupyterLab's terminal, click on the terminal to deselect any text before pressing Ctrl+C.

Paste operation: press Ctrl+V.

==== Globus ====

If Globus is availalbe on the cluster you are using, you may see this icon. This will open your browser to the corresponding Globus collection.

==== Metrix ====

If the Metrix job portal is available on the cluster you are using, this icon will open a page with the statistics of your job.

=== Applications available after loading a module ===
Multiple of the modules we provide will also make a launcher available when they are loaded, even though they are not loaded by default.

==== Julia ====

Loading a module ijulia-kernel will allow you to open a notebook with the Julia language.

==== Ansys suite ====
The Ansys suite has multiple tools which provide a graphical user interface. If you load one of the ansys modules, you will get a series of launcher, most of which work through a VNC connection in the browser.

In addition, Ansys Fluent has a web-based interface, which can be launched with the icon below.

Note that for Ansys Fluent, a password is required to connect to it. That password is generated when you launch it, and written in your personal folder, in the file $HOME/fluent_webserver_token.

Note that for Ansys, you will need to provide your own license, as explained in our Ansys page.

==== Ansys EDT ====
Ansys EDT is in its own separate module. Loading the module ansysedt will make the corresponding launcher appear.

Note that for Ansys EDT, you will need to provide your own license, as explained in our Ansys EDT page.

==== COMSOL ====

COMSOL is a general-purpose software for modelling engineering applications.

Note that you will need to provide your own license file to use this software.

Loading a comsol module will add a launcher to start the graphical user interface for COMSOL through a VNC session. See our page on COMSOL for more details on using this software package.

==== Matlab ====
MATLAB  is available by loading a matlab module, which will add a launcher to start the software in a VNC session. Note that you will need to provide your own license file, as explained in our MATLAB page.

==== NVidia Nsight Systems ====
NVidia Nsight Systems is a performance analysis tool developed primarily for profiling GPUs, but which can profile CPU code as well.

Loading a cuda or a nvhpc module will madd a launcher to start the graphical user interface in a VNC session.

==== Octave ====
GNU Octave is an open-source scientific programming language largely compatible with MATLAB. Loading an octave module will add a launcher to start the graphical user interface for Octave through a VNC session. See our page on  Octave for more details on using this software package.

==== ParaView ====
ParaView is a powerful open-source visualisation software. Loading a paraview module will add a launcher to start the Paraview graphical user interface through a VNC session. See our page on ParaView for more details on using this software package.

==== QGIS ====
QGIS is a powerful open-source software for visualizing and processing geographic information systems (GIS) data.  Loading a qgis module will add a launcher to start the QGIS graphical user interface through a VNC session. See our page on QGIS for more details on this software package.

==== StarCCM+ ====
Siemens's Star-CCM+ is a commercial computational fluid dynamic simulation software. It is available by loading one of the starccm or the starccm-mixed modules, which will add a launcher to start the StarCCM+ graphical user interface through a VNC session. As for all commercial packages, you will need to provide your own license. See our page on Star-CCM+ for more details on using this software.

== Additional information on running Python notebooks ==

==== Python notebook ====

If any of the following scientific Python packages is required by your notebook, before you open this notebook, you must load the scipy-stack module from the JupyterLab Softwares tool:
* ipython, ipython_genutils, ipykernel, ipyparallel
* matplotlib
* numpy
* pandas
* scipy
* See SciPy stack for more on this

Note: You may also install needed packages by running for example the following command inside a cell: pip install --no-index numpy.
* For some packages (like plotly, for example), you may need to restart the notebook's kernel before importing the package.
* The installation of packages in the default Python kernel environment is temporary to the lifetime of the JupyterLab session; you will have to reinstall these packages the next time you start a new JupyterLab session. For a persistent Python environment, you must configure a custom Python kernel.

To open an existing Python notebook:
* Go back to the File Browser.
* Browse to the location of the *.ipynb file.
* Double-click on the *.ipynb file.
** This will open the Python notebook in a new JupyterLab tab.
** An IPython kernel will start running in the background for this notebook.

To open a new Python notebook in the current File Browser directory:
* Click on the Python 3.x launcher under the Notebook section.
** This will open a new Python 3 notebook in a new JupyterLab tab.
** A new IPython kernel will start running in the background for this notebook.

=== Running notebooks as Python scripts ===
1. From the console, or in a new notebook cell, install nbconvert :
!pip install --no-index nbconvert

2. Convert your notebooks to Python scripts
!jupyter nbconvert --to python my-current-notebook.ipynb

3. Create your non-interactive submission script, and submit it.

In your submission script, run your converted notebook with:
python my-current-notebook.py

And submit your non-interactive job: