---
title: "JupyterLab/en"
url: "https://docs.alliancecan.ca/wiki/JupyterLab/en"
category: "General"
last_modified: "2025-12-03T19:36:53Z"
page_id: 15666
display_title: "JupyterLab/en"
---

`<languages />`{=html}

# JupyterLab

JupyterLab is the recommended general-purpose user interface to use on a JupyterHub. From a JupyterLab server, you can manage your remote files and folders, and you can launch Jupyter applications like a terminal, (Python 3) notebooks, RStudio and a Linux desktop.

You can add your own \"kernels\", which appear as application tiles described below. To configure such kernels, please see [Adding kernels](https://docs.alliancecan.ca/JupyterNotebook#Adding_kernels "wikilink").

## Launching JupyterLab {#launching_jupyterlab}

There are a few ways to launch JupyterLab.

The traditional way would be to use [JupyterHub](https://docs.alliancecan.ca/JupyterHub#JupyterHub_on_clusters "wikilink"), but more recently, sites have deployed Open OnDemand which sometimes can launch the interface below. In the table below, the column \"Fully-featured\" indicates whether the JupyterLab interface available has all of the features described below. If there is a link, it is to that cluster\'s JupyterHub or Open OnDemand server.

  Cluster     JupyterHub                                          
  ----------- --------------------------------------------------- ----------------
              Available                                           Fully-featured
  Fir         [Yes](https://jupyterhub.fir.alliancecan.ca/)       
  Killarney   No                                                  
  Narval      [Yes](https://jupyterhub.narval.alliancecan.ca/)    
  Nibi        No                                                  
  Rorqual     [Yes](https://jupyterhub.rorqual.alliancecan.ca/)   
  tamIA       No                                                  
  Trillium    No                                                  
  Vulcan      No                                                  

It is also possible to launch JupyterLab by [installing it yourself in a virtual environment](https://docs.alliancecan.ca/Advanced_Jupyter_configuration "wikilink"), but this is not recommended. You will also not benefit from any of the pre-configured applications described below.

## The JupyterLab interface {#the_jupyterlab_interface}

When you open JupyterLab in one of our most recent clusters, you will be presented with a dashboard pre-populated with a few launchers. Default launchers include Python 3.11, LibreQDA, Mate Desktop (VNC), OpenRefine, RStudio, VS Code and XFCE4 Desktop (VNC). In addition, you may find links to the cluster\'s [Globus](https://docs.alliancecan.ca/Globus "wikilink") collection, to the cluster\'s job portal, as well as links to relevant documentation pages. By loading modules, you will see new launchers appear in the dashboard (see below).

In the menu bar on the top, please note that in order to close your session, you may do so through the *File* menu:

-   `<i>`{=html}Hub Control Panel`</i>`{=html}: if you want to manually stop the JupyterLab server and the corresponding job on the cluster. This is useful when you want to start a new JupyterLab server with more or less resources.
-   `<i>`{=html}Log Out`</i>`{=html}: the session will end, which will also stop the JupyterLab server and the corresponding job on the cluster.

Most other menu items are related to notebooks and Jupyter applications. ![Default home tab when JupyterLab is loaded](https://docs.alliancecan.ca/JupyterLab_Launcher_with_modules.png "Default home tab when JupyterLab is loaded"){width="750" height="750"}

### Tool selector on left {#tool_selector_on_left}

On the left side of the interface, you will find the tool selector. This changes the content of the frame on the right. The most relevant ones are:

#### `<i>`{=html}File Browser`</i>`{=html} (folder icon) {#file_browser_folder_icon}

This is where you can browse in your home, project and scratch spaces. It is also possible to use it to upload files. ![File browser](https://docs.alliancecan.ca/File_browser.png "File browser")

#### `<i>`{=html}Running Terminals and Kernels`</i>`{=html} (stop icon) {#running_terminals_and_kernels_stop_icon}

This is to stop kernel sessions and terminal sessions

#### `<i>`{=html}GPU Dashboards`</i>`{=html} (GPU card icon) {#gpu_dashboards_gpu_card_icon}

If your job uses GPUs, this will give you access to some resource monitoring options.

#### `<i>`{=html}Software Modules`</i>`{=html} {#software_modules}

![Software module selector](https://docs.alliancecan.ca/Software_module_selector.png "Software module selector") This is where you can load or unload [software modules](https://docs.alliancecan.ca/Available_software "wikilink") available in our environment. Depending on the modules loaded, icons directing to the corresponding [Jupyter applications](https://docs.alliancecan.ca/#Prebuilt_applications "wikilink") will appear in the `<i>`{=html}Launcher`</i>`{=html} tab. By default, we load a number of modules to provide you access to basic tools.

The search box can search for any [available module](https://docs.alliancecan.ca/Available_software "wikilink") and show the result in the `<i>`{=html}Available Modules`</i>`{=html} subpanel. Note: Some modules are hidden until their dependency is loaded: we recommend that you first look for a specific module with `module spider module_name` from a terminal.

The next subpanel is the list of `<i>`{=html}Loaded Modules`</i>`{=html} in the whole JupyterLab session.

The last subpanel is the list of `<i>`{=html}Available modules`</i>`{=html}, similar to the output of `module avail`. By clicking on a module\'s name, detailed information about the module is displayed. By clicking on the `<i>`{=html}Load`</i>`{=html} link, the module will be loaded and added to the `<i>`{=html}Loaded Modules`</i>`{=html} list.

### Status bar at the bottom {#status_bar_at_the_bottom}

-   By clicking on the icons, this brings you to the `<i>`{=html}Running Terminals and Kernels`</i>`{=html} tool.

## Prebuilt applications {#prebuilt_applications}

JupyterLab offers access to a terminal, an IDE (Desktop), a Python console and different options to create text and markdown files. This section presents only the main supported Jupyter applications that work with our software stack.

### Applications that are available by default {#applications_that_are_available_by_default}

A number of software modules are loaded by default, to give you access to those applications without any further actions.

#### Python

![Python launcher icon](https://docs.alliancecan.ca/Python_launcher_icon.png "Python launcher icon") A Python kernel, with the default version, is automatically loaded. This allows you to start python notebooks automatically using the icon.

We load a default version of the Python software, but you may use a different one by loading another version of the `ipython-kernel` modules.

This python environment does not come with most pre-installed packages. However, you can load some modules, such as `scipy-stack` in order to get additional features.

You can also install python packages directly in the notebook\'s environment, by running

`pip install --no-index package-name`

in a cell of your notebook and then restarting your kernel.

#### VS Code {#vs_code}

![VS Code launcher icon](https://docs.alliancecan.ca/VS_Code_launcher_icon.png "VS Code launcher icon") VS Code (Visual Studio Code) is a code editor originally developed by Microsoft, but which is an open standard on which [code-server](https://github.com/coder/code-server) is based to make the application available through any browser.

The version which we have installed comes with a large number of [extensions](https://github.com/ComputeCanada/easybuild-easyconfigs-installed-avx2/blob/main/2023/code-server/code-server-4.101.2.eb#L27) pre-installed. For more details, see our page on [Visual Studio Code](https://docs.alliancecan.ca/Visual_Studio_Code "wikilink").

For a new session, the `<i>`{=html}VS Code`</i>`{=html} session can take up to 3 minutes to complete its startup.

It is possible to reopen an active VS Code session after the web browser tab was closed.

The VS Code session will end when the JupyterLab session ends.

#### LibreQDA

![LibreQDA launcher icon](https://docs.alliancecan.ca/LibreQDA_launcher_icon.png "LibreQDA launcher icon") `<i>`{=html}[LibreQDA](https://aide.libreqda.org/)`</i>`{=html} is an application for qualitative analysis, forked from [Taguette](https://www.taguette.org/).

This icon will launch a single-user version of the software, which can be used for text analysis.

For a new session, the `<i>`{=html}LibreQDA`</i>`{=html} session can take up to 3 minutes to complete its startup.

It is possible to reopen an active LibreQDA session after the web browser tab was closed.

The LibreQDA session will end when the JupyterLab session ends.

#### RStudio

![RStudio launcher icon](https://docs.alliancecan.ca/RStudio_launcher_icon.png "RStudio launcher icon") [RStudio](https://posit.co/download/rstudio-desktop/) is an integrated development environment primarily use for the [R](https://docs.alliancecan.ca/R "wikilink") language.

We load a default version of the R software, but you may use a different one by loading another version of the `rstudio-server` modules. Please do so **before** launching RStudio, otherwise you may have to restart your JupyterLab session.

This `<i>`{=html}RStudio`</i>`{=html} launcher will open or reopen an RStudio interface in a new web browser tab.

It is possible to reopen an active RStudio session after the web browser tab was closed.

The RStudio session will end when the JupyterLab session ends.

Note that simply quitting RStudio or closing the RStudio and JupyterHub tabs in your browser will not release the resources (CPU, memory, GPU) nor end the underlying Slurm job. `<b>`{=html}Please end your session with the menu item `File > Log Out` on the JupyterLab browser tab`</b>`{=html}.

#### OpenRefine

![OpenRefine launcher icon](https://docs.alliancecan.ca/OpenRefine_launcher_icon.png "OpenRefine launcher icon") [OpenRefine](https://openrefine.org/) is a powerful, free and open-source tool to clean up messy data, to transform it, and to extend it in order to add value to it.

It is commonly used to correct typos in manually collected survey data.

For a new session, the `<i>`{=html}OpenRefine`</i>`{=html} session can take up to 3 minutes to complete its startup.

It is possible to reopen an active OpenRefine session after the web browser tab was closed.

The OpenRefine session will end when the JupyterLab session ends.

#### Desktop

![Desktop launchers](https://docs.alliancecan.ca/Desktop_launchers.png "Desktop launchers") Two different Desktop environments are available by default. [Mate Desktop](https://mate-desktop.org/), and [XFCE Desktop](https://www.xfce.org/). You may choose whichever you prefer. XFCE yields a more modern UI, while Mate is lighter to use. These launchers will open or reopen a remote Linux desktop interface in a new web browser tab.

This is equivalent to running a [VNC server on a compute node](https://docs.alliancecan.ca/VNC#Compute_Nodes "wikilink"), then creating an [SSH tunnel](https://docs.alliancecan.ca/SSH_tunnelling "wikilink") and finally using a [VNC client](https://docs.alliancecan.ca/VNC#Setup "wikilink"), but you need nothing of all this with JupyterLab!

For a new session, the `<i>`{=html}Desktop`</i>`{=html} session can take up to 3 minutes to complete its startup.

It is possible to reopen an active desktop session after the web browser tab was closed.

The desktop session will end when the JupyterLab session ends.

#### Terminal

![Terminal launcher](https://docs.alliancecan.ca/Terminal_launcher.png "Terminal launcher") JupyterLab also natively allows you to open a terminal session. This may be useful to run bash commands, submit jobs, or edit files.

The terminal runs a (Bash) shell on the remote compute node without the need of an SSH connection.

Gives access to the remote filesystems (`/home`, `/project`, `/scratch`).

Allows running compute tasks.

The terminal allows copy-and-paste operations of text:

Copy operation: select the text, then press Ctrl+C.

Note: Usually, Ctrl+C is used to send a SIGINT signal to a running process, or to cancel the current command. To get this behaviour in JupyterLab\'s terminal, click on the terminal to deselect any text before pressing Ctrl+C.

Paste operation: press Ctrl+V.

#### Globus

![Globus launcher](https://docs.alliancecan.ca/Globus_launcher.png "Globus launcher") If [Globus](https://docs.alliancecan.ca/Globus "wikilink") is availalbe on the cluster you are using, you may see this icon. This will open your browser to the corresponding Globus collection.

#### Metrix

![Metrix launcher](https://docs.alliancecan.ca/Metrix_launcher.png "Metrix launcher") If the [Metrix job portal](https://docs.alliancecan.ca/Metrix/fr "wikilink") is available on the cluster you are using, this icon will open a page with the statistics of your job.

### Applications available after loading a module {#applications_available_after_loading_a_module}

Multiple of the modules we provide will also make a launcher available when they are loaded, even though they are not loaded by default.

#### Julia

![Julia launcher](https://docs.alliancecan.ca/Julia_launcher.png "Julia launcher") Loading a module `ijulia-kernel` will allow you to open a notebook with the Julia language.

#### Ansys suite {#ansys_suite}

The [Ansys](https://docs.alliancecan.ca/Ansys "wikilink") suite has multiple tools which provide a graphical user interface. If you load one of the `ansys` modules, you will get a series of launcher, most of which work through a VNC connection in the browser.

+--------------------------------------------------------------------+-----------------------------------------------------------------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
| ![Ansys CFX launcher](https://docs.alliancecan.ca/Ansys_CFX_launcher.png "Ansys CFX launcher") | ![Ansys Fluent launcher](https://docs.alliancecan.ca/Ansys_Fluent_launcher.png "Ansys Fluent launcher") | ![Ansys Mapdl launcher](https://docs.alliancecan.ca/Ansys_Mapdl_launcher.png "Ansys Mapdl launcher") | ![Ansys Workbench launcher](https://docs.alliancecan.ca/Ansys_Workbench_launcher.png "Ansys Workbench launcher") |
+====================================================================+=============================================================================+==========================================================================+======================================================================================+
+--------------------------------------------------------------------+-----------------------------------------------------------------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------------+

In addition, Ansys Fluent has a web-based interface, which can be launched with the icon below. ![Ansys Fluent web launcher](https://docs.alliancecan.ca/Ansys_Fluent_web_launcher.png "Ansys Fluent web launcher") Note that for Ansys Fluent, a password is required to connect to it. That password is generated when you launch it, and written in your personal folder, in the file `$HOME/fluent_webserver_token`.

Note that for Ansys, you will need to provide your own license, as explained in our [Ansys](https://docs.alliancecan.ca/Ansys "wikilink") page.

#### Ansys EDT {#ansys_edt}

[Ansys EDT](https://www.ansys.com/products/electronics) is in its own separate module. Loading the module `ansysedt` will make the corresponding launcher appear.

Note that for Ansys EDT, you will need to provide your own license, as explained in our [Ansys EDT](https://docs.alliancecan.ca/AnsysEDT "wikilink") page. ![Ansys EDT launcher](https://docs.alliancecan.ca/Ansys_EDT_launcher.png "Ansys EDT launcher")

#### COMSOL

![COMSOL launcher](https://docs.alliancecan.ca/COMSOL_launcher.png "COMSOL launcher") [COMSOL](http://www.comsol.com) is a general-purpose software for modelling engineering applications.

Note that you will need to provide your own license file to use this software.

Loading a `comsol` module will add a launcher to start the graphical user interface for COMSOL through a VNC session. See our page on [COMSOL](https://docs.alliancecan.ca/COMSOL "wikilink") for more details on using this software package.

#### Matlab

[MATLAB](https://www.mathworks.com/?s_tid=gn_logo) is available by loading a `matlab` module, which will add a launcher to start the software in a VNC session. Note that you will need to provide your own license file, as explained in our [MATLAB](https://docs.alliancecan.ca/MATLAB "wikilink") page. ![MATLAB launcher](https://docs.alliancecan.ca/MATLAB_launcher.png "MATLAB launcher")

#### NVidia Nsight Systems {#nvidia_nsight_systems}

[NVidia Nsight Systems](https://developer.nvidia.com/nsight-systems) is a performance analysis tool developed primarily for profiling GPUs, but which can profile CPU code as well. ![NVidia Nsight Systems launcher](https://docs.alliancecan.ca/NVidia_Nsight_Systems_launcher.png "NVidia Nsight Systems launcher") Loading a `cuda` or a `nvhpc` module will madd a launcher to start the graphical user interface in a VNC session.

#### Octave

[GNU Octave](https://octave.org/) is an open-source scientific programming language largely compatible with MATLAB. Loading an `octave` module will add a launcher to start the graphical user interface for Octave through a VNC session. See our page on [Octave](https://docs.alliancecan.ca/Octave "wikilink") for more details on using this software package. ![Octave launcher](https://docs.alliancecan.ca/Octave_launcher.png "Octave launcher")

#### ParaView

[ParaView](https://www.paraview.org/) is a powerful open-source visualisation software. Loading a `paraview` module will add a launcher to start the Paraview graphical user interface through a VNC session. See our page on [ParaView](https://docs.alliancecan.ca/ParaView "wikilink") for more details on using this software package. ![ParaView launcher](https://docs.alliancecan.ca/ParaView_launcher.png "ParaView launcher")

#### QGIS

[QGIS](https://qgis.org/) is a powerful open-source software for visualizing and processing geographic information systems (GIS) data. Loading a `qgis` module will add a launcher to start the QGIS graphical user interface through a VNC session. See our page on [QGIS](https://docs.alliancecan.ca/QGIS "wikilink") for more details on this software package. ![QGIS launcher](https://docs.alliancecan.ca/QGIS_launcher.png "QGIS launcher")

#### StarCCM+

Siemens\'s [Star-CCM+](https://plm.sw.siemens.com/en-US/simcenter/fluids-thermal-simulation/star-ccm/) is a commercial computational fluid dynamic simulation software. It is available by loading one of the `starccm` or the `starccm-mixed` modules, which will add a launcher to start the StarCCM+ graphical user interface through a VNC session. As for all commercial packages, you will need to provide your own license. See our page on [Star-CCM+](https://docs.alliancecan.ca/Star-CCM+ "wikilink") for more details on using this software. ![StarCCM+ launcher](https://docs.alliancecan.ca/StarCCM+_launcher.png "StarCCM+ launcher")

## Additional information on running Python notebooks {#additional_information_on_running_python_notebooks}

#### Python notebook {#python_notebook}

![Searching for scipy-stack modules](https://docs.alliancecan.ca/JupyterLab_Softwares_ScipyStack.png "Searching for scipy-stack modules") If any of the following scientific Python packages is required by your notebook, before you open this notebook, you must load the `scipy-stack` module from the JupyterLab `<i>`{=html}Softwares`</i>`{=html} tool:

-   `ipython`, `ipython_genutils`, `ipykernel`, `ipyparallel`
-   `matplotlib`
-   `numpy`
-   `pandas`
-   `scipy`
-   See [SciPy stack](https://docs.alliancecan.ca/Python#SciPy_stack "wikilink") for more on this

Note: You may also install needed packages by running for example the following command inside a cell: `pip install --no-index numpy`.

-   For some packages (like `plotly`, for example), you may need to restart the notebook\'s kernel before importing the package.
-   The installation of packages in the default Python kernel environment is temporary to the lifetime of the JupyterLab session; you will have to reinstall these packages the next time you start a new JupyterLab session. For a persistent Python environment, you must configure a `<b>`{=html}[custom Python kernel](https://docs.alliancecan.ca/Advanced_Jupyter_configuration#Python_kernel "wikilink")`</b>`{=html}.

To open an existing Python notebook:

-   Go back to the `<i>`{=html}File Browser`</i>`{=html}.
-   Browse to the location of the `*.ipynb` file.
-   Double-click on the `*.ipynb` file.
    -   This will open the Python notebook in a new JupyterLab tab.
    -   An IPython kernel will start running in the background for this notebook.

To open a new Python notebook in the current `<i>`{=html}File Browser`</i>`{=html} directory:

-   Click on the `<i>`{=html}Python 3.x`</i>`{=html} launcher under the `<i>`{=html}Notebook`</i>`{=html} section.
    -   This will open a new Python 3 notebook in a new JupyterLab tab.
    -   A new IPython kernel will start running in the background for this notebook.

### Running notebooks as Python scripts {#running_notebooks_as_python_scripts}

1\. From the console, or in a new notebook cell, install `nbconvert` :

``` bash
!pip install --no-index nbconvert
```

2\. Convert your notebooks to Python scripts

``` bash
!jupyter nbconvert --to python my-current-notebook.ipynb
```

3\. Create your [non-interactive submission script](https://docs.alliancecan.ca/Running_jobs#Use_sbatch_to_submit_jobs "wikilink"), and submit it.

In your submission script, run your converted notebook with:

``` bash
python my-current-notebook.py
```

And submit your non-interactive job:
