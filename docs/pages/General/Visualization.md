---
title: "Visualization/en"
url: "https://docs.alliancecan.ca/wiki/Visualization/en"
category: "General"
last_modified: "2025-10-21T16:16:29Z"
page_id: 719
display_title: "Visualization"
---

`<languages />`{=html}

## Popular visualization packages {#popular_visualization_packages}

### ParaView

[ParaView](http://www.paraview.org) is a general-purpose 3D scientific visualization tool. It is open-source and compiles on all popular platforms (Linux, Windows, Mac), understands a large number of input file formats, provides multiple rendering modes, supports Python scripting, and can scale up to tens of thousands of processors for rendering of very large datasets.

- [Using ParaView on Alliance systems](https://docs.alliancecan.ca/ParaView "Using ParaView on Alliance systems"){.wikilink}
- [ParaView official documentation](http://www.paraview.org/documentation)
- [ParaView gallery](http://www.paraview.org/gallery)
- [ParaView wiki](http://www.paraview.org/Wiki/ParaView)
- [ParaView Python scripting](http://www.paraview.org/Wiki/ParaView/Python_Scripting)

### VisIt

Similar to ParaView, [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/) is an open-source, general-purpose 3D scientific data analysis and visualization tool that scales from interactive analysis on laptops to very large HPC projects on tens of thousands of processors.

- [Using VisIt on Alliance systems](https://docs.alliancecan.ca/VisIt "Using VisIt on Alliance systems"){.wikilink}
- [VisIt website](https://visit-dav.github.io/visit-website)
- [VisIt gallery](https://visit-dav.github.io/visit-website/examples)
- [VisIt user community wiki](http://www.visitusers.org)
- [VisIt tutorials](http://www.visitusers.org/index.php?title=VisIt_Tutorial) along with [sample datasets](http://www.visitusers.org/index.php?title=Tutorial_Data)

### VMD

[VMD](http://www.ks.uiuc.edu/Research/vmd) is an open-source molecular visualization program for displaying, animating, and analyzing large biomolecular systems in 3D. It supports scripting in Tcl and Python and runs on a variety of platforms (MacOS X, Linux, Windows). It reads many molecular data formats using an extensible plugin system and supports a number of different molecular representations.

- [Using VMD on Alliance systems](https://docs.alliancecan.ca/VMD "Using VMD on Alliance systems"){.wikilink}
- [VMD User\'s Guide](http://www.ks.uiuc.edu/Research/vmd/current/ug)

### VTK

The Visualization Toolkit (VTK) is an open-source package for 3D computer graphics, image processing, and visualization. The toolkit includes a C++ class library as well as several interfaces for interpreted languages such as Tcl/Tk, Java, and Python. VTK was the basis for many excellent visualization packages including ParaView and VisIt.

- [Using VTK on Alliance systems](https://docs.alliancecan.ca/VTK "Using VTK on Alliance systems"){.wikilink}
- [VTK tutorials](https://itk.org/Wiki/VTK/Tutorials)

### YT

YT is a Python library for analyzing and visualizing volumetric, multi-resolution data. Initially developed for astrophysical simulation data, it can handle any uniform and multiple-resolution data on Cartesian, curvilinear, unstructured meshes and on particles.

- [Using YT on Alliance systems](https://docs.alliancecan.ca/yt "Using YT on Alliance systems"){.wikilink}

## Visualization on Alliance systems {#visualization_on_alliance_systems}

There are many options for remote visualization on our systems. In general, whenever possible, for interactive rendering we recommend **client-server visualization** on interactive or high-priority nodes, and for non-interactive visualization we recommend **off-screen batch jobs** on regular compute nodes.

Other, *less efficient* options are X11-forwarding and VNC. For some packages these are the only available remote GUI options.

### Client-server interactive visualization {#client_server_interactive_visualization}

In the client-server mode, supported by both ParaView and VisIt, all data will be processed remotely on the cluster, using either CPU or GPU rendering, while you interact with your visualization through a familiar GUI client on your local computer. You can find the details of setting up client-server visualization in [ParaView](https://docs.alliancecan.ca/ParaView "ParaView"){.wikilink} and [VisIt](https://docs.alliancecan.ca/VisIt "VisIt"){.wikilink} pages.

### Remote windows with X11-forwarding {#remote_windows_with_x11_forwarding}

In general, X11-forwarding should be avoided for any heavy graphics, as it requires many round trips and is much slower than VNC (below). However, in some cases you can connect via ssh with X11. Below we show how you would do this on our clusters. We assume you have an X-server installed on your local computer.

`<tabs>`{=html} `<tab name="Rorqual, Fir, Nibi, Narval">`{=html}

Connect to the cluster with the `-X/-Y` flag for X11-forwarding. You can start your graphical application on the login node (small visualizations)

`  module load vmd`\
`  vmd`

or you can request interactive resources on a compute node (large visualizations)

` salloc --time=1:00:0 --ntasks=1 --mem=3500 --account=def-someprof --x11`

:   and, once the job is running, start your graphical application inside the job

` module load vmd`\
` vmd`

`</tab>`{=html} `<tab name="Trillium">`{=html}

Since runtime is limited on the login nodes, you might want to request a testing job in order to have more time for exploring and visualizing your data. On the plus side, you will have access to 40 cores on each of the nodes requested. For performing an interactive visualization session in this way please follow these steps:

1.  ssh into trillium.alliancecan.ca with the `-X/-Y` flag for X11-forwarding
2.  Request an interactive job, ie.

debugjob This will connect you to a node, let\'s say for the argument \"niaXYZW\".

<li>

Run your visualization program, eg. VMD

</li>

`  module load vmd`\
`  vmd`

<li>

Exit the debug session.

</ol>

`</tab>`{=html} `</tabs>`{=html}

### Remote off-screen windows via Xvfb {#remote_off_screen_windows_via_xvfb}

Some applications insist on displaying graphical output, but you don\'t actually need to see them since the results are saved in a file. To work with offscreen rendering, the job can run as a regular batch job, using either the CPU or the GPU for 3D rendering. To enable this you can run the application you are calling with the X virtual frame buffer (Xvfb) in a job script as follows:

` xvfb-run ``<name-of-application>`{=html}

if using the CPU for rendering or

` xvfb-run vglrun -d egl ``<name-of-application>`{=html}

if using the GPU for rendering, in which case you need to reserve one GPU with Slurm, see [Using GPUs with Slurm](https://docs.alliancecan.ca/Using_GPUs_with_Slurm "Using GPUs with Slurm"){.wikilink}. Note that, depending on the workload the GPU may not necessarily be faster than the CPU, so it\'s important to benchmark before committing to using the more expensive GPU.

### Start a remote desktop via VNC {#start_a_remote_desktop_via_vnc}

Frequently, it may be useful to start up graphical user interfaces for various software packages like Matlab. Doing so over X11-forwarding can result in a very slow connection to the server. Instead, we recommend using VNC to start and connect to a remote desktop. For more information, please see [the article on VNC](https://docs.alliancecan.ca/VNC "the article on VNC"){.wikilink}.

# Visualization training {#visualization_training}

Please [let us know](https://docs.alliancecan.ca/mailto:support@tech.alliancecan.ca) if you would like to see a visualization workshop at your institution.

### Full- or half-day workshops {#full__or_half_day_workshops}

- [VisIt workshop slides](https://docs.alliancecan.ca/mediawiki/images/5/5d/Visit201606.pdf) from HPCS\'2016 in Edmonton by `<i>`{=html}Marcelo Ponce`</i>`{=html} and `<i>`{=html}Alex Razoumov`</i>`{=html}
- [ParaView workshop slides](https://docs.alliancecan.ca/mediawiki/images/6/6c/Paraview201707.pdf) from July 2017 by `<i>`{=html}Alex Razoumov`</i>`{=html}
- [Gnuplot, xmgrace, remote visualization tools (X-forwarding and VNC), python\'s matplotlib](https://support.scinet.utoronto.ca/~mponce/courses/ss2016/ss2016_visualization-I.pdf) slides by `<i>`{=html}Marcelo Ponce`</i>`{=html} (SciNet/UofT) from Ontario HPC Summer School 2016
- [Brief overview of ParaView & VisIt](https://support.scinet.utoronto.ca/~mponce/courses/ss2016/ss2016_visualization-II.pdf) slides by `<i>`{=html}Marcelo Ponce`</i>`{=html} (SciNet/UofT) from Ontario HPC Summer School 2016

### Webinars and other short presentations {#webinars_and_other_short_presentations}

[Western Canada visualization training materials page](https://training.westdri.ca/tools/visualization) has video recordings and slides from many visualization webinars including:

- YT series: "Using YT for analysis and visualization of volumetric data" (Part 1) and \"Working with data objects in YT" (Part 2)
- "Scientific visualization with Plotly"
- "Novel Visualization Techniques from the 2017 Visualize This Challenge"
- "Data Visualization on Compute Canada's Supercomputers" contains recipes and demos of running client-server ParaView and batch ParaView scripts on both CPU and GPU partitions of our HPC clusters
- "Using ParaViewWeb for 3D Visualization and Data Analysis in a Web Browser"
- "Scripting and other advanced topics in VisIt visualization"
- "CPU-based rendering with OSPRay"
- "3D graphs with NetworkX, VTK, and ParaView"
- "Graph visualization with Gephi"

Other visualization presentations:

- [Remote Graphics on SciNet\'s GPC system (Client-Server and VNC)](https://oldwiki.scinet.utoronto.ca/wiki/images/5/51/Remoteviz.pdf) slides by `<i>`{=html}Ramses van Zon`</i>`{=html} (SciNet/UofT) from October 2015 SciNet User Group Meeting
- [VisIt Basics](https://support.scinet.utoronto.ca/education/go.php/242/file_storage/index.php/download/1/files%5B%5D/6399/), slides by `<i>`{=html}Marcelo Ponce`</i>`{=html} (SciNet/UofT) from February 2016 SciNet User Group Meeting
- [Intro to Complex Networks Visualization, with Python](https://oldwiki.scinet.utoronto.ca/wiki/images/e/ea/8_ComplexNetworks.pdf), slides by `<i>`{=html}Marcelo Ponce`</i>`{=html} (SciNet/UofT)
- [Introduction to GUI Programming with Tkinter](https://oldwiki.scinet.utoronto.ca/wiki/images/9/9c/Tkinter.pdf), from Sept.2014 by `<i>`{=html}Erik Spence`</i>`{=html} (SciNet/UofT)

## Tips and tricks {#tips_and_tricks}

This section will describe visualization workflows not included into the workshop/webinar slides above. It is meant to be user-editable, so please feel free to add your cool visualization scripts and workflows here so that everyone can benefit from them.

## Regional and other visualization pages {#regional_and_other_visualization_pages}

- [`<b>`{=html}National Visualization Team\'s page`</b>`{=html}](https://ccvis.netlify.app) with many sci-vis examples
- [SFU\'s visualization webinars archive](https://training.westdri.ca/tools/visualization)

### [SciNet HPC at the University of Toronto](http://www.scinet.utoronto.ca) {#scinet_hpc_at_the_university_of_toronto}

- [Visualization in Niagara](https://docs.scinet.utoronto.ca/index.php/Visualization)
- [visualization software](https://oldwiki.scinet.utoronto.ca/wiki/index.php/Software_and_Libraries#anchor_viz)
- [VNC](https://oldwiki.scinet.utoronto.ca/wiki/index.php/VNC)
- [visualization nodes](https://oldwiki.scinet.utoronto.ca/wiki/index.php/Visualization_Nodes)
- [further resources and viz-tech talks](https://oldwiki.scinet.utoronto.ca/wiki/index.php/Knowledge_Base:_Tutorials_and_Manuals#Visualization)
- [using ParaView](https://oldwiki.scinet.utoronto.ca/wiki/index.php/Using_Paraview)

## How to get visualization help {#how_to_get_visualization_help}

Please contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.
