---
title: "Visualization/en"
url: "https://docs.alliancecan.ca/wiki/Visualization/en"
category: "General"
last_modified: "2025-10-21T16:16:29Z"
page_id: 719
display_title: "Visualization"
---

== Popular visualization packages ==

=== ParaView ===
ParaView is a general-purpose 3D scientific visualization tool. It is open-source and compiles on all popular platforms (Linux, Windows, Mac), understands a large number of input file formats, provides multiple rendering modes, supports Python scripting, and can scale up to tens of thousands of processors for rendering of very large datasets.

* Using ParaView on Alliance systems
* ParaView official documentation
* ParaView gallery
* ParaView wiki
* ParaView Python scripting

=== VisIt ===
Similar to ParaView, VisIt is an open-source, general-purpose 3D scientific data analysis and visualization tool that scales from interactive analysis on laptops to very large HPC projects on tens of thousands of processors.

* Using VisIt on Alliance systems
* VisIt website
* VisIt gallery
* VisIt user community wiki
* VisIt tutorials along with sample datasets

=== VMD ===
VMD is an open-source molecular visualization program for displaying, animating, and analyzing large biomolecular systems in 3D. It supports scripting in Tcl and Python and runs on a variety of platforms (MacOS X, Linux, Windows). It reads many molecular data formats using an extensible plugin system and supports a number of different molecular representations.

* Using VMD on Alliance systems
* VMD User's Guide

=== VTK ===
The Visualization Toolkit (VTK) is an open-source package for 3D computer graphics, image processing, and visualization. The toolkit includes a C++ class library as well as several interfaces for interpreted languages such as Tcl/Tk, Java, and Python. VTK was the basis for many excellent visualization packages including ParaView and VisIt.

* Using VTK on Alliance systems
* VTK tutorials

=== YT ===
YT is a Python library for analyzing and visualizing volumetric, multi-resolution data. Initially developed for astrophysical simulation data, it can handle any uniform and multiple-resolution data on Cartesian, curvilinear, unstructured meshes and on particles.

* Using YT on Alliance systems

== Visualization on Alliance systems ==

There are many options for remote visualization on our systems. In general, whenever possible, for interactive rendering we recommend client-server visualization on interactive or high-priority nodes, and for non-interactive visualization we recommend off-screen batch jobs on regular compute nodes.

Other, less efficient options are X11-forwarding and VNC. For some packages these are the only available remote GUI options.

=== Client-server interactive visualization ===

In the client-server mode, supported by both ParaView and VisIt, all data will be processed remotely on the cluster, using either CPU or GPU rendering, while you interact with your visualization through a familiar GUI client on your local computer. You can find the details of setting up client-server visualization in ParaView and VisIt pages.

=== Remote windows with X11-forwarding ===

In general, X11-forwarding should be avoided for any heavy graphics, as it requires many round trips and is much slower than VNC (below). However, in some cases you can connect via ssh with X11. Below we show how you would do this on our clusters. We assume you have an X-server installed on your local computer.

Connect to the cluster with the -X/-Y flag for X11-forwarding. You can start your graphical application on the login node (small visualizations)

   module load vmd
   vmd

or you can request interactive resources on a compute node (large visualizations)

  salloc --time=1:00:0 --ntasks=1 --mem=3500 --account=def-someprof --x11

: and, once the job is running, start your graphical application inside the job

  module load vmd
  vmd

Since runtime is limited on the login nodes, you might want to request a testing job in order to have more time for exploring and visualizing your data. On the plus side, you will have access to 40 cores on each of the nodes requested. For performing an interactive visualization session in this way please follow these steps:

 ssh into trillium.alliancecan.ca with the -X/-Y flag for X11-forwarding
 Request an interactive job, ie.
   debugjob
This will connect you to a node, let's say for the argument "niaXYZW".
 Run your visualization program, eg. VMD

   module load vmd
   vmd

 Exit the debug session.

=== Remote off-screen windows via Xvfb ===

Some applications insist on displaying graphical output, but you don't actually need to see them since the results are saved in a file.
To work with offscreen rendering, the job can run as a regular batch job, using either the CPU or the GPU for 3D rendering. To enable this you can run
the application you are calling with the X virtual frame buffer (Xvfb) in a job script as follows:

  xvfb-run

if using the CPU for rendering or

  xvfb-run vglrun -d egl

if using the GPU for rendering, in which case you need to reserve one GPU with Slurm, see Using GPUs with Slurm.
Note that, depending on the workload the GPU may not necessarily be faster than the CPU, so it's important to benchmark before
committing to using the more expensive GPU.

=== Start a remote desktop via VNC ===

Frequently, it may be useful to start up graphical user interfaces for various software packages like Matlab. Doing so over X11-forwarding can result in a very slow connection to the server. Instead, we recommend using VNC to start and connect to a remote desktop. For more information, please see the article on VNC.

= Visualization training =

Please let us know if you would like to see a visualization workshop at your institution.

=== Full- or half-day workshops ===
* VisIt workshop slides from HPCS'2016 in Edmonton by Marcelo Ponce and Alex Razoumov
* ParaView workshop slides from July 2017 by Alex Razoumov
* Gnuplot, xmgrace, remote visualization tools (X-forwarding and VNC), python's matplotlib slides by Marcelo Ponce (SciNet/UofT) from Ontario HPC Summer School 2016
*  Brief overview of ParaView & VisIt slides by Marcelo Ponce (SciNet/UofT) from Ontario HPC Summer School 2016

=== Webinars and other short presentations ===

Western Canada visualization training materials page has video recordings and slides from many visualization webinars including:

* YT series: “Using YT for analysis and visualization of volumetric data” (Part 1) and "Working with data objects in YT” (Part 2)
* “Scientific visualization with Plotly”
* “Novel Visualization Techniques from the 2017 Visualize This Challenge”
* “Data Visualization on Compute Canada’s Supercomputers” contains recipes and demos of running client-server ParaView and batch ParaView scripts on both CPU and GPU partitions of our HPC clusters
* “Using ParaViewWeb for 3D Visualization and Data Analysis in a Web Browser”
* “Scripting and other advanced topics in VisIt visualization”
* “CPU-based rendering with OSPRay”
* “3D graphs with NetworkX, VTK, and ParaView”
* “Graph visualization with Gephi”

Other visualization presentations:

* Remote Graphics on SciNet's GPC system (Client-Server and VNC) slides by Ramses van Zon (SciNet/UofT) from October 2015 SciNet User Group Meeting
* VisIt Basics, slides by Marcelo Ponce (SciNet/UofT) from February 2016 SciNet User Group Meeting
* Intro to Complex Networks Visualization, with Python, slides by Marcelo Ponce (SciNet/UofT)
* Introduction to GUI Programming with Tkinter, from Sept.2014 by Erik Spence (SciNet/UofT)

== Tips and tricks ==

This section will describe visualization workflows not included into the workshop/webinar slides above. It is meant to be user-editable, so please feel free to add your cool visualization scripts and workflows here so that everyone can benefit from them.

== Regional and other visualization pages ==

* National Visualization Team's page with many sci-vis examples
* SFU's visualization webinars archive

=== SciNet HPC at the University of Toronto ===
* Visualization in Niagara
* visualization software
* VNC
* visualization nodes
* further resources and viz-tech talks
* using ParaView

== How to get visualization help ==
Please contact Technical support.