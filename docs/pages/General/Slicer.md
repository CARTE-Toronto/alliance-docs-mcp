---
title: "Slicer/en"
url: "https://docs.alliancecan.ca/wiki/Slicer/en"
category: "General"
last_modified: "2019-03-05T20:48:02Z"
page_id: 10153
display_title: "Slicer"
---

3D Slicer is an open source software platform for medical image informatics, image processing, and three-dimensional visualization. Built over two decades through support from the National Institutes of Health and a worldwide developer community, Slicer brings free, powerful cross-platform processing tools to physicians, researchers, and the general public.

IMPORTANT: Never make intense use of Slicer on the login nodes! Submit jobs using the command line whenever possible and if you must visualize your data using the GUI, please do so on an interactive node. Using parallel rendering on the shared login nodes will result in the termination of your session.

== Using the GUI ==
Using the Slicer GUI requires X11 forwarding, which you should make sure is enabled.

You can find information on how to connect with MobaXTerm and use X11 fowarding at Connecting with MobaXTerm.

==== Use an interactive node ====

Runtime is limited on the login nodes, so you should request an interactive job to have more time for exploring and visualizing your data. Additionally, by doing so, you will be able to use more cores for faster processing.

Request an interactive job, ie.
  [name@server ~]$ salloc --time=1:0:0 --ntasks=2 --x11 --account=def-someuser

This will connect you to a compute node. Note the --x11 argument.

You can then load Slicer and run it on the interactive node.

Once you are done and have closed Slicer, don't forget to terminate the interactive job:

== Using the command line ==

If you have to perform the same task on a large number of images,
or if you don't need to see the processed images as they are created,
you should use the command line interface and batch processing.
For examples see
* https://discourse.slicer.org/t/rtstruct-to-label-map/2437
* https://discourse.slicer.org/t/slicer-batch-processing-question-no-main-window-python-script/1863