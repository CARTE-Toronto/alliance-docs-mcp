---
title: "OpenCV/en"
url: "https://docs.alliancecan.ca/wiki/OpenCV/en"
category: "General"
last_modified: "2024-04-09T14:59:02Z"
page_id: 20503
display_title: "OpenCV"
---

OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision.

== CUDA ==
OpenCV is also available with CUDA.

where X.Y.Z represent the desired version.

== Extra modules ==
The module also contains the extra modules (contrib).

== Python bindings ==
The module contains bindings for multiple Python versions.
To discover which are the compatible Python versions, run

Or search directly opencv_python, by running

where X.Y.Z represent the desired version.

=== Usage ===
1. Load the required modules.

where X.Y.Z represent the desired version.

2. Import OpenCV.

If the command displays nothing, the import was successful.

==== Available Python packages  ====
Other Python packages depend on OpenCV bindings in order to be installed.
OpenCV provides four different packages:
* opencv_python
* opencv_contrib_python
* opencv_python_headless
* opencv_contrib_python_headless

 grep opencv
|result=
opencv-contrib-python              4.5.5
opencv-contrib-python-headless     4.5.5
opencv-python                      4.5.5
opencv-python-headless             4.5.5
}}

With the opencv module loaded, your package dependency for one of the OpenCV named will be satisfied.

== Use with OpenEXR ==
In order to read EXR files with OpenCV, the module must be activated through an environment variable.
1 python }}