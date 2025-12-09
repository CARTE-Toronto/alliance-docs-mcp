---
title: "OpenSim"
url: "https://docs.alliancecan.ca/wiki/OpenSim"
category: "General"
last_modified: "2025-02-11T20:56:31Z"
page_id: 27867
display_title: "OpenSim"
---

== Description ==

"OpenSim is a freely available, user extensible software system that lets users develop models of musculoskeletal structures and create dynamic simulations of movement. "
OpenSim includes Python and Matlab APIs. It is commonly used with OpenSim Moco.

The OpenSim module available through our software stack includes support for OpenSim Moco, as well as bindings to enable scripting in Python or Matlab.

== Using OpenSim via Matlab ==

=== Setup ===
Before first use of OpenSim on a cluster, you must configure the necessary Java paths, by running:

After exiting and relaunching Matlab, you can verify that OpenSim is imported by running in Matlab:  org.opensim.modeling.opensimCommon.GetVersion()

== Using OpenSim via Python ==

In order to use the OpenSim Python package, an OpenSim module must be loaded, and a numpy package must be available through a  virtual environment  or loading a  scipy-stack module. You should subsequently be able to import the opensim package in Python.