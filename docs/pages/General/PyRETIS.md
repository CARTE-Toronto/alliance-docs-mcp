---
title: "PyRETIS/en"
url: "https://docs.alliancecan.ca/wiki/PyRETIS/en"
category: "General"
last_modified: "2020-07-30T17:24:01Z"
page_id: 14078
display_title: "PyRETIS"
---

PyRETIS is a Python library for rare event molecular simulations with emphasis on methods based on transition interface sampling (TIS) and replica exchange transition interface sampling (RETIS).

== Installing PyRETIS ==

We provide pre-compiled Python Wheels for PyRETIS in our Wheelhouse that are compatible with different versions of Python and can be installed within a virtual Python environment.

As of July 2020, PyRETIS 2.5.0 is compatible with Python versions 3.6 and 3.7. According to the PyRETIS installation instructions the dependency MDTraj has to be installed after PyRETIS.

A Python virtualenv with PyRETIS can be created by running the following series of commands (lines beginning with # are comments, $ is a prompt and (env_PyRETIS) $ is a prompt with an activated virtualenv):

 # load the Python module we want to use, e.g. python/3.7:
 $ module load python/3.7

 # create a virtualenv
 $ virtualenv --no-download ~/env_PyRETIS

 # activate the virtualenv
 $ source ~/env_PyRETIS/bin/activate

 # install PyRETIS and then mdtraj
 (env_PyRETIS) $ pip install --no-index pyretis
 (env_PyRETIS) $ pip install --no-index mdtraj

 # run PyRETIS
 (env_PyRETIS) $ pyretisrun --help

In order to use pyretisrun (e.g. in our job scripts) we only need to activate the module again:

 source ~/env_PyRETIS/bin/activate
 pyretisrun --input INPUT.rst  --log_file LOG_FILE.log

PyRETIS also offers an analysis tool, named PyVisA. Its GUI requires PyQt5 to be executed. PyQt5 is installed part of the Qt modules.
In order to allow the Python version from the virtualenv to find PyQt5, it is important to first load the modules for Python and Qt before activating the PyRETIS virtualenv:

 $ module load python/3.7 qt/5.11.3
 $ source ~/env_PyRETIS/bin/activate
 (env_PyRETIS) $ pyretisanalyse  -pyvisa  ...

== Using PyRETIS ==

The usage of PyRETIS is documented on the PyRETIS website at http://www.pyretis.org/ and in the group's papers:

# Lervik A, Riccardi E, van Erp TS. PyRETIS: A well-done, medium-sized python library for rare events. J Comput Chem. 2017;38: 2439–2451. doi:10.1002/jcc.24900
# Riccardi E, Lervik A, Roet S, Aarøen O, Erp TS. PyRETIS 2: An improbability drive for rare events. J Comput Chem. 2020;41: 370–377. doi:10.1002/jcc.26112