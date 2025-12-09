---
title: "Anaconda/en"
url: "https://docs.alliancecan.ca/wiki/Anaconda/en"
category: "General"
last_modified: "2025-12-02T14:59:54Z"
page_id: 4505
display_title: "Anaconda"
---

Anaconda is a Python distribution.

= Why is Anaconda not recommended on a cluster ? =

Anaconda may cause issues on a cluster for multiple reasons:

* Anaconda very often installs software (compilers, scientific libraries etc.) which already exist on our clusters as modules, with a configuration that is not optimal, and which may cause conflicts.
* It installs binaries which are not optimized for the processor architecture on our clusters. Your jobs may be slower because of it.
* It makes incorrect assumptions about the location of various system libraries. Your jobs may encounter errors when running.
* Anaconda uses the $HOME directory for its installation, where it writes an enormous number of files. A single Anaconda installation can easily absorb almost half of your quota for the number of files in your home directory.
* Anaconda is slower than the installation of packages via Python wheels.
* Anaconda modifies the $HOME/.bashrc file, which can easily cause conflicts.

= What are alternatives ? =
The first step you should take is to contact our Technical support, so that our experts investigate with your what is the best alternative for your needs. If you prefer to attempt it yourself, two main options are listed below.

== Transition from Conda to virtualenv ==

A virtual environment offers you all the functionality which you need to use Python on our clusters. This should be the first option that you explore. Here is how to convert to the use of virtual environments if you use Anaconda on your personal computer:

# List the dependencies (requirements) of the application you want to use. To do so, you can:
## Run pip show  from your virtual environment (if the package exists on PyPI)
## Or, check if there is a requirements.txt file in the Git repository.
## Or, check the variable install_requires of the file setup.py, which lists the requirements.
# Find which dependencies are Python modules and which are libraries provided by Anaconda. For example, CUDA and CuDNN are libraries which are available on Anaconda Cloud but which you should not install yourself on our clusters - they are already installed.
# Remove from the list of dependencies everything which is not a Python module (e.g. cudatoolkit and cudnn).
# Use a virtual environment in which you will install your dependencies.

Your software should run - if it doesn't, don't hesitate to contact us.

== Using Apptainer ==

In some situations, the complexity of the dependencies of a program requires the use of a solution where you can control the entire software environment. In these situations, we recommend the tool  Apptainer; note that a Docker image can be converted into an Apptainer image. The only disadvantage of Apptainer is its consumption of disk space. If your research group plans on using several images, it would be wise to collect all of them together in a single directory of the group's project space to avoid duplication.