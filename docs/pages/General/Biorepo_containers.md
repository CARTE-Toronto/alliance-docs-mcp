---
title: "Biorepo containers"
url: "https://docs.alliancecan.ca/wiki/Biorepo_containers"
category: "General"
last_modified: "2023-06-07T12:14:08Z"
page_id: 22804
display_title: "Biorepo containers"
---

=Introduction=

Containers are now a common method of distributing software, especially software with complex dependencies.  Container management systems such as Docker are popular.  We support containerization on Alliance systems using Apptainer (formerly Singularity).  You can simply build a container and store it in your own project space, and share it with your research group that way.  But if you have collaborators that need to share a container, if you wish to run the same container on many systems, or if you need to preserve a complex workflow for some period of time, you may choose to deposit container images into the  repository we describe below.  If you wish to use containers from this repository, this article will outline available containers and some information on their use.  If you wish to contribute containers to this repository, this article outlines the requirements that must be met.

=Container repository=

The Research Software National Team (RSNT) and the Bioinformatics National Team (BNT) operate a repository of Apptainer images in CVMFS to make these containers available on all Alliance systems.  The BNT maintains the portion of that repository of interest to bioinformatics researchers.  This reduces the effort to run a containerized workflow on our infrastructure, allows for the hard work of creating workflows to be shared between members of the research community, and enables better research reproducibility.

Some examples of use cases are:
# A container with a mature workflow for a clinical study, that must remain stable for a long time (e.g. 3-5 years). Reproducibility is of great importance in this case; using a container all but guarantees this.
# A workflow that requires Conda and can not easily be ported to use pip and our modules.  A container can be used when the desired tools must be or are much more easily installed with Conda or a similar package manager.
# A container with most of the tools, packages and libraries needed for some widely-used workflow, such as single-cell RNA sequencing (scRNA-seq). This type of container has the commonly used tools as well as the prerequisite libraries and system files which make it easier to update the version of the desired tools or install new ones.

= Using the repository =

Still to come...

= Contributing a container =

This service makes container images available to all users on Alliance systems.  If a container or the workflow it embodies should be kept private, do not use this service.  Additionally, all applications in a container must be open source and governed by a license that allows us to distribute the software in this manner.

Only the sandbox format of container images (i.e. an unpacked directory) should be published on CVMFS, not .sif files.

If you have a container image you would like to incorporate into this system, please provide us with the following information:
# A brief description of the container for the list of below.
# An Apptainer or Singularity recipe file. Ensure that the recipe is reproducible: Pin the version of software being installed as much as possible.
# Instructions and/or examples of usage if your container implements a complex workflow rather than simply making a piece of software available.
# The audience for this image: Containers that we are intended to have a reach beyond their originating contributor. What is the number of research groups/PIs involved and the number of users who would be using the image?
# The justification for this image: Our primary way of deploying software is through modules and python wheels. Why can this application not be provided as a software module or python wheel?

Contact Technical support for more details or any follow up questions.

Note that there may be additional metadata and deployment processes needed to accommodate security requirements as set out by the Alliance security team.  These requirements and any other security remediations will be enumerated in the governing policy document for the container repository which will be linked here once published.

= Available containers =

 List of available containers:

Image Name        	Description                                                                        	Contributed By
timsconvert_v1.0.0	Convert raw Bruker timsTOF Pro and fleX MS data formats to open source data formats	Jean-Francois Lucier
ncov-tools        	QC pipeline on coronavirus sequencing results                                      	Jose Hector Galvez
genpipes          	Bioinformatics analysis pipeline                                                   	Jose Hector Galvez