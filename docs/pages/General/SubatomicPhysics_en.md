---
title: "SubatomicPhysics/en"
url: "https://docs.alliancecan.ca/wiki/SubatomicPhysics/en"
category: "General"
last_modified: "2021-07-27T21:02:01Z"
page_id: 13580
display_title: "SubatomicPhysics"
---

`<languages />`{=html}

## Subatomic and High Energy Physics Software {#subatomic_and_high_energy_physics_software}

This page is maintained by the Compute Canada Subatomic Physics National Team and was last updated July 2021.

There is a reference page for the [Astronomy and High Energy Physics Interactive Analysis Facility](https://docs.alliancecan.ca/Astronomy_and_High_Energy_Physics_Interactive_Analysis_Facility "Astronomy and High Energy Physics Interactive Analysis Facility"){.wikilink}.

Many of the subatomic experimental physics groups are relying on CVMFS repositories from CERN or the Open Science Grid and specific repositories for each experiment.

The CCenv that is setup for regular users can create conflicts with some of the setups from these repositories as standard libraries are provided from the Compute Canada soft.computecanada.ca CVMFS repository which uses Nix and Easybuild to provide access rather than having the software installed on the base OS of the compute nodes.

For ATLAS users, there are pages housed on the TRIUMF twiki that should be helpful. **NOTE: ATLAS users should use the recommended setups for Tier-3 use rather than reinventing techniques that are described below.**

- <https://twiki.atlas-canada.ca/bin/view/AtlasCanada/ComputeCanadaTier3s>
- <https://twiki.atlas-canada.ca/bin/view/AtlasCanada/Containers>

Many setups assume that the base nodes have the [HEP_OSLibs](https://gitlab.cern.ch/linuxsupport/rpms/HEP_OSlibs/blob/master/README.md) packages/rpms setup\[1\], which is **not** true on the CC computing nodes. One might be able to get away with some simple setups from the \'sft.cern.ch\' repository, but the suggested approach is to use singularity containers which have the necessary rpm\'s installed, which is described in the next section below. This also allows use of different OS bases (e.g. SL6) on the CentOS7-based Compute Canada infrastructure.

To setup a CentOS7 based view from sft.cern.ch (e.g. with gcc8)

`source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_95 x86_64-centos7-gcc8-opt`

this will include the necessary paths to compilers, geant4, ROOT, etc.

Available setups for `<arch-os-complier>`{=html} for LCG_95 are:

` x86_64-centos7-gcc7-dbg`\
` x86_64-centos7-gcc7-opt`\
` x86_64-centos7-gcc8-dbg`\
` x86_64-centos7-gcc8-opt`\
` x86_64-slc6-gcc62-opt`\
` x86_64-slc6-gcc7-dbg`\
` x86_64-slc6-gcc7-opt`\
` x86_64-slc6-gcc8-dbg`\
` x86_64-slc6-gcc8-opt`\
` x86_64-ubuntu1804-gcc7-opt`\
` x86_64-ubuntu1804-gcc8-dbg`\
` x86_64-ubuntu1804-gcc8-opt`

\[1\] A list of all the rpm\'s installed via HEPOS_Libs for CentOS7 is available at <https://gitlab.cern.ch/linuxsupport/rpms/HEP_OSlibs/blob/7.2.11-3.el7/dependencies/HEP_OSlibs.x86_64.dependencies-recursive-flat.txt>

### Running in containers {#running_in_containers}

As of this writing there are two main repositories for container images that we are aware of for HEP-related software, both distributed via CVMFS repositories. One from ATLAS and the other from WLCG.

- ATLAS - The ATLAS distributions of Singularity images are documented well at <https://twiki.cern.ch/twiki/bin/view/AtlasComputing/ADCContainersDeployment>

`  - single file packed images:  /cvmfs/atlas.cern.ch/repo/containers/images/singularity/`\
`   - unpacked images: /cvmfs/atlas.cern.ch/repo/containers/fs/singularity/`\

- WLCG unpacked repository - This is a development project that uses [DUCC](https://cvmfs.readthedocs.io/en/stable/cpt-ducc.html) to automate publishing of container images from a Docker image registry to CVMFS. The images are published to CVMFS in a standard directory structure format that is used by Singularity, as well as the layered format used by Docker, allowing Docker to instantiate images directly from CVMFS using the [graph driver](https://cvmfs.readthedocs.io/en/stable/cpt-graphdriver.html) plugin. There is additional documentation for this project at <https://github.com/cvmfs/ducc>. The list of images that are automatically published is [here](https://gitlab.cern.ch/unpacked/sync/blob/master/recipe.yaml) and includes the atlas-grid-centos7 image. To add one to the list you can submit a merge request.

`  - Images are under /cvmfs/unpacked.cern.ch/`

## Invoking Singularity Image {#invoking_singularity_image}

The singularity executable is provided on Compute Canada sites in slight variants as this is run in a setuid environment by default so is installed outside of the usual Compute Canada CVMFS stack. There are various versions available on each site and defaults may be changed, so best to invoke the necessary version (as of this writing 2.6.1, 3.2.0 and 3.3.0 are the most likely candidates).

`  cedar - /opt/software/singularity-x.x.x`\
`  graham - /opt/software/singularity-x.x.x`\
`  niagara - module load singularity; /opt/singularity/2`

Invoking a container from a CVMFS repository can be either done directly as the image will be cached or the image can be pulled and stored in your local area which may give some performance benefits, depending on the system.
