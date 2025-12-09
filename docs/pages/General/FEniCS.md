---
title: "FEniCS/en"
url: "https://docs.alliancecan.ca/wiki/FEniCS/en"
category: "General"
last_modified: "2025-12-09T19:11:02Z"
page_id: 3859
display_title: "FEniCS/en"
---

FEniCS is a popular open-source computing platform for solving partial differential equations (PDEs).

FEniCS can be built with various extensions, so we do not offer a single, global installation. Please choose between
* Installation in a virtual environment
* Using a Singularity container

= Installation in a virtual environment =
These are instructions for installing FEniCS version 2019.1.0, under StdEnv/2020 with OpenMPI and GCC 9.3.0.

You can run the script below by copying it to the cluster you are using and running bash fenics-install.sh.

Note that the installation will warn you that it will create (or replace) the application directory, and
will give usage instructions when the installation is successful. The script can be modified to change
the installation directory if needed.

==FEniCS add-ons==

This section has not been updated to work with StdEnv/2020.

First install FEniCS following instructions above.

===mshr===

Then run

= Using a Singularity container =

The following Singularity Recipe will download the FEniCS Docker image, install it, and download additional packages, e.g., various Python packages. This recipe must be run on your own machine, that is, a Linux machine with Singularity installed where you have root access.

To build your FEniCS image using this recipe, run the following command:

  sudo singularity build FEniCS.simg FEniCS-ComputeCanada-Singularity-Recipe

and then upload FEniCS.simg to your account. The FEniCS Docker image places a number of files in /home/fenics.

= FEniCS Legacy (2019) Installation on Trillium =

Go to your home directory and follow the instructions below to set up and test the container for the legacy FEniCS 2019 version.

== 1. Download the Docker image as an Apptainer SIF ==

apttainer pull fenics-legacy.sif docker://ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

== 2. Make a writable sandbox directory ==
Create a writable directory tree (fenics-legacy.sandbox) from the SIF file so you can modify or install extra packages:

apptainer build --sandbox fenics-legacy.sandbox fenics-legacy.sif

Note:
* fenics-legacy.sandbox is just a directory name the command will create.
* You can call it something else (e.g. fenics-dev/ or my_rw_image/).
* The .sandbox suffix is just a convention, not required.

== 3. Fix pip certificate bundle path ==
Inside the sandbox, create a certs folder and symlink the CA bundle so pip/SSL trusts HTTPS:

apptainer exec --writable fenics-legacy.sandbox sh -c "mkdir -p /etc/pki/tls/certs && ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt"

== 4. Create a new SIF from the sandbox ==
After modifications, create a new read-only image (portable, reproducible):

apptainer build fenics-legacy-updated.sif fenics-legacy.sandbox

== 5. Run quick tests ==

apptainer exec --bind $PWD:/root/shared --pwd /root/shared fenics-legacy-updated.sif python3 -c "import ufl_legacy; print('ufl_legacy ok. version:', ufl_legacy.__version__)"

Note:
* --bind $PWD:/root/shared mounts your current host directory in the container.
* --pwd sets the working directory there.

== Important Notes ==
* FEniCS Legacy (2019.1.x) requires UFL Legacy, already bundled.
* The Python package is named ufl_legacy, not ufl.
* Compatible UFL version is 2022.3.0 (provided by ufl_legacy).
* A plain import ufl should fail, while import ufl_legacy should succeed.

== Aliasing ufl_legacy as ufl ==
Some downstream packages (like Oasis) assume import ufl. To avoid patching them all, you can provide a shim package that re-exports ufl_legacy as ufl.

Create the file /pyshims/ufl/__init__.py with the following contents:

import sys
import ufl_legacy as ufl

api = [k for k in ufl.__dict__.keys() if not k.startswith('__') and not k.endswith('__')]
for key in api:
    sys.modules['ufl.{}'.format(key)] = getattr(ufl, key)
del api

== Test the aliasing ==
Prepend the shim path to PYTHONPATH when launching inside the container:

APPTAINERENV_PYTHONPATH=:$PYTHONPATH apptainer exec --bind /scratch:/scratch ~/fenics-legacy-updated.sif python3 -c "from ufl.tensors import ListTensor; print('UFL tensors ok')"