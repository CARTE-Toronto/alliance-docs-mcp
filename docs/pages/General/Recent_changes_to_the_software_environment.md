---
title: "Recent changes to the software environment"
url: "https://docs.alliancecan.ca/wiki/Recent_changes_to_the_software_environment"
category: "General"
last_modified: "2024-04-24T12:49:19Z"
page_id: 16844
display_title: "Recent changes to the software environment"
---

Installation of software packages within the Alliance software environment is always performed using scripts. We use multiple tools, including [EasyBuild](https://easybuild.io/), [Gentoo Overlays](https://overlays.gentoo.org/) (starting in 2020), and [NixOS](https://github.com/NixOS/nixpkgs) (formerly). This software environment can be [used on any computer in the world](https://docs.alliancecan.ca/Accessing_CVMFS "used on any computer in the world"){.wikilink} thanks to [CVMFS](https://cvmfs.readthedocs.io/en/stable/).

We also track all changes and new installations made to the software environment through [Git](https://docs.alliancecan.ca/Git "Git"){.wikilink}, and you can see a list of recent changes in the links below.

Changes to software installed as modules:

- [Changes to the core modules and modules installed for the AVX2 CPU architecture](https://github.com/ComputeCanada/easybuild-easyconfigs-installed-avx2/commits/main)
- [Changes to the modules installed for the AVX512 CPU architecture](https://github.com/ComputeCanada/easybuild-easyconfigs-installed-avx512/commits/main)
- [Changes to the modules installed for the AVX CPU architecture](https://github.com/ComputeCanada/easybuild-easyconfigs-installed-avx/commits/main)
- [Changes to the modules installed for the SSE3 CPU architecture](https://github.com/ComputeCanada/easybuild-easyconfigs-installed-sse3/commits/main)

Other changes:

- [Changes to the primary configuration files](https://github.com/ComputeCanada/software-stack-config/commits/main)
- [Changes to the EasyBuild configuration](https://github.com/ComputeCanada/easybuild-computecanada-config/commits/main)
- [Changes to custom modules and scripts](https://github.com/ComputeCanada/software-stack-custom/commits/main)
- [Changes to the core of the Gentoo layer, for the module `gentoo/YYYY`, used with `StdEnv/2020` and `StdEnv/2023`](https://github.com/ComputeCanada/gentoo-overlay/commits/main)
- ***Deprecated*** [Changes to the core of the Nix layer, for the module `nixpkgs/16.09`, used in `StdEnv/2016.4`, `StdEnv/2018.3`](https://github.com/ComputeCanada/nixpkgs/commits/computecanada-16.09)
