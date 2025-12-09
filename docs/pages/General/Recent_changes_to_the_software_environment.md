---
title: "Recent changes to the software environment"
url: "https://docs.alliancecan.ca/wiki/Recent_changes_to_the_software_environment"
category: "General"
last_modified: "2024-04-24T12:49:19Z"
page_id: 16844
display_title: "Recent changes to the software environment"
---

Installation of software packages within the Alliance software environment is always performed using scripts. We use multiple tools, including EasyBuild, Gentoo Overlays (starting in 2020), and NixOS (formerly). This software environment can be used on any computer in the world thanks to CVMFS.

We also track all changes and new installations made to the software environment through Git, and you can see a list of recent changes in the links below.

Changes to software installed as modules:
* Changes to the core modules and modules installed for the AVX2 CPU architecture
* Changes to the modules installed for the AVX512 CPU architecture
* Changes to the modules installed for the AVX CPU architecture
* Changes to the modules installed for the SSE3 CPU architecture

Other changes:
* Changes to the primary configuration files
* Changes to the EasyBuild configuration
* Changes to custom modules and scripts
* Changes to the core of the Gentoo layer, for the module gentoo/YYYY, used with StdEnv/2020 and StdEnv/2023
* Deprecated Changes to the core of the Nix layer, for the module nixpkgs/16.09, used in StdEnv/2016.4, StdEnv/2018.3