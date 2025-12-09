---
title: "CESM"
url: "https://docs.alliancecan.ca/wiki/CESM"
category: "General"
last_modified: "2025-12-04T20:02:09Z"
page_id: 24186
display_title: "CESM"
---

"The Community Earth System Model is a fully coupled global climate model developed in collaboration with colleagues in the research community. CESM provides state of the art computer simulations of Earth's past, present, and future climate states."

=Porting and Validating=

The below configuration files and commands are designed for a local installation of CESM 2.1. Local installations allow for source code changes which may be useful for specific research purposes. Before making the adaptations as described in the sections below, please download CESM 2.1 from the CESM developers in your local directory.

This version is based on the latest official release.

This older version has been very popular in the research community, but it may become obsolete because this version is no longer officially supported.

To make this version work, some external dependencies must be replaced with newer versions which are no longer exactly matching the 2.1.3 version of CESM; researchers are responsible for confirming validity.

==Checkout externals==

Before your first use of CESM, you may checkout the individual model components by running the checkout_externals script.

You may need to accept a certificate from the CESM repository to download input files. To validate, run the same script with -S.

See this documentation page for an example of a valid output.

==Local machine file==

Create and edit the file ~/.cime/config_machines.xml from the following minimal content per cluster; update both configuration lines having def-EDIT_THIS with the compute account you want to use on the cluster.

Note: despite the Intel software dependencies, the below configuration works on Narval's AMD processors.

Note: despite the Intel software dependencies, the below configuration works on Rorqual's AMD processors.

Validate your XML machine file with the following commands:

Check the official template for additional parameters:

==Local batch file==

Create and edit the file ~/.cime/config_batch.xml from the following minimal content:

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0

regular

beluga

}}

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0

regular

cedar

}}

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0

regular

graham

}}

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0

regular

narval

}}

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0
--constraint=[skylakecascade]

regular

niagara

}}

--nodes=
--ntasks-per-node=
--output=
--exclusive
--mem=0

regular

rorqual

}}

Validate your XML batch file with the following commands:

Check the documentation for additional configuration parameters and examples.

==Local compilers file==

Create and edit the file ~/.cime/config_compilers.xml from the following minimal content per cluster:

Validate your XML compiler file with the following commands:

==Creating a test case==

The following commands assume the default model cesm and the current machine:

=Reference=

* Main website
** CESM Quickstart Guide (CESM2.1)
** CESM Coupled Model XML Files