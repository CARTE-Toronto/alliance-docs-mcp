---
title: "BEAST/en"
url: "https://docs.alliancecan.ca/wiki/BEAST/en"
category: "General"
last_modified: "2022-12-12T15:34:23Z"
page_id: 8812
display_title: "BEAST"
---

== Description ==

BEASTBEAST2 Homepage: http://beast2.org/ is a cross-platform program for Bayesian MCMC analysis of molecular  sequences. It is entirely orientated towards rooted, time-measured phylogenies inferred using  strict or relaxed molecular clock models. It can be used as a method of reconstructing phylogenies  but is also a framework for testing evolutionary hypotheses without conditioning on a single  tree topology. BEAST uses MCMC to average over tree space, so that each tree is weighted  proportional to its posterior probability.

BEAST can use the beagle-libBeagle-lib Homepage: https://github.com/beagle-dev/beagle-lib, which is a high-performance library that can perform the core calculations at the heart of most Bayesian and Maximum Likelihood phylogenetics packages.

== Usage ==

Loading the BEAST module with: module load beast, will automatically load it's dependencies, namely the beagle-lib and java modules, and set the environment variable EBROOTBEAST to point to the directory where BEAST's program files are located.

=== Managing BEAST Packages/Add-ons ===

BEAST has been installed without any packages (add-ons). You can use the packagemanager command (for BEAST v2.5.1 and newer; in older versions of BEAST, the command is addonmanager) to install the desired packages within your home directory.

  $ module load beast/2.5.1
  $ packagemanager -list
  Name    | Installation Status | Latest Version | Dependencies | Description
  --------------------------------------------------------------------------
  BEAST   | 2.5.1               | 2.5.0          |              | BEAST core
  --------------------------------------------------------------------------
  bacter  | NA                  | 2.2.0          |              | Bacterial ARG inference.
  BADTRIP | NA                  | 1.0.0          |              | Infer transmission time for [...]
  [...]
  SNAPP   | NA                  | 1.4.1          |              | SNP and AFLP Phylogenies
  [...]

  $ packagemanager -add SNAPP
  Package SNAPP is installed in ~/.beast/2.5/SNAPP.

  $ packagemanager -list
  Name    | Installation Status | Latest Version | Dependencies | Description
  --------------------------------------------------------------------------
  BEAST   | 2.5.1               | 2.5.0          |              | BEAST core
  --------------------------------------------------------------------------
  [...]
  SNAPP   | 1.4.1               | 1.4.1          |              | SNP and AFLP Phylogenies
  [...]

  $ module load beast/2.4.0
  $ addonmanager -list
  Name    | Installation Status | Latest Version | Dependencies | Description
  ---------------------------------------------------------------------------
  BEAST   | 2.4.0               | 2.4.8          |              | BEAST core
  ---------------------------------------------------------------------------
  bacter  | not installed       | 1.2.3          |              | Bacterial ARG inference.
  BASTA   | not installed       | 2.3.2          |              | Bayesian structured coalescent approximation
  [...]
  SNAPP   | not installed       | 1.3.0          |              | SNP and AFLP Phylogenies
  [...]

  $ addonmanager -add SNAPP
  Package SNAPP is installed in ~/.beast/2.4/SNAPP.

  $ addonmanager -list
  Name    | Installation Status | Latest Version | Dependencies | Description
  ---------------------------------------------------------------------------
  BEAST   | 2.4.0               | 2.4.8          |              | BEAST core
  ---------------------------------------------------------------------------
  [...]
  SNAPP   | 1.3.0               | 1.3.0          |              | SNP and AFLP Phylogenies
  [...]

For more information on how to manage BEAST packages please read
the section "Server machines" at: http://www.beast2.org/managing-packages/

=== Simple Jobscript for BEAST ===

=== Jobscript for BEAST with more Memory ===

== References ==