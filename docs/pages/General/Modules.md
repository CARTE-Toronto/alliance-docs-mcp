---
title: "Modules/en"
url: "https://docs.alliancecan.ca/wiki/Modules/en"
category: "General"
last_modified: "2022-06-22T01:35:02Z"
page_id: 17114
display_title: "Modules"
---

In computing, a module is a unit of software that is designed to be independent, interchangeable, and contains everything necessary to provide the desired functionality.
Wikipedia, "Modular programming"
The term "module" may sometimes have a more specific meaning depending on the context.
This page describes a few types of modules and suggests links to further documentation content.

== Disambiguation ==

=== Lmod modules ===

Also called "environment modules", Lmod modules are used to alter your (shell) environment so as to enable you to use a particular software package,
or to use a non-default version of certain common software packages such as compilers.  See Using modules.

=== Python modules ===

In Python, a module is a file of code (usually Python code) which can be loaded with the import ... or from ... import ... statements to provide functionality.  A Python package is a collection of Python modules; the terms "package" and "module" are frequently interchanged in casual use.
Tutorialspoint.com, "What is the difference between a python module and a python package?"

Certain frequently used Python modules such as Numpy can be imported if you first load the scipy-stack Lmod module at the shell level.
See SciPy stack for details.

We maintain a large collection of Python "wheels."
These are modules which are pre-compiled to be compatible with the Standard software environments.
Before importing modules from our wheels, you should create a virtual environment.

Python modules which are not in the scipy-stack Lmod module or in our wheels collection can be installed from the internet
as described in the Installing packages section.

== Other related topics ==

The main Available software page is a good starting point. Other related pages are:
* Standard software environments: as of April 1, 2021, StdEnv/2020 is the default collection of Lmod modules
* Lmod modules specific to Niagara
* Tables of Lmod modules optimized for AVX, AVX2 and AVX512 CPU instructions
* Category Software: a list of different software pages in this wiki, including commercial or licensed software

== Footnotes ==