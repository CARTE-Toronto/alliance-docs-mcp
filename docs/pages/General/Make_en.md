---
title: "Make/en"
url: "https://docs.alliancecan.ca/wiki/Make/en"
category: "General"
last_modified: "2024-07-22T22:22:04Z"
page_id: 2001
display_title: "Make"
---

`<languages />`{=html}

## Description

[make](http://en.wikipedia.org/wiki/Makefile) is a utility that automatically builds files, such as executables or libraries, from other files, such as source code.

The `make` command interprets and executes the instructions within a file named `makefile`. Unlike a simple script, `make` only executes the commands that are necessary. The goal is to arrive at a result (compiled or installed software, formatted documentation, etc.) without needing to redo all steps.

The `makefile` contains information on *dependencies*. For example, if the `makefile` indicates that an object (`.o`) file depends on a source file, and the source file has changed, then the source file is recompiled to update the object file. In the same way, if an executable depends on any object files which have changed then the linking step will be rerun to update the executable. All dependencies must be included in the `makefile`. Then it is not necessary to recompile all files for every modification; the `make` command takes care of recompiling and relinking only what is necessary.

## Examples for using make {#examples_for_using_make}

The main argument of the `make` command is the *target*. The *target* may be the name of some file that `make` should build, or it may be an abstract target such as *all*, *test*, *check*, *clean*, or *install*. The targets that are available depend on the contents of the `makefile`, but the ones just listed are conventional and are specified in many makefiles. If `make` is invoked with no target specified, like so:

then the typical behaviour is to construct everything, equivalent to:

The *test* or *check* targets are generally used to run tests to validate if the application or compiled library functions correctly. Usually these targets depend on the *all* target. Hence you can verify the compilation using

or

The *clean* target erases all previously compiled binary files to be able to recompile from scratch. There is sometimes also a *distclean* target, which not only deletes files made by `make`, but also files created at configuration time by [configure](https://docs.alliancecan.ca/Autotools "configure"){.wikilink} or [cmake](https://docs.alliancecan.ca/CMake "cmake"){.wikilink}. So to clean the compilation directory, you can usually run

and sometimes

The *install* target normally installs a compiled program or library. Where the installation is put depends on the `makefile`, but can often be modified using an additional *prefix* parameter, like this: \$HOME/PROGRAM}}

The targets `all, test, check, clean, distclean` and `install` are only conventions and a `makefile` author could very well choose another convention. To get more information on typical target names, notably supported by all GNU applications, visit [this page](http://www.gnu.org/software/make/manual/make.html#Standard-Targets). Options to configure installation and other directories are [listed here](http://www.gnu.org/software/make/manual/make.html#Directory-Variables).

## Example of a `Makefile` {#example_of_a_makefile}

The following example, of general use, includes a lot of explanations and comments. For a detailed guide on how to create a `makefile`, visit [the GNU Make web site](http://www.gnu.org/software/make/manual/make.html#Introduction).
