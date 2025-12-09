---
title: "README files/en"
url: "https://docs.alliancecan.ca/wiki/README_files/en"
category: "General"
last_modified: "2025-07-29T16:23:18Z"
page_id: 29080
display_title: "README files"
---

In your project spaces, your data should be documented such that you know the purpose of each file. A README file is usually the first reference point.

Using README files on clusters is part of active research data management. It will be useful for future publications and for team members wondering what are the files in some directory.

= What to write in a README file =

* Source of the files
** Website or external database
** Authors
** Year
* Types of files present in the directory
** Structure of directories
* Which files are temporary
* Which files are actively used
* Which files could be archived
* Who should be able to access what and when:
** On the cluster;
** On a data repository (in some future).

= Formats of a README file =

* README or README.txt
** Free text format.
** Better than nothing, but no conventional style is enforced.
* README.md (Markdown), README.rst (reStructuredText)
** Structured text format that remains human readable.
** Can be compiled into formatted text (HTML or PDF).
* README.yaml (YAML), README.json (JSON), README.xml (XML)
** Slightly less human-readable.
** Machine-readable, which means a program can validate the contents of the README file.
** Could be used to generate a README file in another format.

= References =

* McMaster - README Generator
* UBC - Create a README file
* UWaterloo - README Files for Data Deposits