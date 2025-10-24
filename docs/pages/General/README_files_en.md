---
title: "README files/en"
url: "https://docs.alliancecan.ca/wiki/README_files/en"
category: "General"
last_modified: "2025-07-29T16:23:18Z"
page_id: 29080
display_title: "README files"
---

`<languages />`{=html} In your project spaces, your data should be documented such that you know the purpose of each file. A README file is usually the first reference point.

Using README files on clusters is part of active research data management. It will be useful for future publications and for team members wondering what are the files in some directory.

# What to write in a README file {#what_to_write_in_a_readme_file}

- Source of the files
  - Website or external database
  - Authors
  - Year
- Types of files present in the directory
  - Structure of directories
- Which files are temporary
- Which files are actively used
- Which files could be archived
- Who should be able to access what and when:
  - On the cluster;
  - On a data repository (in some future).

# Formats of a README file {#formats_of_a_readme_file}

- `README` or `README.txt`
  - Free text format.
  - Better than nothing, but no conventional style is enforced.
- `README.md` ([Markdown](https://www.markdownguide.org/)), `README.rst` ([reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html))
  - Structured text format that remains human readable.
  - Can be compiled into formatted text (HTML or PDF).
- `README.yaml` ([YAML](https://yaml.org)), `README.json` ([JSON](https://en.wikipedia.org/wiki/JSON)), `README.xml` ([XML](https://developer.mozilla.org/en-US/docs/Web/XML/Guides/XML_introduction))
  - Slightly less human-readable.
  - Machine-readable, which means a program can validate the contents of the README file.
  - Could be used to generate a README file in another format.

# References

- [McMaster - README Generator](https://rdm.mcmaster.ca/readme)
- [UBC - Create a README file](https://ubc-library-rc.github.io/rdm/content/03_create_readme.html)
- [UWaterloo - README Files for Data Deposits](https://subjectguides.uwaterloo.ca/rdm/basics#readme)
