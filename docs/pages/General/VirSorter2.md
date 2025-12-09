---
title: "VirSorter2/en"
url: "https://docs.alliancecan.ca/wiki/VirSorter2/en"
category: "General"
last_modified: "2024-07-12T18:02:24Z"
page_id: 24639
display_title: "VirSorter2"
---

`<languages />`{=html}

\_\_TOC\_\_

[VirSorter2](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-020-00990-y) is a tool to identify new viral sequences.

This page discusses how to install and use VirSorter2 v2.2.4.

Source code and documentation for VirSorter2 can be found on their [GitHub page](https://github.com/jiarong/VirSorter2).

Remember to [cite](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-020-00990-y#citeas) VirSorter2 if you use it for your analyses.

## Installing VirSorter2 in a Python virtual environment {#installing_virsorter2_in_a_python_virtual_environment}

These instructions install VirSorter2 in your \$HOME directory using Alliance\'s prebuilt [Python wheels](http://pythonwheels.com/). Custom Python wheels are stored in `/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`. To install a VirSorter2 wheel, we will use the `pip` command and install it into a [ Python virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment " Python virtual environment"){.wikilink}.

1\. Load the necessary modules.

2\. Create and activate a Python virtual environment.

3\. Install VirSorter2 v2.2.4 in the virtual environment. 2.2.4 }} 4. Validate the installation.

5\. Freeze the environment and requirements set.

6\. Download the database in \$SCRATCH with the `--skip-deps-install` option to bypass conda installation and also because dependencies are already installed.

## Testing VirSorter2 {#testing_virsorter2}

1\. Deactivate your virtual environment

2\. Download the test dataset in `$SCRATCH`.

3\. Create a submission script

3\. Start an interactive job. 2G \--cpus-per-task2 \--account`<your-account>`{=html} }}

`salloc: Granted job allocation 1234567`\
`$ bash test-virsorter.sh             # Run the submission script`\
`$ exit                               # Terminate the allocation`\
`salloc: Relinquishing job allocation 1234567`

Upon a successful test run, you can submit a non-interactive job with your own dataset using [`sbatch`](https://docs.alliancecan.ca/wiki/Running_jobs#Use_sbatch_to_submit_jobs).
