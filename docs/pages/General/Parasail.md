---
title: "Parasail/en"
url: "https://docs.alliancecan.ca/wiki/Parasail/en"
category: "General"
last_modified: "2024-06-28T14:07:49Z"
page_id: 25885
display_title: "Parasail"
---

`<languages />`{=html}

[parasail](https://github.com/jeffdaily/parasail) is a SIMD C (C99) library containing implementations of the Smith-Waterman (local), Needleman-Wunsch (global), and various semi-global pairwise sequence alignment algorithms.

# Usage

Find the required versions using

and load the library using

## parasail_aligner Example {#parasail_aligner_example}

When using the binary `parasail_aligner`, it is important to set the number of threads according to the number of cores allocated in our job. We can set it with

``` bash
parasail_aligner -t ${SLURM_CPUS_PER_TASK:-1} ...}}
```

## Python extension {#python_extension}

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run

### Usage {#usage_1}

1\. Load the required modules.

2\. Import parasail 1.3.4.

If the command displays nothing, the import was successful.

### Example

Run a quick local alignment score comparison between BioPython and parasail.

1\. Write the Python script:

2\. Write the job submission script: `<tabs>`{=html} `<tab name="Default StdEnv">`{=html}

`</tab>`{=html} `<tab name="StdEnv/2020">`{=html} 2.1. Identify available wheels first :

Install the desired version in your virtual environment:

`</tab>`{=html} `</tabs>`{=html}

3\. Submit the job with

4\. When the job has run, the output will be in the Slurm output file:

#### Available Python packages {#available_python_packages}

Other Python packages that depend on parasail will have their requirement satisfied by loading the parasail module:

`grep parasail`

\|result= parasail 1.3.4 }}
