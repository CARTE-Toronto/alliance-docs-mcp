---
title: "BEAST/en"
url: "https://docs.alliancecan.ca/wiki/BEAST/en"
category: "General"
last_modified: "2022-12-12T15:34:23Z"
page_id: 8812
display_title: "BEAST"
---

`<languages />`{=html}

## Description

BEAST[^1] is a cross-platform program for Bayesian MCMC analysis of molecular sequences. It is entirely orientated towards rooted, time-measured phylogenies inferred using strict or relaxed molecular clock models. It can be used as a method of reconstructing phylogenies but is also a framework for testing evolutionary hypotheses without conditioning on a single tree topology. BEAST uses MCMC to average over tree space, so that each tree is weighted proportional to its posterior probability.

BEAST can use the beagle-lib[^2], which is a high-performance library that can perform the core calculations at the heart of most Bayesian and Maximum Likelihood phylogenetics packages.

## Usage

Loading the BEAST module with: `module load beast`, will automatically load it\'s dependencies, namely the `beagle-lib` and `java` modules, and set the environment variable `EBROOTBEAST` to point to the directory where BEAST\'s program files are located.

### Managing BEAST Packages/Add-ons {#managing_beast_packagesadd_ons}

BEAST has been installed without any packages (add-ons). You can use the `packagemanager` command (for BEAST v2.5.1 and newer; in older versions of BEAST, the command is `addonmanager`) to install the desired packages within your home directory. `<tabs>`{=html} `<tab name="Beast &ge; 2.5.x" >`{=html}

` $ module load beast/2.5.1`\
` $ packagemanager -list`\
` Name    | Installation Status | Latest Version | Dependencies | Description`\
` --------------------------------------------------------------------------`\
` BEAST   | 2.5.1               | 2.5.0          |              | BEAST core`\
` --------------------------------------------------------------------------`\
` bacter  | NA                  | 2.2.0          |              | Bacterial ARG inference.`\
` BADTRIP | NA                  | 1.0.0          |              | Infer transmission time for [...]`\
` [...]`\
` SNAPP   | NA                  | 1.4.1          |              | SNP and AFLP Phylogenies`\
` [...]`\
`   `\
` $ packagemanager -add SNAPP`\
` Package SNAPP is installed in ~/.beast/2.5/SNAPP.`\
` `\
` $ packagemanager -list`\
` Name    | Installation Status | Latest Version | Dependencies | Description`\
` --------------------------------------------------------------------------`\
` BEAST   | 2.5.1               | 2.5.0          |              | BEAST core`\
` --------------------------------------------------------------------------`\
` [...]`\
` SNAPP   | 1.4.1               | 1.4.1          |              | SNP and AFLP Phylogenies`\
` [...]`

`</tab>`{=html} `<tab name="Beast &le; 2.4.x">`{=html}

` $ module load beast/2.4.0`\
` $ addonmanager -list`\
` Name    | Installation Status | Latest Version | Dependencies | Description`\
` ---------------------------------------------------------------------------`\
` BEAST   | 2.4.0               | 2.4.8          |              | BEAST core`\
` ---------------------------------------------------------------------------`\
` bacter  | not installed       | 1.2.3          |              | Bacterial ARG inference.`\
` BASTA   | not installed       | 2.3.2          |              | Bayesian structured coalescent approximation`\
` [...]`\
` SNAPP   | not installed       | 1.3.0          |              | SNP and AFLP Phylogenies`\
` [...]`\
` `\
` $ addonmanager -add SNAPP`\
` Package SNAPP is installed in ~/.beast/2.4/SNAPP.`\
` `\
` $ addonmanager -list`\
` Name    | Installation Status | Latest Version | Dependencies | Description`\
` ---------------------------------------------------------------------------`\
` BEAST   | 2.4.0               | 2.4.8          |              | BEAST core`\
` ---------------------------------------------------------------------------`\
` [...]`\
` SNAPP   | 1.3.0               | 1.3.0          |              | SNP and AFLP Phylogenies`\
` [...]`

`</tab>`{=html} `</tabs>`{=html}

For more information on how to manage BEAST packages please read the section \"Server machines\" at: <http://www.beast2.org/managing-packages/>

### Simple Jobscript for BEAST {#simple_jobscript_for_beast}

### Jobscript for BEAST with more Memory {#jobscript_for_beast_with_more_memory}

```{=mediawiki}
{{File
  |name=high_memory_beast_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=4000M

# Increase Maximum memory here if necessary:
# "BEAST_MEM" needs to be 250M lower than "--mem="
BEAST_MEM="-Xmx3750M"

module load beast/2.6.3

# Define variables where to find BEAST and BEAGLE-lib
BEAST_LIB="${EBROOTBEAST}/lib"
BEAST_EXTRA_LIBS="${BEAST_LIB}:${BEAGLE_LIB}"
export LD_LIBRARY_PATH="${BEAGLE_LIB}:${LD_LIBRARY_PATH}"

# Build a long java command:
CMD="java -Xms256m ${BEAST_MEM}"                                           # set memory
CMD="$CMD -Djava.library.path=${BEAST_EXTRA_LIBS}"                         # point to libraries
CMD="$CMD -cp ${BEAST_LIB}/launcher.jar beast.app.beastapp.BeastLauncher" # which program to execute

echo ".................................."
echo "The Java command is \"${CMD}\""
echo ".................................."

# Run the command:
$CMD -beagle  input_beast.xml

}}
```
## References

<references />

[^1]: BEAST2 Homepage: <http://beast2.org/>

[^2]: Beagle-lib Homepage: <https://github.com/beagle-dev/beagle-lib>
