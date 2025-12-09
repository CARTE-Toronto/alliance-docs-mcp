---
title: "MAFFT/en"
url: "https://docs.alliancecan.ca/wiki/MAFFT/en"
category: "General"
last_modified: "2020-07-29T15:36:02Z"
page_id: 14095
display_title: "MAFFT"
---

`<languages />`{=html} [MAFFT](https://mafft.cbrc.jp/alignment/software/) is a multiple sequence alignment program for unix-like operating systems. It offers a range of multiple alignment methods, L-INS-i (accurate; for alignment of \<∼200 sequences), FFT-NS-2 (fast; for alignment of \<∼30,000 sequences), etc.

## Single node {#single_node}

MAFFT can benefit from multiple cores on a single node. For more information: <https://mafft.cbrc.jp/alignment/software/multithreading.html>

**Note**: The `MAFFT_TMPDIR` is set to `$SLURM_TMPDIR/maffttmp` when you load the module.

## Multiple nodes (MPI) {#multiple_nodes_mpi}

MAFFT can use MPI to align a large number of sequences: <https://mafft.cbrc.jp/alignment/software/mpi.html>

**Note**: `MAFFT_TMPDIR` is set to `$SCRATCH/maffttmp` when you load the module. If you change this temporary directory, it must be shared by all hosts.
