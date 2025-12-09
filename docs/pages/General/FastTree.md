---
title: "FastTree/en"
url: "https://docs.alliancecan.ca/wiki/FastTree/en"
category: "General"
last_modified: "2024-12-17T21:19:57Z"
page_id: 27265
display_title: "FastTree"
---

`<languages/>`{=html}

[FastTree](https://morgannprice.github.io/fasttree/) infers approximately-maximum-likelihood phylogenetic trees from alignments of nucleotide or protein sequences. FastTree can handle alignments with up to a million sequences in a reasonable amount of time and memory.

# Environment modules {#environment_modules}

We offer software modules for single precision and double precision calculations. Single precision is faster while double precision is more precise. Double precision is recommended when using a highly biased transition matrix, or if you want to resolve very short branches accurately.

To see the available FastTree modules:

`module spider fasttree`

To load a single precision module:

`module load fasttree/2.1.11`

To load a double precision module:

`module load fasttree-double/2.1.11`

# Troubleshooting

- Error message *WARNING! This alignment consists of closely-related and very long sequences*: This likely results in very short and sometimes negative branch lengths. Use a `fasttree-double` module for double precision.

# References

- <https://morgannprice.github.io/fasttree/> FastTree Web page\]
