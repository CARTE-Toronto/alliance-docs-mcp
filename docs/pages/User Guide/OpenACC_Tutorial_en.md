---
title: "OpenACC Tutorial/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial/en"
category: "User Guide"
last_modified: "2024-10-16T16:56:41Z"
page_id: 1390
display_title: "OpenACC Tutorial"
---

`<languages />`{=html}

This tutorial is strongly inspired from the OpenACC Bootcamp session presented at [GPU Technology Conference 2016](http://www.gputechconf.com/).

OpenACC is an application programming interface (API) for porting code onto accelerators such as GPU and coprocessors. It has been developed by Cray, CAPS, NVidia and PGI. Like in [OpenMP](https://docs.alliancecan.ca/OpenMP "OpenMP"){.wikilink}, the programmer annotates C, C++ or Fortran code to identify portions that should be parallelized by the compiler.

A self-paced course on this topic is available from SHARCNET: [Introduction to GPU Programming](https://training.sharcnet.ca/courses/enrol/index.php?id=173).

## Lesson plan {#lesson_plan}

- [Introduction](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Introduction "Introduction"){.wikilink}
- [Gathering a profile and getting compiler information](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Profiling "Gathering a profile and getting compiler information"){.wikilink}
- [Expressing parallelism with OpenACC directives](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Adding_directives "Expressing parallelism with OpenACC directives"){.wikilink}
- [Expressing data movement](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Data_movement "Expressing data movement"){.wikilink}
- [Optimizing loops](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Optimizing_loops "Optimizing loops"){.wikilink}

## External references {#external_references}

Here are some useful external references:

- [OpenACC Programming and Best Practices Guide (PDF)](https://www.openacc.org/sites/default/files/inline-files/openacc-guide.pdf)
- [OpenACC API 2.7 Reference Guide (PDF)](https://www.openacc.org/sites/default/files/inline-files/API%20Guide%202.7.pdf)
- [Getting Started with OpenACC](https://developer.nvidia.com/blog/getting-started-openacc/)
- [PGI Compiler](https://docs.nvidia.com/hpc-sdk/pgi-compilers/legacy.html)
- [PG Profiler](http://www.pgroup.com/resources/pgprof-quickstart.htm)
- [NVIDIA Visual Profiler](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual-profiler)
