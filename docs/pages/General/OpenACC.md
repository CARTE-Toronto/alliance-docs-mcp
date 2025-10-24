---
title: "OpenACC"
url: "https://docs.alliancecan.ca/wiki/OpenACC"
category: "General"
last_modified: "2020-11-30T17:54:41Z"
page_id: 15290
display_title: "OpenACC"
---

OpenACC makes it relatively easy to offload vectorized code to accelerators such as GPUs, for example. Unlike [CUDA](https://docs.alliancecan.ca/CUDA "CUDA"){.wikilink} and OpenCL where kernels need to be coded explicitly, OpenACC minimizes the amount of modifications to do on a serial or [OpenMP](https://docs.alliancecan.ca/OpenMP "OpenMP"){.wikilink} code. The compiler converts the OpenACC code into a binary executable that can make use of accelerators. The performance of OpenACC codes can be similar to the one of a [CUDA](https://docs.alliancecan.ca/CUDA "CUDA"){.wikilink} code, except that OpenACC requires less code development.

# OpenACC directives {#openacc_directives}

Similar to [OpenMP](https://docs.alliancecan.ca/OpenMP "OpenMP"){.wikilink}, OpenACC can convert a `for` loop into parallel code that would run on an accelerator. This can be achieved with compiler directives `#pragma acc ...` before structured blocks of code like, for example, a `for` loop. All supported `pragma` directives are described in the [OpenACC specification](https://www.openacc.org/specification).

# Code examples {#code_examples}

OpenACC can be used in [Fortran](https://docs.alliancecan.ca/Fortran "Fortran"){.wikilink}, [C](https://docs.alliancecan.ca/C "C"){.wikilink} and [C++](https://docs.alliancecan.ca/C++ "C++"){.wikilink}, which we illustrate here using a simple program that computes a decimal approximation to π based on a definite integral which is equal to arctan(1), i.e. π/4. `<tabs>`{=html} `<tab name="C">`{=html} `{{File
  |name=pi.c
  |lang="C"
  |contents=
#include <stdio.h>

const int vl = 512;
const long long N = 2000000000;

int main(int argc,char** argv) {
  double pi = 0.0f;
  long long i;

  #pragma acc parallel vector_length(vl) 
  #pragma acc loop reduction(+:pi)
  for (i = 0; i < N; i++) {
    double t = (double)((i + 0.5) / N);
    pi += 4.0 / (1.0 + t * t);
  }
  printf("pi = %11.10f\n", pi / N);
  return 0;
}
}}`{=mediawiki} `</tab>`{=html} `<tab name="C++">`{=html} `{{File
  |name=pi.cxx
  |lang="C++"
  |contents=
#include <iostream>
#include <iomanip>

const int vl = 512;
const long long N = 2000000000;

int main(int argc,char** argv) {
  double pi = 0.0f;
  long long i;

  #pragma acc parallel vector_length(vl)
  #pragma acc loop reduction(+:pi)
  for (i = 0; i < N; i++) {
    double t = double((i + 0.5) / N);
    pi += 4.0/(1.0 + t * t);
  }
  std::cout << std::fixed;
  std::cout << std::setprecision(10);
  std::cout << "pi = " << pi/double(N) << std::endl;
  return 0;
}
}}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

# Compilers

## PGI

- Module `pgi`, any version from 13.10
  - Newer versions support newest GPU capabilities.

Compilation example:

`# TODO`

## GCC

- Module `gcc`, any version from 9.3.0
  - Newer versions support newest GPU capabilities.

Compilation example:

`gcc -fopenacc -march=native -O3 pi.c -o pi`

# Tutorial

See our [OpenACC_Tutorial](https://docs.alliancecan.ca/OpenACC_Tutorial "OpenACC_Tutorial"){.wikilink}.

# References

- [OpenACC official documentation - Specification 3.1 (PDF)](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.1-final.pdf)
- [NVIDIA OpenACC API - Quick Reference Guide (PDF)](https://www.nvidia.com/docs/IO/116711/OpenACC-API.pdf)
