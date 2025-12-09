---
title: "MRCC/en"
url: "https://docs.alliancecan.ca/wiki/MRCC/en"
category: "General"
last_modified: "2024-05-08T21:09:02Z"
page_id: 25469
display_title: "MRCC"
---

`<languages />`{=html}

## Introduction

[MRCC](https://www.mrcc.hu/index.php) is a suite of ab initio and density functional quantum chemistry programs for high-accuracy electronic structure calculations developed and maintained by the quantum chemistry research group at the Department of Physical Chemistry and Materials Science, TU Budapest. Its special feature, the use of automated programming tools enabled the development of tensor manipulation routines which are independent of the number of indices of the corresponding tensors, thus significantly simplifying the general implementation of quantum chemical methods. Applying the automated tools of the program several quantum chemistry models and techniques of high complexity have been implemented so far, including arbitrary single-reference coupled-cluster (CC) and configuration interaction (CI) methods, multi-reference CC approaches, CC and CI energy derivatives and response functions, arbitrary perturbative CC approaches. Many features of the package are also available with relativistic Hamiltonians allowing for accurate calculations on heavy element systems. The developed cost-reduction techniques and local correlation approaches also enable high-precision calculations for medium-sized and large molecules.

## License limitations {#license_limitations}

The Alliance has signed a license agreement with Prof. Dr. Mihaly Kallay who acts for the developers of the MRCC Software.

In order to use the current installed version on the Alliance systems, each user must agree to certain conditions. Please contact support with a copy of the following statement:

1\) I will use MRCC only for academic research.

2\) I will not copy the MRCC software, nor make it available to anyone else.

3\) I will properly acknowledge original papers related to MRCC and to the Alliance in my publications, for more details: <https://www.mrcc.hu/index.php/citation>

4\) I understand that the agreement for using MRCC can be terminated by one of the parties: MRCC developers or the Alliance.

5\) I will notify the Alliance of any change in the above acknowledgement.

## Module

The MRCC version from \"2023-08-28\" is available on all clusters by loading a [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink}:

``` bash
module load intel/2023.2.1  openmpi/4.1.5 mrcc/20230828
```

The module was installed with OpenMP and MPI support. Once the module is loaded, you can access all the binaries and the basis. The list of binaries is:

``` bash
[ ~]$ module load intel/2023.2.1  openmpi/4.1.5 mrcc/20230828
[ ~]$ ls $EBROOTMRCC/bin/
ccsd      cis                      dmrcc      drpa      goldstone  minp      mrcc      mulli   ovirt  qmmod  scf_mpi  xmrcc
ccsd_mpi  dirac_mointegral_export  dmrcc_mpi  drpa_mpi  integ      mp2f12    mrcc_mpi  orbloc  prop   scf    uccsd    xmrcc_mpi
```

## Examples and job scripts {#examples_and_job_scripts}

Coming soon

## Citations

As indicated in the license, users are asked to cite the original papers in their publications. For more information, please [see this page](https://www.mrcc.hu/index.php/citation).

## Documentation

- A detailed documentation about the usage of the program is available on the [MRCC](https://www.mrcc.hu/index.php/documentation) website.
- Another useful source of information about the program is [MRCC forum](https://www.mrcc.hu/index.php/forum).
