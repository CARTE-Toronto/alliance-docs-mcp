---
title: "CFOUR/en"
url: "https://docs.alliancecan.ca/wiki/CFOUR/en"
category: "General"
last_modified: "2024-04-03T17:28:45Z"
page_id: 9101
display_title: "CFOUR"
---

= Introduction =

"CFOUR (Coupled-Cluster techniques for Computational Chemistry) is a program package for performing high-level quantum chemical calculations on atoms and molecules. The major strength of the program suite is its rather sophisticated arsenal of high-level ab-initio methods for the calculation of atomic and molecular properties. Virtually all approaches based on Møller-Plesset (MP) perturbation theory and the coupled-cluster approximation (CC) are available; most of these have complementary analytic derivative approaches within the package as well."

"CFOUR is not a commercial code. It is rather a program that is undergoing development; new techniques and improvements are constantly being made." See the CFOUR web site for more information.

= License limitations =

The Alliance has signed a license agreement with Prof. Dr. J. Gauss who acts for the developers of the CFOUR Software.

In order to use the current installed version on the Alliance systems, each user must agree to certain conditions. Please  contact support with a copy of the following statement:

# I will use CFOUR only for academic research.
# I will not copy the CFOUR software, nor make it available to anyone else.
# I will properly acknowledge original papers related to CFOUR and to the Alliance in my publications (see the license form for more details).
# I understand that the agreement for using CFOUR can be terminated by one of the parties: CFOUR developers or the Alliance.
# I will notify the Alliance of any change in the above acknowledgement.

When your statement is received, we will allow you to access the program.

= Module =

You can access the MPI version of CFOUR by loading a module.

module load intel/2023.2.1  openmpi/4.1.5 cfour-mpi/2.1

For the serial version, use:

module load intel/2023.2.1 cfour/2.1

There is a mailing list as a forum for user experiences with the CFOUR program system. For how to subscribe and other information, see this page.

== Examples and job scripts ==

To run CFOUR, you need to have at least the input file ZMAT with all information concerning geometry, requested quantum-chemical method, basis set, etc. The second file is GENBAS that contains the required information for the basis sets available to the user. If GENBAS is not present in the directory from where you start your job, CFOUR will create a symlink and use the existing file provided by the module. The file is located at: $EBROOTCFOUR/basis/GENBAS.

= Related links =

* Manual
* Features