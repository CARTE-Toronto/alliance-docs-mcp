---
title: "SAIGE/en"
url: "https://docs.alliancecan.ca/wiki/SAIGE/en"
category: "General"
last_modified: "2023-07-07T16:32:21Z"
page_id: 22859
display_title: "SAIGE"
---

SAIGE
is an R package developed with Rcpp for genome-wide association tests in large-scale data sets and biobanks.

The method

* accounts for sample relatedness based on the generalized mixed models;

* allows for model fitting with either full or sparse genetic relationship matrix (GRM);

* works for quantitative and binary traits;

* handles case-control imbalance of binary traits;

* computationally efficient for large data sets;

* performs single-variant association tests;

* provides effect size estimation through Firth’s Bias-Reduced Logistic Regression;

* performs conditional association analysis.

This page discusses how to install SAIGE package 1.0.0.

== Installing SAIGE ==

1.  Load the appropriate modules.

2. Create the installation directory.
~/.local/R/$EBVERSIONR/
}}
3. Install the R dependencies.
"https://cloud.r-project.org/")'
}}

4. Download SAIGE version 1.0.0.

5. Patch the installation.

First, remove the configure file to avoid installing already available dependencies. Then, change the library name to correctly link to the Makevars file to make sure that the linking options will use FlexiBLAS. Doing so will prevent the i>unable to find -llapack error message displayed at installation. Read more information on FlexiBLAS, BLAS and LAPACK.

6. Compile and install.

7. Test that it is available.