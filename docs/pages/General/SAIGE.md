---
title: "SAIGE/en"
url: "https://docs.alliancecan.ca/wiki/SAIGE/en"
category: "General"
last_modified: "2026-01-29T14:36:01Z"
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

== Installing SAIGE under the environment StdEnv/2020 ==

1.  Load the appropriate modules.

2. Create the installation directory.
~/.local/R/$EBVERSIONR/
}}
3. Install the R dependencies. It is important to install these exact versions. During installation, if you are prompted to install the latest version of any dependency, simply press Enter to decline.

[name@server ~]$ R -e 'install.packages("remotes", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("Rcpp", version="1.0.10", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("RcppParallel", version="5.1.6", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("data.table", version="1.17.8", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("RcppArmadillo", version="14.0.2-1", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("SPAtest", version="3.1.2", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("RcppEigen", version="0.3.3.9.3", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("BH", version="1.81.0-1", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("optparse", version="1.7.3", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("SKAT", version="2.2.5", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("MetaSKAT", version="0.82", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("qlcMatrix", version="0.9.5", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("RhpcBLASctl", version="0.23-42", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("RSQLite", version="2.3.8", repos="https://cloud.r-project.org/")'
[name@server ~]$ R -e 'remotes::install_version("dplyr", version="1.1.0", repos="https://cloud.r-project.org/")'

4. Download SAIGE version 1.0.0.

5. Patch the installation.

First, remove the configure file to avoid installing already available dependencies. Then, change the library name to correctly link to the Makevars file to make sure that the linking options will use FlexiBLAS. Doing so will prevent the i>unable to find -llapack error message displayed at installation. Read more information on FlexiBLAS, BLAS and LAPACK.

6. Compile and install.

7. Test that it is available.