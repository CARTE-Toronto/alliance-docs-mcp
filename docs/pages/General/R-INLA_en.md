---
title: "R-INLA/en"
url: "https://docs.alliancecan.ca/wiki/R-INLA/en"
category: "General"
last_modified: "2025-07-16T19:13:42Z"
page_id: 28933
display_title: "R-INLA"
---

`<languages />`{=html}

[R-INLA](https://www.r-inla.org/) is a package in [R](https://docs.alliancecan.ca/R "R"){.wikilink} that do approximate Bayesian inference for Latent Gaussian Models.

## Installation

The installation of the R-INLA package is a bit more complicated than most [R](https://docs.alliancecan.ca/R "R"){.wikilink} packages, as it downloads other pre-compiled executables that need to be made compatible with our [Standard software environment](https://docs.alliancecan.ca/Standard_software_environments "Standard software environment"){.wikilink}.

The scripts below have been tested with the versions mentioned therein. Because R will always install the latest versions of packages, the versions of the modules will likely have to be adjusted in the future.

`<tabs>`{=html} `<tab name="StdEnv/2023 r/4.4">`{=html} `{{File
  |name=install_INLA_StdEnv2023.sh
  |lang="bash"
  |contents=
#!/bin/bash

# (1)
module load StdEnv/2023 gcc/12.3 r/4.4.0 geos/3.12.0 gdal/3.9.1 udunits/2.2.28 gsl/2.7 jags/4.3.2

LOGFILE=r_INLA_install_${EBVERSIONR}_${CC_CLUSTER}_$(date --iso=min).log

# (2)
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/${EBVERSIONR:0:3}"
echo  "R_LIBS is $R_LIBS"
mkdir -p $R_LIBS
R -e 'install.packages("remotes", repos=c("https://mirror.csclub.uwaterloo.ca/CRAN/"))'
R -e 'install.packages("INLA", version="25.06.07", repos=c("https://mirror.csclub.uwaterloo.ca/CRAN/", INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE, Ncpus=2)' \
    & tee  $LOGFILE

# (3)
R -e 'library(INLA); inla.binary.install(os="Rocky Linux-8")' & tee -a $LOGFILE

# (4)
chmod u+x $R_LIBS/INLA/bin/linux/64bit/{*.so.*,*.so,first/*.so}
sed -i  's/\(^.*export LD_LIBRARY_PATH\)/echo "Skipping LD_LIBRARY_PATH." #\1/g' $R_LIBS/INLA/bin/linux/64bit/*.run
setrpaths.sh --path $R_LIBS/INLA/bin/linux/64bit/malloc --add_path "\$ORIGIN:$EBROOTGENTOO/lib/gcc/x86_64-pc-linux-gnu/${EBVERSIONGCC::2}"
setrpaths.sh --path $R_LIBS/INLA/bin/linux/64bit --add_path '$ORIGIN/first:$ORIGIN:$ORIGIN/malloc'
}}`{=mediawiki}`</tab>`{=html} `<tab name="StdEnv/2020 r/4.2.1">`{=html} `{{File
  |name=install_INLA_StdEnv2020.sh
  |lang="bash"
  |contents=
#!/bin/bash

# (1)
module load StdEnv/2020 gcc/9.3.0 r/4.2.1 geos/3.9.1 gdal/3.2.3 udunits/2.2.26 gsl/2.6

LOGFILE=r_INLA_install_${EBVERSIONR}_${CC_CLUSTER}_$(date --iso=sec).log

# (2)
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/${EBVERSIONR:0:3}"
echo  "R_LIBS is $R_LIBS"
mkdir -p $R_LIBS
R -e 'install.packages("INLA", repos=c("https://cran.utstat.utoronto.ca/", INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE, Ncpus=2)' \
     & tee  $LOGFILE

# (3)
R -e 'library(INLA); inla.binary.install(os="CentOS Linux-7")' & tee -a $LOGFILE

# (4)
chmod u+x $R_LIBS/INLA/bin/linux/64bit/{*.so.*,*.so,first/*.so}
sed -i  's/\(^.*export LD_LIBRARY_PATH\)/echo "Skipping LD_LIBRARY_PATH." #\1/g' $R_LIBS/INLA/bin/linux/64bit/*.run
setrpaths.sh --path $R_LIBS/INLA/bin/linux --add_path '$ORIGIN/first:$ORIGIN'

}}`{=mediawiki}`</tab>`{=html} `</tabs>`{=html} Comments in the script:

- \(1\) Load required modules. The same modules have to be loaded in the job script as well.
- \(2\) Install the R-INLA package and its dependencies
- \(3\) Install the pre-compiled executables that R-INLA needs
- \(4\) Patch the pre-compiled executables so that they are compatible with our [Standard software environment](https://docs.alliancecan.ca/Standard_software_environments "Standard software environment"){.wikilink}
