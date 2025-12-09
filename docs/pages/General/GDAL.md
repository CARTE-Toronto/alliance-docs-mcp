---
title: "GDAL/en"
url: "https://docs.alliancecan.ca/wiki/GDAL/en"
category: "General"
last_modified: "2025-02-11T16:22:00Z"
page_id: 24833
display_title: "GDAL"
---

`<languages />`{=html}

[GDAL](https://www.gdal.org/) is an open source translator library for raster geospatial data formats. It can be used as a a library, as which presents a single abstract data model to the calling application for all supported formats. It also comes with a variety of useful command line utilities for data translation and processing.

GDAL is used by a [long list of software packages](https://gdal.org/software_using_gdal.html#software-using-gdal) and its functionality can be used in scripts written in [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} or [R](https://docs.alliancecan.ca/R "R"){.wikilink}.

## Using GDAL from Python {#using_gdal_from_python}

GDAL functionality can be used via the [osgeo](https://gdal.org/api/python/osgeo.html) package, which we install as an extension to the GDAL module. In order to use it, you need to load a compatible Python module alongside the GDAL module.

### Using osgeo under StdEnv/2020 {#using_osgeo_under_stdenv2020}

Check which Python modules are compatible with e.g. `gdal/3.5.1`:

We have the choice between Python 3.8, 3.9 and 3.10. Let\'s choose `python/3.10` for this.

### Using osgeo under StdEnv/2023 {#using_osgeo_under_stdenv2023}

Check which Python modules are compatible with e.g. `gdal/3.7.2`:

We have the choice between Python 3.10 and 3.11. Let\'s choose `python/3.11` for this.

## Using GDAL from R {#using_gdal_from_r}

Several [R-packages for Analysis of Spatial Data](https://cran.r-project.org/web/views/Spatial.html) directly depend on GDAL as a System dependency. For example:

- [sf](https://CRAN.R-project.org/package=sf): Simple Features for R
- [terra](https://CRAN.R-project.org/package=terra): Spatial Data Analysis

The older package [rgdal](https://CRAN.R-project.org/package=rgdal) has been discontinued in favor of sf and terra.

### Installing `sf` and `terra` under StdEnv/2020 {#installing_sf_and_terra_under_stdenv2020}

Installing these packages not only requires loading a `gdal` module, but also `udunits` which is required by [units](https://CRAN.R-project.org/package=units).

### Installing `sf` and `terra` under StdEnv/2023 {#installing_sf_and_terra_under_stdenv2023}

Note that under StdEnv/2023, in addition to modules `gdal` and `udunits` also `hdf/4.3.1` is required.
