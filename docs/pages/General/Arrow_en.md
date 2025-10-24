---
title: "Arrow/en"
url: "https://docs.alliancecan.ca/wiki/Arrow/en"
category: "General"
last_modified: "2024-07-16T15:13:02Z"
page_id: 13183
display_title: "Arrow"
---

`<languages />`{=html}

[Apache Arrow](https://arrow.apache.org/) is a cross-language development platform for in-memory data. It uses a standardized language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations. It also provides computational libraries and zero-copy streaming messaging and interprocess communication. Languages currently supported include C, C++, C#, Go, Java, JavaScript, MATLAB, Python, R, Ruby, and Rust.

## CUDA

Arrow is also available with CUDA.

where X.Y.Z represent the desired version.

## Python bindings {#python_bindings}

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run

where `X.Y.Z` represent the desired version.

Or search directly *pyarrow*, by running

### PyArrow

The Arrow Python bindings (also named *PyArrow*) have first-class integration with NumPy, Pandas, and built-in Python objects. They are based on the C++ implementation of Arrow.

1\. Load the required modules.

where `X.Y.Z` represent the desired version.

2\. Import PyArrow.

If the command displays nothing, the import was successful.

For more information, see the [Arrow Python](https://arrow.apache.org/docs/python/) documentation.

#### Fulfilling other Python package dependency {#fulfilling_other_python_package_dependency}

Other Python packages depends on PyArrow in order to be installed. With the `arrow` module loaded, your package dependency for `pyarrow` will be satisfied.

`grep pyarrow`

\|result= pyarrow 17.0.0 }}

#### Apache Parquet format {#apache_parquet_format}

The [Parquet](http://parquet.apache.org/) file format is available.

To import the Parquet module, execute the previous steps for `pyarrow`, then run

If the command displays nothing, the import was successful.

## R bindings {#r_bindings}

The Arrow package exposes an interface to the Arrow C++ library to access many of its features in R. This includes support for analyzing large, multi-file datasets ([open_dataset()](https://arrow.apache.org/docs/r/reference/open_dataset.html)), working with individual Parquet files ([read_parquet()](https://arrow.apache.org/docs/r/reference/read_parquet.html), [write_parquet()](https://arrow.apache.org/docs/r/reference/write_parquet.html)) and Feather files ([read_feather()](https://arrow.apache.org/docs/r/reference/read_feather.html), [write_feather()](https://arrow.apache.org/docs/r/reference/write_feather.html)), as well as lower-level access to the Arrow memory and messages.

### Installation

1\. Load the required modules.

2\. Specify the local installation directory. \~/.local/R/\$EBVERSIONR/ }}

3\. Export the required variables to ensure you are using the system installation. \$EBROOTARROW/lib/pkgconfig \|export INCLUDE_DIR\$EBROOTARROW/include \|export LIB_DIR\$EBROOTARROW/lib }}

4\. Install the bindings. \"<https://cloud.r-project.org/>\")\'}}

### Usage

After the bindings are installed, they have to be loaded.

1\. Load the required modules.

2\. Load the library.

For more information, see the [Arrow R documentation](https://arrow.apache.org/docs/r/index.html)
