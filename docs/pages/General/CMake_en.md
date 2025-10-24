---
title: "CMake/en"
url: "https://docs.alliancecan.ca/wiki/CMake/en"
category: "General"
last_modified: "2023-06-18T00:45:02Z"
page_id: 2297
display_title: "CMake"
---

`<languages />`{=html}

## Description

[CMake](http://www.cmake.org/) is a free multi-language multi-platform compilation tool (the name stands for `<i>`{=html}cross-platform make`</i>`{=html}). Although [Autotools](https://docs.alliancecan.ca/Autotools "Autotools"){.wikilink} is the traditional tool used on Linux---by GNU projects among others---, various projects have changed to CMake during the last few years, for different reasons, including for example KDE and MySQL. Those who have already had difficulties with Autotools in their own project will probably find CMake much easier to use. In fact, according to KDE, the main reason for which they have changed from Autotools to CMake is that the compilation is a lot quicker and the build files are a lot easier to write.

## Basic usage {#basic_usage}

CMake works in the same way as that Autotools requires to run a `configure` script, followed by a build with `make`. However, instead of calling `./configure`, you call `cmake ``<i>`{=html}`directory``</i>`{=html}. For example, if you are inside the directory where you would like to build the application, you run

Hence, to configure, build and install an application or a library, the simplest way to do this is with

## Useful options on our clusters {#useful_options_on_our_clusters}

Our clusters are configured such that compilation of a new software package will automatically add information to the resulting binary to ensure that it finds the libraries that it depends on. This is done through a mechanism called `RUNPATH` or `RPATH`. Some packages using CMake also do the same, through a feature provided by CMake. When both of these are used at the same time, it sometimes creates conflicts. In order to avoid errors related to this, you can add the option

- `-DCMAKE_SKIP_INSTALL_RPATH=ON`

to your command line. Moreover, our clusters have libraries installed in non-standard locations. This sometimes causes CMake not to find them easily. It can be useful to the following option to your `cmake` command invocation:

- `-DCMAKE_SYSTEM_PREFIX_PATH=$EBROOTGENTOO`

Sometimes, even this is not sufficient, and you may have to add more specific options for libraries that are used by your software package. For example:

- `-DCURL_LIBRARY=$EBROOTGENTOO/lib/libcurl.so -DCURL_INCLUDE_DIR=$EBROOTGENTOO/include`
- `-DPYTHON_EXECUTABLE=$EBROOTPYTHON/bin/python`
- `-DPNG_PNG_INCLUDE_DIR=$EBROOTGENTOO/include -DPNG_LIBRARY=$EBROOTGENTOO/lib/libpng.so`
- `-DJPEG_INCLUDE_DIR=$EBROOTGENTOO/include -DJPEG_LIBRARY=$EBROOTGENTOO/lib/libjpeg.so`
- `-DOPENGL_INCLUDE_DIR=$EBROOTGENTOO/include -DOPENGL_gl_LIBRARY=$EBROOTGENTOO/lib/libGL.so -DOPENGL_glu_LIBRARY=$EBROOTGENTOO/lib/libGLU.so`
- `-DZLIB_ROOT=$EBROOTGENTOO`

## Customizing the configuration {#customizing_the_configuration}

Just like with `autotools`, it is possible to customize the configuration of an application or a library. This can be done by different command line options, but also using a command line interface with the `ccmake` command.

### `ccmake`

You call `ccmake` in the same way as you call `cmake`, by giving the directory to build from. So if this is the current directory, you should call

You should run `ccmake` `<i>`{=html}after`</i>`{=html} having ran `cmake`. So, generally you would do

`ccmake` gives you then a list of options that are defined by the project. You will then see a relatively short list like this:

As is written at the bottom of this display, you can edit a value by pressing the `enter` key. If you modify a value, you will want to press the `c` key to try out the configuration with this new value. If this new configuration succeeds with that new value, you will then have the option `g` to generate the `Makefile` with the new configuration, or you can quit using the `q` key. Lastly, you can activate advanced mode using the `t` key. You will then have a much longer list of variables which allows you to precisely configure the application. Here is a list of options

As you can see, `ccmake` in advanced mode displays equally well the libraries that were found as those that were not found. If you would like to use a specific version of [BLAS](https://docs.alliancecan.ca/BLAS_and_LAPACK/en "BLAS"){.wikilink} for example, you will immediately know which one was found by CMake, and modify this if necessary. `ccmake` also displays the list of flags that are passed to the C, C++, and other compilers, to the linker, depending on the build type.

### Command line options {#command_line_options}

All command line options that are displayed by `ccmake` can be modified on the command line, using the following syntax: VALUE}}

For example, to specify the install location: /home/user/my_directory}}

To configure the compilation, you might want to change the following values:

  Option                        Description
  ----------------------------- ---------------------------------------------
  `CMAKE_C_COMPILER`            Change the C compiler
  `CMAKE_CXX_COMPILER`          Change the C++ compiler
  `CMAKE_LINKER`                Change the linker
  `CMAKE_C_FLAGS`               Change the flags passed to the C compiler
  `CMAKE_CXX_FLAGS`             Change the flags passed to the C++ compiler
  `CMAKE_SHARED_LINKER_FLAGS`   Change the flags passed to the linker
                                

A more exhaustive list option is available [on the official CMake page](http://www.cmake.org/Wiki/CMake_Useful_Variables).

If you do not want to get into adventures with these specific options, CMake also provides a simpler option, called `CMAKE_BUILD_TYPE`. This option defines which compilation type must be used. Possible values are

  Option           Description
  ---------------- ----------------------------------------------------------------------------------------------
  \-               No value
  Debug            Activate debugging options, deactivate optimization options
  Release          Deactivate debugging options, activate usual optimizations
  MinSizeRel       Deactivate debugging options, activate optimization options that minimize the binary\'s size
  RelWithDebInfo   Activate debugging options and usual optimizations
                   

These different compilation types define compiler options that vary from compiler to compiler. So you do not need to check which exact compiler flags have to be used.

## References

- [A simple example](http://www.cmake.org/cmake/help/examples.html) on the official site.
- [A tutorial](http://www.cmake.org/cmake/help/cmake_tutorial.html) that is more complete on the official site.
