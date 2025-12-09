---
title: "Installing software in your home directory/en"
url: "https://docs.alliancecan.ca/wiki/Installing_software_in_your_home_directory/en"
category: "Getting Started"
last_modified: "2025-10-22T18:37:28Z"
page_id: 2933
display_title: "Installing software in your home directory"
---

`<languages />`{=html} Most academic software is freely available on the internet. You can email Alliance [support](https://docs.alliancecan.ca/mailto:support@tech.alliancecan.ca) staff, provide them with a URL, and ask them to install any such package so that you and other users will be able to access it via a [module load](https://docs.alliancecan.ca/Using_modules "module load"){.wikilink} command. If the license terms and technical requirements are met they will make it available, as soon as possible.

You are permitted to install software in your own home space or project space if you wish. You might choose to do this, for example,

- if you plan to make your own modifications to the code, or
- if you wish to evaluate it quickly.

`<b>`{=html}Read the installation instructions that accompany the software.`</b>`{=html} These instructions often fall into one of the classes described below.

## configure; make; make install {#configure_make_make_install}

is a very common instruction pattern. Variations include `cmake .` replacing `./configure`, and `sudo make install` replacing `make install`.

Sometimes this will work exactly as prescribed, but sometimes it will fail at `make install` because the package expects to be able to write to `/usr/local` or some other shared area in the file system. It will always fail if `sudo make install` is attempted, because `sudo` is a request for \"root\" or administrator privileges. The usual solution is to supply a `--prefix` flag at the `configure` step, to direct the installation to go to the directory of your choice, e.g., /my/project/directory/some-package && make && make install}}

If other errors arise, contact [support](https://docs.alliancecan.ca/mailto:support@computecanada.ca). For more information see [Make](https://docs.alliancecan.ca/Make "Make"){.wikilink}, [Autotools](https://docs.alliancecan.ca/Autotools "Autotools"){.wikilink}, and [CMake](https://docs.alliancecan.ca/CMake "CMake"){.wikilink}.

## Using libraries {#using_libraries}

Normally the simplest way to make use of a library on a Alliance system is to first load the corresponding module.

With the module loaded, you can now modify the link phase of your build process to include the library, for example

if I wanted to link with the NetCDF library.

The link line needs to contain `-l` prefixed to the library name, which will be a file that has the extension `.a` or `.so`. The documentation for the library will typically inform you of the name of this file and, if there is more than one such file, the order in which they should be linked.

You will also need to load the library module when you wish to run this software, not only during the building of it.

Loading a library [ module](https://docs.alliancecan.ca/Using_modules " module"){.wikilink} will set environment variables `CPATH` and `LIBRARY_PATH` pointing to the location of the library itself and its header files. These environment variables are supported by most compilers (for example [Intel](https://software.intel.com/en-us/node/522775) and [GCC](https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html)), which will automatically try the directories listed in those environment variables during compilation and linking phases. This feature allows you to easily link against the library without specifying its location explicitly by passing the `-I` and `-L` options to the compiler. If your make- or config- file calls for an explicit location of the library to pass to the compiler via `-I` and `-L`, you can usually omit the location of the library and leave these lines blank in the make- or config- file.

In some cases, however, particularly with `cmake`, it may be necessary to specify explicitly the location of the library provided by the module. The preferred and the most robust way to do so is to use an EasyBuild environment variable, `EBROOT...`, instead of manually typing a path. This will allow you to switch easily between toolchains without modifying the compilation instructions, and will also reduce the risk of linking a mismatched library. For example, if you need to specify the location of the GSL library, the option you provide to `cmake` might look like `-DGSL_DIR=$EBROOTGSL`. The `EBROOT...` environment variables adhere to the same construction pattern: `EBROOT` followed by the name of the package, for example `EBROOTGCC`.

## BLAS/LAPACK and MKL {#blaslapack_and_mkl}

Please refer to our dedicated page on [BLAS and LAPACK](https://docs.alliancecan.ca/BLAS_and_LAPACK "BLAS and LAPACK"){.wikilink}.

## apt-get and yum {#apt_get_and_yum}

If the software includes instructions to run `apt-get` or `yum`, it is unlikely that you will be able to install it using those instructions. Look for instructions that say \"to build from source\", or contact [support](https://docs.alliancecan.ca/mailto:support@computecanada.ca) for assistance.

## Python, R, and Perl packages {#python_r_and_perl_packages}

[Python](https://docs.alliancecan.ca/Python "Python"){.wikilink}, [R](https://docs.alliancecan.ca/R "R"){.wikilink}, and [Perl](https://docs.alliancecan.ca/Perl "Perl"){.wikilink} are languages with large libraries of extension packages, and package managers that can easily install almost any desired extension in your home directory. See the page for each language to find out if the package you\'re looking for is already available on our systems. If it is not, you should also find detailed guidance there on using that language\'s package manager to install it for yourself.

## Installing binary packages {#installing_binary_packages}

If you install pre-compiled binaries in your home directory they may fail using errors such as `/lib64/libc.so.6: version 'GLIBC_2.18' not found`. Often such binaries can be patched using our `setrpaths.sh` script, using the syntax `setrpaths.sh --path path [--add_origin]` where path refers to the directory where you installed that software. This script will make sure that the binaries use the correct interpreter, and search for the libraries they are dynamically linked to in the correct folder. The option `--add_origin` will also add \$ORIGIN to the RUNPATH. This is sometimes helpful if the library cannot find other libraries in the same folder as itself.

Note:

- Some archive file, such as java (`.jar` files) or [python wheels](https://pythonwheels.com/) (`.whl` files) may contain shared objects that need to be patched. The `setrpaths.sh` script extracts and patches these objects and updates the archive.

## The Alliance software stack {#the_alliance_software_stack}

Almost all software that is used on the new clusters is distributed centrally, using the CVMFS file system. What this means in practice is that this software is not installed under `/usr/bin`, `/usr/include`, and so on, as it would be in a typical Linux distribution, but instead somewhere under `/cvmfs/soft.computecanada.ca`, and is identical on all new clusters.

The core of this [software stack](https://docs.alliancecan.ca/Standard_software_environments "software stack"){.wikilink} is provided by the `gentoo/2023` module, which is loaded by default. This stack, internally managed using the Gentoo package manager, is located at `/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr`. The environment variable `$EBROOTGENTOO` should be used to refer to this path. Under this location you can find all of the common packages typically included with Linux distributions, for instance `make`, `ls`, `cat`, `grep`, and so on. Typically, when you compile some software, the compiler and linker will automatically look for header files and libraries in the right location (via the environment variables `$CPATH` and `$LIBRARY_PATH`, respectively). Some software, however, has been hard-coded to look under `/usr`. If that is the case, the compilation will typically fail, and needs to be explicitly told about `$EBROOTGENTOO`. Sometimes that means adjusting a Makefile, sometimes it needs to be specified in a certain `--with-` flag for the configure script, or a configuration file needs to be edited. If you are not sure how to do this please do not hesitate to ask for help.

Similarly, if a package depends on a library that is provided by a module other than `gentoo`, you may need to provide the location of the header files and libraries of that module. Those other modules also provide an environment variable that starts with EBROOT and ends with the capitalized module name. For example, after you issue the command `module load hdf5`, you can find its installation in `$EBROOTHDF5`, its header files in `$EBROOTHDF5/include`, its library files in `$EBROOTHDF5/lib`, and so on.

If a header file or library that would usually be provided by an RPM or other package manager in a typical Linux distribution is neither present via `gentoo` or via another module, please let us know. Most likely it can be easily added to the existing stack.

Notes:

- all binaries under `/cvmfs/soft.computecanada.ca` use what is called a RUNPATH, which means that the directories for the runtime libraries that these binaries depend on are put inside the binary. That means it is generally `<b>`{=html}not`</b>`{=html} necessary to use `$LD_LIBRARY_PATH`. In fact, `$LD_LIBRARY_PATH` overrides this runpath and you should `<b>`{=html}not`</b>`{=html} set that environment variable to locations such as `/usr/lib64` or `$EBROOTGENTOO/lib64`. Many binaries will no longer work if you attempt this.
- if all else fails you can use `module --force purge` to remove the CVMFS environment. You are then left with a bare-bones AlmaLinux-9 installation without modules. This may help for special situations such as compiling GCC yourself or using custom toolchains such as the [MESA SDK](http://www.astro.wisc.edu/~townsend/static.php?ref=mesasdk). Purging modules would then `<b>`{=html}only`</b>`{=html} be necessary when you compile such software; the modules can be reloaded when running it.

## Compiling on compute nodes {#compiling_on_compute_nodes}

In most situations you can compile on the login nodes. However, if the code needs to be built on a node

- with a GPU, or
- with a Skylake CPU,

then you should start an [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} on a host with the hardware you need, and compile from there.
