---
title: "C++/en"
url: "https://docs.alliancecan.ca/wiki/C%2B%2B/en"
category: "General"
last_modified: "2024-03-20T17:56:39Z"
page_id: 3102
display_title: "C++"
---

`<languages />`{=html}

C++ is a general-purpose, high-level, multi-paradigm programming language created by Bjarne Stroustrup at Bell Labs in 1979 by extending the C programming language. A number of ISO C++ standards have been released corresponding to the years 1998, 2003, 2011, 2014, 2017, and 2020. These are usually referred to as C++98, C++03, C++11, C++14, C++17, and C++20. The ISO C++ committee is currently releasing its standards on a three-year cycle, i.e., after C++20, the next standard will be in 2023.

A definitive, up-to-date, free online wiki reference for C++ (and its C Standard Library subset) is [cppreference.com](http://en.cppreference.com/w/).

It is important to understand that the C language is a distinct language from the C++ language. While a lot of C code can be used as-is in C++ there are times when differences will affect compilation and/or program execution due to differences between C and C++. If you are compiling your C code as C++ code, then when asking for help refer to your code as C++ code as it is being compiled as C++ code.

## Well-defined concurrency and memory models {#well_defined_concurrency_and_memory_models}

Prior to 2011 the ISO C++ standards had no definitions of concurrency and memory models in them, thus, in pre-C++11 compiled code there are no guarantees concerning the ordering of memory reads and writes under concurrency, i.e., such is likely undefined behaviour which the compiler vendor may or may not have documented. **It is therefore preferable to compile concurrent C++ code as C++11 code (or newer).**

Another reason to use the newer compiler and standards possible with C++ code is to take full advantage of optimizations as a result of features such as `constexpr`. With each newer C++ standard, besides additions, there are updates to existing definitions that improve code correctness and/or optimizations that can be applied. Using an old compiler and/or an older standard when newer ones can be used will not be able to take advantage of newer compiler abilities and C++ definitions.

## Compiler support {#compiler_support}

### Language features {#language_features}

Various compilers implement various language features differently. Compiler releases at times only partially implement a specific ISO C++ standard. This can sometimes be frustrating when compiling code with a compiler that does not yet implement a specific language feature. Fortunately there is a wiki page covering virtually all major C++ compilers and [listing the earlier compiler version implementing specific language features](http://en.cppreference.com/w/cpp/compiler_support) at cppreference.com. This page also provides reference links to each compiler\'s web site concerning the details of such.

### Standard library implementation {#standard_library_implementation}

It is important to realize that many C++ compilers under Linux do not actually provide their own implementation of the C++ Standard Library under certain operating systems (especially Linux). Instead these compilers will use one that is normally installed on the system. Typically this implies that libstdc++, which is distributed with GCC, is used.

**NOTE:** While you need not worry about this, this is a reason C++ compilers other than GCC on systems across Compute Canada must be configured by administrators to use a specific version of libstdc++ as several versions of GCC (and therefore libstdc++) are typically installed on a system. If such is set improperly, then there may be issues. This is also a reason why users should **never** hard-code paths to administrator-installed libraries in order to compile software.

The GCC documentation has a section which details [Standard Library components are supported in libstdc++](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html).

## New to C++ or need an update? {#new_to_c_or_need_an_update}

If you are new to C++ or need an update then start by checking out the ISO C++ advocacy site\'s [Get Started](https://isocpp.org/get-started) page \--especially its recommended books. All of these books are excellent.

## Pitfalls

### The `volatile` keyword {#the_volatile_keyword}

The reader should note that `volatile` in C and C++ have very specific meanings, e.g., see [this page](http://en.cppreference.com/w/cpp/language/cv). Needing to use `volatile` in C/C++ code is a rare event. Within high-performance computing code, `volatile` in rare instances might be used to declare a variable in order tell the compiler to not optimize away load/store side-effects from/to that variable. That said, **never** use volatile for synchronization: instead, use C11/C++11 atomic operations and locks, etc. Finally, know `volatile` in other programming languages, e.g., Java, is often different than in C/C++. Typically the latter would be implemented in C/C++ using synchronization constructs.

### Compilers

#### GCC

##### -O3

The GCC compiler\'s -O3 option includes possibly unsafe optimizations for some types of code (e.g., code relying on aliasing). If unsure, compile and optimize code using the -O2 option instead. If you\'ve more time, read the man page (e.g., `man g++`) and unset the appropriate options by searching for \"-O3\" to see which options are turned on and turn off the settings that are not safe.

##### Linking with older previously compiled binaries {#linking_with_older_previously_compiled_binaries}

The transition from GCC version 4.9 to version 5.1 introduced a major change to its ABI. If all source code including all dependent libraries is recompiled using the same version of the compiler then there will be no issues. If different compilers are used, the ABI change may cause linking to fail. The latter is likely to occur if you are linking to precompiled libraries provided in a vendor\'s product. If this occurs, you can use GCC\'s Dual ABI[^1] feature to tell GCC to use the old ABI in order for your application to link properly with those legacy libraries, e.g., you would pass `-D_GLIBCXX_USE_CXX11_ABI=0` to GCC if using GCC v5.1 or higher to link to libraries built using the older ABI.

An example of how the ABI is affected by various GCC command-line options here: [GCC C++ Dual ABI](https://docs.alliancecan.ca/GCC_C++_Dual_ABI "GCC C++ Dual ABI"){.wikilink}.

Generally speaking, the C++ ABI is frequently updated. One should assume each major compiler release might break the C++ ABI enough that older binaries will have trouble linking C++ code. The solution is typically to keep using the same compiler, or, recompile the older binaries from source with the newer compiler. With GCC such options can be controlled, e.g., see [this page](https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html) on `-fabi-version`.

#### Intel

Intel C/C++ compilers may default to using possibly unsafe optimizations for floating-point operations. Users using the Intel compilers should read the Intel man pages (e.g., `man icpc`) and are recommended to use one of two options, `-fp-model precise` or `-fp-model source`, for ANSI/ISO/IEEE standards-compliant floating-point support. For more details, read this Intel slideshow called, [Floating-point control in the Intel compiler and libraries](https://software.intel.com/sites/default/files/article/326703/fp-control-2012-08.pdf).

## References

<references/>

[^1]: Free Software Foundation. The GNU C++ Library, Chapter 3. <https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html>
