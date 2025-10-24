---
title: "Using Nix"
url: "https://docs.alliancecan.ca/wiki/Using_Nix"
category: "General"
last_modified: "2022-07-25T18:23:45Z"
page_id: 10136
display_title: "Using Nix"
---

# Overview

[Nix](https://nixos.org/nix/) is a software building and composition system that allows users to manage their own persistent software environments. It is only available on SHARCNET systems (i.e., graham and legacy).

- Supports one-off, per-project, and per-user usage of compositions
- Compositions can be built, installed, upgraded, downgraded, and removed as a user
- Operations either succeed or fail leaving everything intact (operations are atomic).
- Extremely easy to add and share compositions

Currently nix is building software in a generic manner (e.g., without AVX2 or AVX512 vector instructions support), so module loaded software should be preferred for longer running simulations when it exists.

**NOTE:** The message `failed to lock thread to CPU XX` is a harmless warning that can be ignored.

## Enabling and disabling the nix environment {#enabling_and_disabling_the_nix_environment}

The user's current nix environment is enabled by loading the nix module. This creates some *.nix\** files and sets some environment variables.

``` bash
[name@cluster:~]$ module load nix
```

It is disabled by unloading the nix module. This unsets the environment variables but leaves the *.nix\** files alone.

``` bash
[name@cluster:~]$ module unload nix
```

## Completely resetting the nix environment {#completely_resetting_the_nix_environment}

Most per-user operations can be undone with the `--rollback` option (i.e., `nix-env --rollback` or `nix-channel --rollback`). Sometimes it is useful to entirely reset nix though. This is done by unloading the module, erasing all user related nix files, and then reloading the module file.

``` bash
[name@cluster:~]$ module unload nix
[name@cluster:~]$ rm -fr ~/.nix-profile ~/.nix-defexpr ~/.nix-channels ~/.config/nixpkgs
[name@cluster:~]$ rm -fr /nix/var/nix/profiles/per-user/$USER /nix/var/nix/gcroots/per-user/$USER
[name@cluster:~]$ module load nix
```

# Existing compositions {#existing_compositions}

The `nix search` command can be used to locate already available compositions

``` bash
[user@cluster:~]$ nix search git
...
* nixpkgs.git (git-minimal-2.19.3)
  Distributed version control system
...
```

Pro tips include

- you need to specify `-u` after upgrading your channel (this will take awhile)
- the search string is actually a regular expression and multiple ones are ANDed together

Often our usage of a composition is either a one-off, a per-project, or an all the time situations. Nix supports all three of these cases.

## One offs {#one_offs}

If you just want to use a composition once, the easiest was is to use the `nix run` command. This command will start a shell in which `PATH` has been extended to include the specified composition

``` bash
[user@cluster:~]$ nix run nixpkg.git
[user@cluster:~]$ git
[user@cluster:~]$ exit
```

Note that this does not protect the composition from being garbage collected overnight (e.g., the composition is only guaranteed to be around temporarily for your use until sometime in the wee-morning hours). Pro tips include

- you can specify more than one composition in the same `nix run` command
- you can specify a command instead of a shell with `-c <cmd> <args> ...`

## Per-project {#per_project}

If you want to use a program for a specific project, the easiest way is with the `nix build` command. This command will create a symbolic link (by default named `result`) from which you can access the programs *bin* directory to run it.

``` bash
[user@cluster:~]$ nix build nixpkgs.git
[user@cluster:~]$ ./result/bin/git
```

Note that (currently) the composition will only be protected from overnight garbage collection if you output the symlink into your *home* directory and do not rename or move it. Pro tips include

- you can specify the output symlink name with the `-o <name>` option
- add the *bin* directory to your `PATH` to not have to type it in every time

## Per-user {#per_user}

Loading the `nix` module adds the per-user common *\~/.nix-profile/bin* directory to your `PATH`. You can add and remove compositions from this directory with the `nix-env` command

``` bash
[user@cluster:~]$ nix-env --install --attr nixpkgs.git
[user@cluster:~]$ nix-env --query
git-minimal-2.19.3
```

``` bash
[user@cluster:~]$ nix-env --uninstall git-minimal
uninstalling 'git-minimal-2.19.3'
[user@cluster:~]$ nix-env --query
```

Each command actually creates a new version, so all prior versions remain and can be used

``` bash
[user@cluster:~]$ nix-env --list-generations
   1   2020-07-29 13:10:03
   2   2020-07-29 13:11:52   (current)
[user@cluster:~]$ nix-env --switch-generation 1
[user@cluster:~]$ nix-env --query
git-minimal-2.19.3
[user@cluster:~]$ nix-env --switch-generation 2
[user@cluster:~]$ nix-env --query
```

Pro tips include

- `nix-env --rollback` moves up one generation
- `nix-env --delete-generations <time>` deletes environments older than `<time>` (e.g., `30d`)
- see our [nix-env page](https://docs.alliancecan.ca/Using_Nix:_nix-env "nix-env page"){.wikilink} for a much more in-depth discussion of using `nix-env`

# Creating compositions {#creating_compositions}

Often we require our own unique composition. A basic example would be to bundle all the binaries from multiple existing compositions in a common *bin* directory (e.g., `make`, `gcc`, and `ld` to build a simple C program). A more complex example would be to bundle python with a set of python libraries by wrapping the python executables with shell scripts to set `PYTHON_PATH` for the python libraries before running the real python binaries.

All of these have a common format. You write a nix expression in a `.nix` file that composes together existing compositions and then you tell the above commands to use that with the `-f <nix file>` option. For example, say the file `python.nix` has an expression for a python environment in it, you can create a per-project *bin* directory with

``` bash
[user@cluster:~]$ nix build -f python.nix -o python
[user@cluster:~]$ ./python/bin/python
```

The nix expression you put in the file generally

- does `with import <nixpkgs> {}` to bring the set of nixpkgs into scope
- calls an existing composition functions with a list of space-separated components to include

The template for doing the second these follows below as it differs slightly across the various eco-systems.

A pro tip is

- there are many [languages and framework supported](https://nixos.org/nixpkgs/manual/#chap-language-support) but only a few described here, send us an email if you would like a missing supported one added here

## Generic

Nixpkgs provides a `buildEnv` function that does a basic composition of compositions (by combining their *bin*, *lib*, etc. directories). The list of packages are the same as used before minus the leading `nixpkgs` prefix as it was imported (e.g., `git` instead of `nixpkgs.git`).

``` nix
with import <nixpkgs> {};
buildEnv {
  name = "my environment";
  paths = [
    ... list of compositions ...
  ];
}
```

## Python

Nixpkgs provides the following python related attributes

- `python<major><minor>` - a composition providing the given python
- `python<major><minor>.pkgs` - the set of python compositions using the given python
- `python<major><minor>.withPackages` - wraps python with `PYTHON_PATH` set to a given set of python packages

We can use the former directly to use the programs provided by python compositions

``` bash
[user@cluster:~]$ nix run python36.pkgs.spambayes
[user@cluster:~]$ sb_filter.py --help
[user@cluster:~]$ exit
```

and the later in a `.nix` file to create a python composition that enables a given set of libraries (e.g., a `python` command we can run and access the given set of python packages from)

``` nix
with import <nixpkgs> { };
python.withPackages (packages:
  with packages; [
    ... list of python packages ...
  ]
)
```

Some pro tips are

- the aliases `python` and `python<major>` given default `python<major><minor>` versions
- the aliases `pythonPackages<major><minor>` are short for `python<major><minor>.pkgs` (including default version variants)
- the function `python<major><minor>.pkgs.buildPythonPackage` can be used to build your own python packages

## R

Nixpkgs provides the following R related attributes

- `R` - a composition providing R
- `rstudio` - a composition providing RStudio
- `rPackages` - the set of R packages
- `rWrapper` - a composition that wraps R with `R_LIBS` set to a minimal set of R packages
- `rstudioWrapper` - a composition that wrapped RStudio with `R_LIBS` set to a minimal set of R packages

We can use `rPackages` directly to examine the content of R packages

``` bash
[user@cluster:~]$ nix build rPackages.exams -o exams
[user@cluster:~]$ cat exams/library/exams/NEWS
[user@cluster:~]$ exit
```

and the latter two can be overridden in a `.nix` file to create R and RStudio wrappers to create a composition enabling a given set of R libraries (e.g., a `R` or `rstudio` command we can run and access the given set of R packages from)

``` nix
with import <nixpkgs> { };
rWrapper.override {
  packages = with rPackages; [
    ... list of R packages ...
  ];
}
```

A pro tips is

- the function `rPackages.buildRPackage` can be used to build your own R packages

## Haskell

Nixpkgs provides the following haskell related attributes

- `haskell.compiler.ghc<major><minor><patch>` - composition providing the given ghc
- `haskell.packages.ghc<major><minor><patch>` - the set of haskell packages compiled by the given ghc
- `haskell.packages.ghc<major><minor><patch>.ghc.withPackages` - composition wrapping ghc to enable the given packages
- `haskell.packages.ghc<major><minor><patch>.ghc.withHoogle` - composition wrapping ghc to enable the given packages with hoogle and documentation indices

We can use the first directly to use programs provided by haskell packages

``` bash
[user@cluster:~]$ nix run haskell.packages.ghc864.pandoc
[user@cluster:~]$ pandoc --help
```

and the last two in a `.nix` file create a ghc environment to enable a given set of haskell package (e.g., a `ghci` we can run and access the given set of packages from)

    with import &lt;nixpkgs&gt; { };
    haskell.packages.ghc864.ghc.withPackages (packages:
      with packages; [
        ... list of Haskell packages ...
      ];
    }

Some pro tips are

- the alias `haskellPackages` gives a default `haskell.packages.ghc<major><minor><patch>`
- the attributes in `haskell.lib` contains a variety of useful attributes for tweaking haskell packages (e.g., enabling profiling, etc.)
- the upstream maintainer has a useful [youtube video](https://www.youtube.com/watch?v=KLhkAEk8I20) on how to fix broken haskell packages

## Emacs

Nixpkgs provides the following emacs related attributes (append a `Ng` suffix for older versions of nixpkgs, e.g., `emacs25Ng` and `emacs25PackagesNg`)

- `emacs<major><minor>` - a composition providing the given emacs editor
- `emacs<major><minor>Packages` - the set of emacs packages for the given emacs editor
- `emacs<major><minor>Packages.emacsWithPackages` - composition wrapping emacs to enable the given packages

We can use the second directly examine the content of packages

``` bash
[user@cluster:~]$ nix build nixpkgs.emacs25Packages.magit -o magit
[user@cluster:~]$ cat magit/share/emacs/site-lisp/elpa/magit*/AUTHORS.md
[user@cluster:~]$ exit
```

and the last one in a `.nix` file create a composition giving emacs with the given set of packages enabled

    with import &lt;nixpkgs&gt; { };
    emacs25Packages.emacsWithPackages (packages:
      with packages; [
        ... list of emacs packages ...
      ];
    }

Some pro tips are

- the aliases `emacs` and `emacsPackages` give a default `emacs<major><minor>` and `emacsPackages<major><minor>` version
- the alias `emacs<major><minor>WithPackages` are short for `emacs<major><minor>Packages.emacsWithPackages` (including default version variants)
