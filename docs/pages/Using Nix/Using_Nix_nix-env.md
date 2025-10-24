---
title: "Using Nix: nix-env"
url: "https://docs.alliancecan.ca/wiki/Using_Nix:_nix-env"
category: "Using Nix"
last_modified: "2020-08-12T22:52:36Z"
page_id: 14176
display_title: "Using Nix: nix-env"
---

This page details using the legacy `nix-env` command to manage a per-user environment. For an overview of Nix, start with our [using nix page](https://docs.alliancecan.ca/Using_Nix "using nix page"){.wikilink}.

# Querying, installing and removing compositions {#querying_installing_and_removing_compositions}

The `nix-env` command is used to manage your per-user Nix environment. It is actually a legacy command that has not yet been replaced by a newer `nix <command>` command.

## What do I have installed and what can I install {#what_do_i_have_installed_and_what_can_i_install}

Lets first see what we currently have installed.

``` bash
[name@cluster:~]$ nix-env --query
```

Now let's see what is available. We request the attribute paths (unambiguous way of specifying an existing composition) and the descriptions too (cursor to the right to see them). This takes a bit of time as it visits a lot of small files. Especially over NFS it can be a good idea to pipe it to a file and then grep that in the future.

``` bash
[name@cluster:~]$ nix-env --query --available --attr-path --description
```

The newer `nix search` command is often a better way to locate compositions as it saves a cache so subsequent invocations are quite fast.

## Installing compositions {#installing_compositions}

Let's say that we need a newer version of git than provided by default. First lets check what our OS comes with.

``` bash
[name@cluster:~]$ git --version
[name@cluster:~]$ which git
```

Let's tell Nix to install its version in our environment.

``` bash
[name@cluster:~]$ nix-env --install --attr nixpkgs.git
[name@cluster:~]$ nix-env --query
```

Let's checkout what we have now (it may be necessary to tell bash to to forget remembered executable locations with `hash -r` so it notices the new one).

``` bash
[name@cluster:~]$ git --version
[name@cluster:~]$ which git
```

## Removing compositions {#removing_compositions}

For completeness, lets add in the other usual version-control suspects.

``` bash
[name@cluster:~]$ nix-env --install --attr nixpkgs.subversion nixpkgs.mercurial
[name@cluster:~]$ nix-env --query
```

Actually, we probably don't really want subversion any more. Let's remove that.

``` bash
[name@cluster:~]$ nix-env --uninstall subversion
[name@cluster:~]$ nix-env --query
```

# Environments

Nix keeps referring to user environments. Each time we install or remove compositions we create a new environment based off of the previous environment.

## Switching between previous environments {#switching_between_previous_environments}

This means the previous environments still exist and we can switch back to them at any point. Let's say we changed our mind and want subversion back. It's trivial to restore the previous environment.

``` bash
[name@cluster:~]$ nix-env --rollback
[name@cluster:~]$ nix-env --query
```

Of course we may want to do more than just move to the previous environment. We can get a list of all our environments so far and then jump directly to whatever one we want. Let's undo the rollback.

``` bash
[name@cluster:~]$ nix-env --list-generations
[name@cluster:~]$ nix-env --switch-generation 4
[name@cluster:~]$ nix-env --query
```

## Operations are atomic {#operations_are_atomic}

Due to the atomic property of Nix environments, we can't be left halfway through installing/updating compositions. They either succeed and create us a new environment or leave us with the previous one intact.

Let's go back to the start when we just had Nix itself and install the one true GNU distributed version control system tla. Don't let it complete though. Hit it with `CTRL+c` partway through.

``` bash
[name@cluster:~]$ nix-env --switch-generation 1
[name@cluster:~]$ nix-env --install --attr nixpkgs.tla
CTRL+c
```

Nothing bad happens. The operation didn't complete so it has no effect on the environment whatsoever.

``` bash
[name@cluster:~]$ nix-env --query
[name@cluster:~]$ nix-env --list-generations
```

## Nix only does things once {#nix_only_does_things_once}

The install and remove commands take the current environment and create a new environment with the changes. This works regardless of which environment we are currently in. Let's create a new environment from our original environment by just adding git and mercurial.

``` bash
[name@cluster:~]$ nix-env --list-generations
[name@cluster:~]$ nix-env --install --attr nixpkgs.git nixpkgs.mercurial
[name@cluster:~]$ nix-env --list-generations
```

Notice how much much faster it was to install git and mercurial the second time? That is because the software already existed in the local Nix store from the previous installs so Nix just reused it.

## Garbage collection {#garbage_collection}

Nix periodically goes through and removes any software not accessible from any existing environments. This means we have to explicitly delete environments we don't want anymore so Nix is able to reclaim the space. We can delete specific environments or any sufficiently old.

``` bash
[name@cluster:~]$ nix-env --delete-generations 30d
```
