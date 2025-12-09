---
title: "Migration to the new standard environment/en"
url: "https://docs.alliancecan.ca/wiki/Migration_to_the_new_standard_environment/en"
category: "General"
last_modified: "2024-02-14T15:21:35Z"
page_id: 16321
display_title: "Migration to the new standard environment"
---

`<languages />`{=html}

# What are the differences between `StdEnv/2023` and the earlier environments? {#what_are_the_differences_between_stdenv2023_and_the_earlier_environments}

The differences are discussed in [Standard software environments](https://docs.alliancecan.ca/Standard_software_environments "Standard software environments"){.wikilink}.

# Can I change my default standard environment? {#can_i_change_my_default_standard_environment}

After 2024 April 1, **`StdEnv/2023` will be the default environment for all clusters.** However, you can specify your own default environment at any time by modifying the `$HOME/.modulerc` file. For example, running the following command will set your default environment to `StdEnv/2020`:

You must log out and log in again for this change to take effect.

# Do I need to reinstall/recompile my code when the `StdEnv` changes? {#do_i_need_to_reinstallrecompile_my_code_when_the_stdenv_changes}

Yes. If you compile your own code, or have installed R or Python packages, you should recompile your code or reinstall the packages you need with the newest version of the standard environment.

# How can I use an earlier environment? {#how_can_i_use_an_earlier_environment}

If you have an existing workflow and want to continue to use the same software versions you are using now, simply add

` module load StdEnv/2020`

to your job scripts before loading any other modules.

# Will the earlier environments be removed? {#will_the_earlier_environments_be_removed}

The earlier environments and any software dependent on them will remain available, `<b>`{=html}but versions 2016.4 and 2018.3 are no longer supported`</b>`{=html}, and we recommend not using them. Our staff will only install software in the new environment 2023.

# Can I mix modules from different environments? {#can_i_mix_modules_from_different_environments}

No, you should use a single environment for a given job - different jobs can use different standard environments by explicitly loading one or the other at the job\'s beginning but within a single job you should only use a single environment. The results of trying to mix different environments are unpredictable but in general will lead to errors of one kind or another.

# Which environment should I use? {#which_environment_should_i_use}

If you are starting a new project, or if you want to use a newer version of an application, you should use `StdEnv/2023` by adding

` module load StdEnv/2023`

to your job scripts. This command does not need to be deleted to use `StdEnv/2023` after April 1.

# Can I keep using an older environment by loading modules in my `.bashrc`? {#can_i_keep_using_an_older_environment_by_loading_modules_in_my_.bashrc}

Loading modules in your `.bashrc` is **not recommended**. Instead, explicitly load modules in your job scripts.

# I don\'t use the HPC clusters but cloud resources only. Do I need to worry about this? {#i_dont_use_the_hpc_clusters_but_cloud_resources_only._do_i_need_to_worry_about_this}

No, this change will only affect the [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink} accessed by [ using environment modules](https://docs.alliancecan.ca/Using_modules " using environment modules"){.wikilink}.

# I can no longer load a module that I previously used {#i_can_no_longer_load_a_module_that_i_previously_used}

More recent versions of most applications are installed in the new environment. To see the available versions, run the `module avail` command. For example,

shows several versions of the GCC compilers, which may be different from those in earlier environments.
