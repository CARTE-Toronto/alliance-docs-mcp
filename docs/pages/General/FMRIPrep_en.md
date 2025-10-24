---
title: "FMRIPrep/en"
url: "https://docs.alliancecan.ca/wiki/FMRIPrep/en"
category: "General"
last_modified: "2023-11-02T19:27:36Z"
page_id: 24490
display_title: "FMRIPrep"
---

`<languages/>`{=html}

# Load fMRIPrep {#load_fmriprep}

[fMRIPrep](https://fmriprep.org/en/stable/) is an [NiPreps](https://www.nipreps.org) app for preprocessing MRI data in [BIDS](https://bids.neuroimaging.io) format. To use this on Alliance resources, first

    module load apptainer fmriprep

# A note about Apptainer {#a_note_about_apptainer}

When you use fMRIPrep on the command line it\'s using Apptainer under the hood to invoke a container with fMRIPrep set up inside it. Command line options will be passed along to the fMRIPrep command in the container but you\'ll also want to affect the Apptainer context. To do this you\'ll have to use [Apptainer environment variables](https://apptainer.org/docs/user/main/appendix.html). See the example a few paragraphs down.

# Setup and download TemplateFlow {#setup_and_download_templateflow}

Further, fMRIPrep within the container will try to download [TemplateFlow](https://www.templateflow.org) templates but will fail. We have to download this data ourselves in advance

    module load python git-annex
    pip3 install datalad
    datalad install -r ///templateflow

Load Python and git-annex, install DataLad (you may want to do this in a virtualenv), and then install the TemplateFlow metadataset. This will be downloaded into a shared directory of the default project you\'re a part of. The directory should look something like `/lustre03/project/GROUPNAME/shared/templateflow`. You\'ll have to change into this directory and download actual template subsets like

    cd /lustre03/project/GROUPNAME/shared/templateflow
    datalad get -r tpl-MNI152NLin2009cAsym tpl-OASIS30ANTs 

Do the same for all templates you want to make available but the above is a good start. **NOTE** that these templates may take a while to download and that you only have to do these DataLad steps **once** and they will be available to you and the rest of your project group until deleted. See [accessing the TemplatFlow archive](https://www.templateflow.org/usage/archive/) for more information.

# Define Apptainer environment variables {#define_apptainer_environment_variables}

Now we can set up our Apptainer and fMRIPrep environment variables

    export APPTAINERENV_TEMPLATEFLOW_HOME=/lustre03/project/GROUPNAME/shared/templateflow
    export APPTAINER_BIND=/path/to/input,/path/to/output,/path/to/output/logs,$APPTAINERENV_TEMPLATEFLOW_HOME

With `APPTAINERENV_TEMPLATEFLOW_HOME` we tell the fMRIPrep app where to find the TemplateFlow templates. With `APPTAINER_BIND` we tell Apptainer where the input, output, and logs will be so that it can mount those directories and make them available to the fMRIprep app within the container. **NOTE** that fMRIPrep does not accept very long paths, so keep these directory and filenames short.

# Run like they\'re chasing you {#run_like_theyre_chasing_you}

Finally, we can run

    fmriprep /path/to/input /path/to/output participant --work-dir /path/to/output

And all should be well in the world. Except \...

# It\'s a bit more complicated with FreeSurfer {#its_a_bit_more_complicated_with_freesurfer}

If you want to run fMRIPrep using FreeSurfer, you\'ll have to [register with FreeSurfer](https://surfer.nmr.mgh.harvard.edu/registration.html), download the license file, copy it to one of the directories in `APPTAINER_BIND` and use the [`--fs-license-file` option](https://fmriprep.org/en/20.2.0/usage.html?highlight=freesurfer#Specific%20options%20for%20FreeSurfer%20preprocessing). This is left as an exercise to the reader \... You can do it!

# Thanks

Thanks to Pierre Rioux for the voluminous input on the instructions on this page!
