---
title: "Structure/en"
url: "https://docs.alliancecan.ca/wiki/Structure/en"
category: "General"
last_modified: "2022-01-21T18:35:08Z"
page_id: 8832
display_title: "Structure"
---

`<languages />`{=html}

## Description

Structure[^1] is a free software package for using multilocus genotype data to investigate population structure. Its uses include inferring the presence of distinct populations, assigning individuals to populations, studying hybrid zones, identifying migrants and admixed individuals, and estimating population allele frequencies in situations where many individuals are migrants or admixed. It can be applied to most of the commonly-used genetic markers, including SNPS, microsatellites, RFLPs and AFLPs. [^2] [^3]

## Installed versions {#installed_versions}

At the time of writing Structure v2.3.4 has been installed (module: `structure/2.3.4`). Please look on page [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink} for module `structure` for an up-to-date list of installed versions.

## Usage

When starting *structure* without any command-line options, it expects to find the following three files in the current working directory:

- `mainparams` - Example mainparams[^4]
- `extraparams` - Example extraparams[^5]
- and a data file. The name of the data file can be either be set the *INFILE* parameter in `mainparams` or supplied via the command line (-i).

Please refer to §7 \"Running *structure* from the command line\" in the Structure Documentation[^6] for a complete description of the available parameters and file-formats. There are also several command-line options to supply alternative filenames or override options. e.g.: -m (mainparams), -e (extraparams), -i (input file), -o (output file). See section §7.4 in the Structure Documentation[^7].

Here is an example submission script:

## Running Structure in parallel: StrAuto & StructureHarvester {#running_structure_in_parallel_strauto_structureharvester}

The Structure in itself is not able to run in parallel, however as population structure inference using the Evanno-method [^8] involves multiple structure-runs, the tools StrAuto [^9] [^10] [^11] and StructureHarvester [^12] [^13] have been developed to automate the process of setting up such runs and aggregating their results respectively.

### Practical considerations {#practical_considerations}

There is an example for running StrAuto jobs on HPC clusters using the Slurm Workload manager outlined in chapter 8 the StrAuto User Guide[^14]. The example works best when the total number of Structure runs is a multiple of the number of requested tasks, as otherwise some of the allocated CPUs will sit idle while the last Structure runs are being executed. This may lead to a significant waste of computing resources.

Moreover, the requested maximum job time needs to be sufficiently large to accommodate multiple subsequent Structure runs. While this can be a sensible choice for relatively short runs, jobs with long walltimes will typically have to wait longer until they are dispatched due to [job scheduling policies](https://docs.alliancecan.ca/Job_scheduling_policies#Time_limits "job scheduling policies"){.wikilink}. Therefore in cases where an individual Structure run takes more than just a few hours to finish, we recommend submitting each Structure run as an individual job to Slurm.

### Running a set of longer Structure runs {#running_a_set_of_longer_structure_runs}

The script `create_strauto_slurm_scripts.py` shown at the bottom of this section is designed to help you run Structure[^15] jobs on Compute Canada HPC clusters that use the Slurm Workload manager. It requires Structure, StrAuto[^16], and StructureHarvester[^17].

Usage:

- Place the `create_strauto_slurm_scripts.py` script shown below into a directory along with:

:\* `strauto_1.py` and `input.py` from StrAuto[^18].

:\* `structureHarvester.py` and `harvesterCore.py` from StructureHarvester[^19].

:\* The file with the Structure dataset e.g. `my_dataset.str`.

- Make `create_strauto_slurm_scripts.py` executable with:

:\* You should now have the following files:

- Edit the settings in `input.py` as described in the StrAuto User Manual[^20].

:\* Make sure to set the option `parallel = True` (question 23).

- Adjust the parameters in this file (lines 65-70):

:\* Set `max_jobtime` to a duration where you can be reasonably sure that no individual Structure run takes longer than this.

:\* In cases where a user can submit under multiple Slurm-accounts `slurm_account` can be used to specify under which account to submit.

:\* To avoid overloading the Scheduler, the submission helper script delays each submission by a time defined by `submit_delay`.

- Run the following commands:

<!-- -->

- Submit the jobs with:

<!-- -->

- After all jobs have completed, run `bash post_strauto.sh` to aggregate the results and run StructureHarvester.

#### The \'create_strauto_slurm_scripts.py\' script {#the_create_strauto_slurm_scripts.py_script}

```{=mediawiki}
{{File
  |name=create_strauto_slurm_scripts.py
  |lang="python"
  |contents=
#!/usr/bin/env python
''' 
create_strauto_slurm_scripts.py
===============================

This script is designed to be a supplement to running Structure[1] jobs on
Compute Canada HPC clusters that are using the Slurm Workload manager.
It is meant to be used with Structure, StrAuto[2] and StructureHarvester[3].

Usage:
******

* Place this script 'create_strauto_slurm_scripts.py' into a directory along with
    * 'strauto_1.py' and 'input.py' from StrAuto[2].
    * 'structureHarvester.py' and 'harvesterCore.py' from StructureHarvester[3]
    * The file with the Structure dataset e.g. 'sim.str'.
* Edit the settings in 'input.py' as described in the StrAuto User Manual.
    * Make sure to set the option 'parallel = True' (question 23).
* Adjust the parameters in this file (ca. lines 65-70):
    * Set 'max_jobtime' to a duration where you can be reasonably sure that no
      individual Structure run takes longer than this.
    * In cases where a user can submit under multiple Slurm-accounts
      'slurm_account' can be used to specify under which account to submit.
    * To avoid overloading the Scheduler, the submission helper script delays
      each submission by a time defined by 'submit_delay'.
* Run the following commands:
    * ./strauto_1.py
    * ./create_strauto_slurm_scripts.py
* Submit the jobs with:
    * ./submit_strauto_jobs.sh
* After all jobs have completed, run './post_strauto.sh' to aggregate the
  results and run StructureHarvester.

[1] Structure: http://web.stanford.edu/group/pritchardlab/structure.html
[2] StrAuto:   http://www.crypticlineage.net/pages/software.html
[3] Harvester: http://alumni.soe.ucsc.edu/~dearl/software/structureHarvester/
               https://github.com/dentearl/structureHarvester

LICENSE:
========
This program is licensed under the conditions of the "MIT License".

Copyright 2017-2020 Oliver Stueker <ostueker(a)ace-net.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''
from __future__ import print_function
import re, os, sys
##############################################################################
### Please adjust the following parameters:
##############################################################################
max_jobtime = '1-0:00:0' # format: d-hh:mm:ss
slurm_account = None     # e.g.: slurm_account='def-somegroup'
submit_delay = '0.5s'    # pause this long between job submissions
job_prefix = None        # (optional) e.g.: job_prefix='strauto'

##############################################################################
### Don't make any changes below this line:
##############################################################################
template = '''#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time={MAXTIME:}              # d-hh:mm:ss
#SBATCH --job-name={JOBNAME:}
#SBATCH --chdir={DIRECTORY:}
#SBATCH --output=%x_%j_slurm.out
{SLURM_ACCT:}
module load structure/2.3.4

{STRUCTURE_COMMAND}
'''

post_template = '''#!/bin/bash
mv {KLIST:} results_f/
mkdir harvester_input
cp results_f/k*/*_f harvester_input
echo "The structure runs have finished."

# Run structureHarvester.py
./structureHarvester.py --dir harvester_input --out harvester --evanno --clumpp
echo 'structureHarvester run has finished.'

# Clean up Harvester input files.
zip {DATASET:}_Harvester_Upload.zip harvester_input/*
mv {DATASET:}_Harvester_Upload.zip harvester/
rm -rf harvester_input
'''

if __name__ == '__main__':
    if not os.path.exists('structureCommands'):
        print('ERROR: File "structureCommands" does not exist!')
        print('You need to run "./strauto_1.py" before running "{}".'
              .format(os.path.basename(sys.argv[0])))
        sys.exit(1)
    # initialize some variables
    scripts_dir = 'slurm_scripts'
    job_directory = os.getcwd()
    if slurm_account:
        slurm_account = '#SBATCH --account='+slurm_account
    else:
        slurm_account = ''
    re_jobname = re.compile(r'-o (k\d+)/((.+)_\1_run\d+) ')
    job_scripts = []
    klist = set()
    
    with open('structureCommands', 'r') as structureCommands:
        print("Creating SLURM job scripts...")
        for line in structureCommands:
            match = re_jobname.search(line)
            if match:
                kn = match.group(1)
                jobname = match.group(2)
                dataset = match.group(3)

                klist.add(kn)
                if job_prefix:
                    jobname = '_'.join([job_prefix, jobname])

                line = line.replace('./structure', 'structure')

                job_script_content = template.format(
                        SLURM_ACCT=slurm_account,
                        MAXTIME=max_jobtime,
                        JOBNAME=jobname,
                        DIRECTORY=job_directory,
                        STRUCTURE_COMMAND=line,
                    )

                if not ( os.path.exists(scripts_dir) and
                         os.path.isdir(scripts_dir) ):
                    os.mkdir(scripts_dir)

                job_script_name = os.path.join( scripts_dir,
                                    'slurm_job_{:}.sh'.format(jobname))
                with open(job_script_name, 'w') as slurm_script:
                    slurm_script.write(job_script_content)
                job_scripts.append(job_script_name)
        print("created {:} job scripts.\n".format(len(job_scripts)))

    print("Creating directories...", end='')
    directories = ['results_f', 'log', 'harvester']
    for kn in sorted(klist):
        directories.append(kn)
        directories.append(os.path.join('log', kn))
    print('done!', end='\n\n')

    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)

    print('Creating submission helper script...')
    helper_script_name='submit_strauto_jobs.sh'
    with open(helper_script_name, 'w') as helper_script:
        helper_script.write('#!/bin/bash\n')
        for job_script in job_scripts:
            helper_script.write('sleep 0.5s ; sbatch {:}\n'.format(job_script))
    print('created helper script: "{:}"\n'.format(helper_script_name))

    print('Creating postprocessing script...')
    post_script_name='post_strauto.sh'
    with open(post_script_name, 'w') as pst_script:
        pst_script.write(post_template.format(KLIST=' '.join(sorted(klist)),
                                              DATASET=dataset))
    print('created post-script: "{:}"\n'.format(post_script_name))

}}
```
## References

<references />

[^1]: Structure Homepage: <http://web.stanford.edu/group/pritchardlab/structure.html>

[^2]: J.K. Pritchard, M. Stephens, and P. Donnelly. Inference of population structure using multilocus genotype data. Genetics, 155:945--959, 2000. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1461096/>

[^3]: M.J. Hubisz, D. Falush, M. Stephens, and J.K. Pritchard. Inferring weak population structure with the assistance of sample group information. Molecular Ecology Resources, 9(5):1322--1332, 2009. doi: <https://doi.org/10.1111/j.1755-0998.2009.02591.x>

[^4]: Example [mainparams](http://web.stanford.edu/group/pritchardlab/software/mainparams) on the Structure Hompage

[^5]: Example [extraparams](http://web.stanford.edu/group/pritchardlab/software/extraparams) on the Structure Hompage

[^6]: Structure Documentation for version 2.3.4 [PDF](http://web.stanford.edu/group/pritchardlab/structure_software/release_versions/v2.3.4/structure_doc.pdf)

[^7]:

[^8]: G. Evanno, S. Regnaut, and J. Goudet. Detecting the number of clusters of individuals using the software structure: a simulation study. Molecular Ecology, 14:2611--2620, 2005. DOI: <https://doi.org/10.1111/j.1365-294X.2005.02553.x>

[^9]: StrAuto Homepage: <https://vc.popgen.org/software/strauto/>

[^10]: StrAuto User Guide: <https://vc.popgen.org/software/strauto/strauto_doc.pdf>

[^11]: Chhatre, VE & Emerson KJ. StrAuto: Automation and parallelization of STRUCTURE analysis. BMC Bioinformatics (2017) 18:192. doi: <http://dx.doi.org/10.1186/s12859-017-1593-0>

[^12]: StructureHarvester Homepage: <http://alumni.soe.ucsc.edu/~dearl/software/structureHarvester/> ; GitHub: <https://github.com/dentearl/structureHarvester>

[^13]: Earl, Dent A. and vonHoldt, Bridgett M. STRUCTURE HARVESTER: a website and program for visualizing STRUCTURE output and implementing the Evanno method. Conservation Genetics Resources (2011) DOI: 10.1007/s12686-011-9548-7

[^14]:

[^15]:

[^16]:

[^17]:

[^18]:

[^19]:

[^20]:
