---
title: "ACPYPE"
url: "https://docs.alliancecan.ca/wiki/ACPYPE"
category: "General"
last_modified: "2025-02-06T19:32:23Z"
page_id: 27862
display_title: "ACPYPE"
---

## General

[ACPYPE: AnteChamber PYthon Parser interfacE](https://alanwilter.github.io/acpype/) (pronounced as ace + pipe), is a tool written in Python to use Antechamber to generate topologies for chemical compounds and to interface with others python applications like CCPN and ARIA.

It will generate topologies for CNS/XPLOR, [GROMACS](https://docs.alliancecan.ca/GROMACS "GROMACS"){.wikilink}, CHARMM and [AMBER](https://docs.alliancecan.ca/AMBER "AMBER"){.wikilink}, that are based on General Amber Force Field (GAFF) and should be used only with compatible forcefields like AMBER and its variants.

We provide [Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Python wheels"){.wikilink} for ACPYPE for [StdEnv/2020 and StdEnv/2023](https://docs.alliancecan.ca/Standard_software_environments "StdEnv/2020 and StdEnv/2023"){.wikilink} in our wheelhouse that you should install into a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.

Please note that you need to load the `openbabel` module before installing ACPYPE and anytime you want to use it.

## Creating a virtual environment for ACPYPE {#creating_a_virtual_environment_for_acpype}

Now the `acpype` command can be used:

`_SMILES_string_ [-c _string_] [-n _int_] [-m _int_] [-a _string_] [-f] etc. or`\
`   acpype -p _prmtop_ -x _inpcrd_ [-d  -w]`

`   output: assuming 'root' is the basename of either the top input file,`\
`           the 3-letter residue name or user defined (-b option)`\
`   root_bcc_gaff.mol2:  final mol2 file with 'bcc' charges and 'gaff' atom type`\
`   root_AC.inpcrd    :  coord file for AMBER`\
`   root_AC.prmtop    :  topology and parameter file for AMBER`\
`   root_AC.lib       :  residue library file for AMBER`\
`   root_AC.frcmod    :  modified force field parameters`\
`   root_GMX.gro      :  coord file for GROMACS`\
`   root_GMX.top      :  topology file for GROMACS`\
`   root_GMX.itp      :  molecule unit topology and parameter file for GROMACS`\
`   root_GMX_OPLS.itp :  OPLS/AA mol unit topol & par file for GROMACS (experimental!)`\
`   em.mdp, md.mdp    :  run parameters file for GROMACS`\
`   root_NEW.pdb      :  final pdb file generated by ACPYPE`\
`   root_CNS.top      :  topology file for CNS/XPLOR`\
`   root_CNS.par      :  parameter file for CNS/XPLOR`\
`   root_CNS.inp      :  run parameters file for CNS/XPLOR`\
`   root_CHARMM.rtf   :  topology file for CHARMM`\
`   root_CHARMM.prm   :  parameter file for CHARMM`\
`   root_CHARMM.inp   :  run parameters file for CHARMM`

options:

` -h, --help            show this help message and exit`\
` -i INPUT, --input INPUT`\
`                       input file type like '.pdb', '.mdl', '.mol2' or SMILES string (mandatory if -p and -x not set)`\
` -b BASENAME, --basename BASENAME`\
`                       a basename for the project (folder and output files)`\
` -x INPCRD, --inpcrd INPCRD`\
`                       amber inpcrd file name (always used with -p)`\
` -p PRMTOP, --prmtop PRMTOP`\
`                       amber prmtop file name (always used with -x)`\
` -c {gas,bcc,user}, --charge_method {gas,bcc,user}`\
`                       charge method: gas, bcc (default), user (users charges in mol2 file)`\
` -n NET_CHARGE, --net_charge NET_CHARGE`\
`                       net molecular charge (int), it tries to guess it if not not declared`\
` -m MULTIPLICITY, --multiplicity MULTIPLICITY`\
`                       multiplicity (2S+1), default is 1`\
` -a {gaff,amber,gaff2,amber2}, --atom_type {gaff,amber,gaff2,amber2}`\
`                       atom type, can be gaff, gaff2 (default), amber (AMBER14SB) or amber2 (AMBER14SB + GAFF2)`\
` -q {mopac,sqm,divcon}, --qprog {mopac,sqm,divcon}`\
`                       am1-bcc flag, sqm (default), divcon, mopac`\
` -k KEYWORD, --keyword KEYWORD`\
`                       mopac or sqm keyword, inside quotes`\
` -f, --force           force topologies recalculation anew`\
` -d, --debug           for debugging purposes, keep any temporary file created (not allowed with arg -w)`\
` -w, --verboseless     print nothing (not allowed with arg -d)`\
` -o {all,gmx,cns,charmm}, --outtop {all,gmx,cns,charmm}`\
`                       output topologies: all (default), gmx, cns or charmm`\
` -z, --gmx4            write RB dihedrals old GMX 4.0`\
` -t, --cnstop          write CNS topology with allhdg-like parameters (experimental)`\
` -s MAX_TIME, --max_time MAX_TIME`\
`                       max time (in sec) tolerance for sqm/mopac, default is 3 hours`\
` -y, --ipython         start iPython interpreter`\
` -g, --merge           Merge lower and uppercase atomtypes in GMX top file if identical parameters`\
` -u, --direct          for 'amb2gmx' mode, does a direct conversion, for any solvent (EXPERIMENTAL)`\
` -l, --sorted          sort atoms for GMX ordering`\
` -j, --chiral          create improper dihedral parameters for chiral atoms in CNS`\
` -v, --version         Show the Acpype version and exit`

}}

## Using ACPYPE {#using_acpype}

### Running as a non-interactive job {#running_as_a_non_interactive_job}

You can run ACPYPE as a short job with a job script similar to the one shown below.

If you have already a file with the 3D-coordinates of your molecule, you can delete the lines that use `obabel` to generate the file `adp.mol2` from the [SMILES](https://en.wikipedia.org/wiki/SMILES) string.

```{=mediawiki}
{{File
  |name=job_acpype_ADP.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M

module load python  openbabel
source ~/venv_acpype/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# generate "adp.mol2" file from SMILES string:
obabel  -:"c1nc(c2c(n1)n(cn2)[C@H]3[C@@H](https://docs.alliancecan.ca/[C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)O)O)O)N" \
    -i smi -o mol2 -O adp.mol2 -h  --gen3d

acpype -i adp.mol2
}}
```
### Running on a login node {#running_on_a_login_node}

As apart of the topology generation, ACPYPE will run a short QM calculation to optimize the structure and determine the partial charges. For small molecules this should take less than two minutes and can therefore be done on a login-node, however in this case the number of threads should be limited by running ACPYPE with: `OMP_NUM_THREADS=2 acpype ...`.

For larger molecules or generating topologies for several molecules you should submit a job as shown above.

First the [python](https://docs.alliancecan.ca/Python "python"){.wikilink} and [openbabel](https://docs.alliancecan.ca/Open_Babel "openbabel"){.wikilink} need to be loaded. We also download a structure file of Adenosine triphosphate (ATP) from [PubChem](https://pubchem.ncbi.nlm.nih.gov/):

3d&response_typesave&response_basenameATP\" -O atp.sdf }}

We run ACPYPE, restricting it to using a maximum of two threads:

2 acpype -i atp.sdf \|result= ============================================================================

`ACPYPE: AnteChamber PYthon Parser interfacE v. 2023.10.27 (c) 2025 AWSdS `

============================================================================ WARNING: no charge value given, trying to guess one\... ==\> \... charge set to 0

## \> Executing Antechamber\... {#executing_antechamber...}

\> \* Antechamber OK \*

## \> \* Parmchk OK \* {#parmchk_ok}

\> Executing Tleap\...

## \> \* Tleap OK \* \[\...\] {#tleap_ok_...}

\> Removing temporary files\... Total time of execution: 1m 32s }}

The directory `atp.acpype` is created

## Useful Links {#useful_links}

- [Frequent Asked Questions about ACPYPE](https://github.com/alanwilter/acpype/wiki/Frequent-Asked-Questions-about-ACPYPE)
- [Tutorial Using ACPYPE for GROMACS](https://github.com/alanwilter/acpype/wiki/Tutorial-Using-ACPYPE-for-GROMACS)
- [Tutorial NAMD](https://github.com/alanwilter/acpype/wiki/Tutorial-NAMD)
