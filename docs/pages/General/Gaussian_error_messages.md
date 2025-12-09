---
title: "Gaussian error messages/en"
url: "https://docs.alliancecan.ca/wiki/Gaussian_error_messages/en"
category: "General"
last_modified: "2022-07-19T22:43:17Z"
page_id: 6205
display_title: "Gaussian error messages"
---

:Parent page: Gaussian
This list was originally assembled by Professor Cory C. Pye.

== Normal termination of Gaussian ==
Normally Gaussian will terminate with a line such as

 Job cpu time:       0 days  0 hours 26 minutes 18.3 seconds.
 Elapsed time:       0 days  0 hours  6 minutes 43.3 seconds.
 Normal termination of Gaussian 16 at Tue Nov 14 15:31:56 2017.

If a job does not end with these lines and no other error message is produced, it usually means thatː
* your disk quota was exceeded (see File quotas and policies);
* the job exceeded the time requested from the scheduler (--time=HH:MM:SS);
* the job exceeded the memory requested (--mem=) (see Monitoring jobs); or
* the job produced more data that can fill up the local disk on the compute node.

== Erroneous Write ==
;Description of error
Near the end of the output, it reads something similar to

Erroneous write. write 122880 instead of 4239360.
fd = 3
Erroneous write. write 122880 instead of 4239360.
fd = 3
writwa
writwa: File exists

or

Erroneous write. write -1 instead of 3648000.

fd = 4
writwa
writwa: No space left on device

or

Erroneous write during file extend. write -1 instead of 8192
Probably out of disk space.
Write error in NtrExt1

;Explanation of error
Typically, this occurs when you have run out of disk space. This could occur if you have exceeded your quota, if a disk is physically full, or (more rarely) a network drive is unavailable because of a communication time out.

;Fixing the error
* Check your disk quota with quota. Delete unnecessary files.
* Your job may simply be too big to run on current hardware. Try using a smaller basis set.

== Link 9999 ==
;Description of error
At the end of your output, you get lines such as

 Error termination request processed by link 9999.
 Error termination via Lnk1e in /disc30/g98/l9999.exe.

A few pages above, the following lines appear:

 Maximum Force            0.020301     0.000450     NO
 RMS     Force            0.007068     0.000300     NO
 Maximum Displacement     0.078972     0.001800     NO
 RMS     Displacement     0.023716     0.001200     NO
 Predicted change in Energy=-3.132299D-05
 Optimization stopped.
    -- Number of steps exceeded,  NStep=  34
    -- Flag reset to prevent archiving.
                       ----------------------------
                       ! Non-Optimized Parameters !
                       ! (Angstroms and Degrees)  !

;Explanation of error
This means that the Gaussian job terminated abnormally for some reason internal to Gaussian. The most common cause is that a geometry optimization has not converged.

;Fixing the error
Geometry optimizations usually fail to converge for one of these reasons:
* If your initial starting structure is not good, can you provide a better starting structure? For example, optimize the structure at a lower level of theory and use that as the new starting structure. However, if it looks as if the structure is converging to what you want, as seen in your visualizer of choice, then restart the optimization from the last step, for example by using geom=allcheck in the route line. It may also be a good idea to use opt=CalcFC in these situations if it is not too expensive. It is probably not too expensive at the HF or DFT level of theory.
* If your starting force constants matrix (Hessian) is poor, use a better one. This typically manifests itself when the force constants vary a lot between levels, or if there is a large geometry change during the optimization. One can carry out a series of linked jobs using --Link1--. If you have a previous job, then usually Opt=ReadFC works well, but occasionally Opt=CalcFC, or rarely Opt=CalcAll are needed. In these cases, the forces are often converged, but the step sizes are not, and the final output will look like

         Item               Value     Threshold  Converged?
 Maximum Force            0.000401     0.000450     YES
 RMS     Force            0.000178     0.000300     YES
 Maximum Displacement     0.010503     0.001800     NO
 RMS     Displacement     0.003163     0.001200     NO

* Rarely, the coordinate system itself may be at fault. If coordinates are specified with a z-matrix, it is easy to make poor choices which result in three of the four atoms used to define a torsion angle (also called a dihedral angle) being collinear, meaning the angle is close to 0 or 180 degrees. This can cause the optimization algorithm to fail. In this case you should either formulate a better z-matrix or use the default redundant internal coordinates.
* If these methods fail, another option is to change the optimization method from the default to another type, such as opt=ef (if the number of variables is less than 50) or opt=gdiis (for floppy molecules).

== angle Alpha is outside the valid range of 0 to 180 ==
;Description of error
At the end of your output, you get a line such as

 ------------------------------------------------------------------------
 Error termination via Lnk1e in /disc30/g98/l716.exe.

The lines above will be a z-matrix, above which will contain lines such as

  Error on Z-matrix card number    9
 angle Alpha is outside the valid range of 0 to 180.
 Conversion from Z-matrix to cartesian coordinates failed:
 ------------------------------------------------------------------------
                         Z-MATRIX (ANGSTROMS AND DEGREES)
 CD Cent Atom  N1     Length/X     N2    Alpha/Y     N3     Beta/Z      J
 ------------------------------------------------------------------------
...
  9   9  H     8   0.962154(  8)   1   -1.879( 16)   2    0.000( 23)   0
...

;Explanation of error
This means that the Gaussian job terminated abnormally because an angle x in the z-matrix changed to become outside the allowed range of 0 < x < 180.

;Fixing the error
This can happen if there are large geometry changes in a molecule, especially one composed of interacting fragments. Re-define the z-matrix, or use a different coordinate system.

== Reading basis center ==
;Description of error
At the end of your output, you get a line such as

 End of file reading basis center.
 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.9 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=   10 Scr=    1

;Explanation of error
This is an input error. You are attempting to read in a general basis set but you forgot to put in the basis set.

;Fixing the error
Put in the basis set, or remove the gen from the route line and specify an internal basis set.

== Operation on file out of range ==
;Description of error
At the end of your output, you get a line such as

 Error termination in NtrErr:
 NtrErr Called from FileIO.

Above this, you have something like

 Operation on file out of range.
FileIO: IOper= 2 IFilNo(1)=-19999 Len=     1829888 IPos=  -900525056 Q=       4352094416

 dumping /fiocom/, unit = 1 NFiles =   109 SizExt =    524288 WInBlk =      1024
                   defal = T LstWrd =  7437256704 FType=2 FMxFil=10000

...followed by a lot of numbers.

;Explanation of error
This typically happens when you try to retrieve something from the checkpoint file (with Opt=ReadFC or guess=read or geom=allcheck/modify) that is not there, either because you did not calculate it previously, or you ran out of disk space or time while running the earlier job and the information needed wasn't written to the checkpoint file.

;Fixing the error
Re-calculate or type in the information required.

== End of file in GetChg ==
;Description of error
At the end of your output, you get a line such as

 Symbolic Z-matrix:
 End of file in GetChg.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   11 Scr=    1

;Explanation of error
You have specified an input in which the charge/multiplicity line is required, but you forgot to put it in. Alternatively, you meant to read the charge/multiplicity from the checkpoint, but forgot to put geom=allcheck in the route section.

;Fixing the error
Put the charge/multiplicity line in, or put geom=allcheck in the route section.

== Change in point group or standard orientation ==
;Description of error
At the end of your output, you get a line such as

 Stoichiometry    CdH14O7(2+)
 Framework group  C2[C2(CdO),X(H14O6)]
 Deg. of freedom   30
 Full point group                 C2      NOp   2
 Omega: Change in point group or standard orientation.

 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  3 hours 35 minutes 40.8 seconds.
 File lengths (MBytes):  RWF=   58 Int=    0 D2E=    0 Chk=   19 Scr=    1

;Explanation of error
During the optimization process, either the standard orientation or the point group of the molecule has changed. If the former, a visualization program will show this as a sudden flipping of the structure, typically by 180 degrees. This error is a lot less common since Gaussian 03 was released.

;Fixing the error
* This can indicate that your z-matrix is not correctly specified, if you go from a point group (e.g C2v) to a subgroup of the point group (C2 or Cs or C1).
* If the point group here is correct, it could indicate that your starting structure had too high symmetry and you should desymmetrize it.
* Rarely: If the point group here is incorrect (of higher symmetry), then your z-matrix should be reformulated with more symmetry.

* If you don't care about symmetry, then you could turn symmetry completely off.

== Unrecognized atomic symbol ==
;Description of error
At the end of your output, you get a line such as

 General basis read from cards:  (6D, 7F)
 Unrecognized atomic symbol ic2

 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.6 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   12 Scr=    1

;Explanation of error
You are reading in a general basis set, but the atom specified (in the above example, ic2) does not match any standard atomic symbol. This can also happen in a link job if a previous step uses default coordinates (which wipes the z-matrix) and then you try to modify the z-matrix with geom=modify. The z-matrix variable section is ignored, but Gaussian may attempt to interpret it as basis set information.

;Fixing the error
Type in the correct atomic symbol.

== Convergence failure -- run terminated ==
;Description of error
At the end of your output, you get a line such as

 >>>>>>>>>> Convergence criterion not met.
 SCF Done:  E(RHF) =  -2131.95693715     A.U. after  257 cycles
             Convg  =    0.8831D-03             -V/T =  2.0048
             S**2   =   0.0000
 Convergence failure -- run terminated.
 Error termination via Lnk1e in /disc30/g98/l502.exe.
 Job cpu time:  0 days  0 hours  5 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=   15 Int=    0 D2E=    0 Chk=    8 Scr=    1

or

  >>>>>>>>>> Convergence criterion not met.
 SCF Done:  E(UHF) =  -918.564956094     A.U. after   65 cycles
             Convg  =    0.4502D-04             -V/T =  2.0002
             S**2   =   0.8616
 Annihilation of the first spin contaminant:
 S**2 before annihilation     0.8616,   after     0.7531
 Convergence failure -- run terminated.
 Error termination via Lnk1e in /disc30/g98/l502.exe.
 Job cpu time:  0 days  0 hours  3 minutes 56.2 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    8 Scr=    1

;Explanation of error
The SCF (self-consistent field) procedure has failed to converge.

;Fixing the error
The SCF procedure might fail to converge if a poor guess is provided for the molecular orbitals.
* Try using a better guess (guess=read) by carrying out an SCF using the same starting structure, but at a lower level of theory such as HF/STO-3G.
* If this doesn't work, try using an alternate SCF converger such as SCF=QC or SCF=XQC.
In some cases, a poor geometry can result in an unconverged SCF, if a bond is way too long or too short.
* Fixing the initial geometry may fix the problem.
In some cases, the optimizer itself takes a bad step, resulting in this error.
* Resubmitting the job with the penultimate (or earlier) geometry and a newly evaluated Hessian can fix this.

== FOPT requested but NVar= XX while NDOF= YY ==
;Description of error
At the end of your output, you get a line such as

 FOPT requested but NVar= 29 while NDOF= 15.
 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.3 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    1 Scr=    1

;Explanation of error
You have requested a full optimization (FOpt), including checking the variables to make sure the correct number are present. The check indicates that there is an error.

;Fixing the error
If NDOF is less than NVar, then it means that the molecule is being run in a lower symmetry than it actually is, and you should consider running it with higher symmetry.
If NVar is less than NDOF, it usually means that your z-matrix has too many constraints, not appropriate to the actual symmetry.
You can bypass the check using Opt instead of FOpt, but this is not recommended.

== Unable to project read-in occupied orbitals. ==
;Description of error
At the end of your output, you get lines such as

 Initial guess read from the checkpoint file:
 BiAq7_3+_C2.chk
 Unable to project full set of read-in orbitals.
 Projecting just the  36 occupied ones.
 Unable to project read-in occupied orbitals.
 Error termination via Lnk1e in /disc30/g98/l401.exe.
 Job cpu time:  0 days  0 hours  0 minutes 29.5 seconds.
 File lengths (MBytes):  RWF=   18 Int=    0 D2E=    0 Chk=   17 Scr=    1

;Explanation of error
You are reading in a molecular-orbital guess from the checkpoint file, but the projection from the old to the new basis set has failed. This can happen if certain pseudopotential basis sets (CEP-121G*) are used with polarization functions where no polarization functions actually exist. In some cases Gaussian uses placeholder polarization functions with zero exponent.

;Fixing the error
* Don't use CEP-121G*, use CEP-121G for the elements in question. They are the same for many elements.
* A workaround is not to use the guess.

== KLT.ge.NIJTC in GetRSB ==
;Description of error
At the end of your output, you get lines such as

 (rs|ai) integrals will be sorted in core.
 KLT.ge.NIJTC in GetRSB.
 Error termination via Lnk1e in /disc30/g98/l906.exe.
 Job cpu time:  0 days  0 hours  0 minutes 32.7 seconds.
 File lengths (MBytes):  RWF=  514 Int=    0 D2E=    0 Chk=   10 Scr=    1

;Explanation of error
The MP2 calculation has failed. It may be related to the pseudopotential problem above.

;Fixing the error
Don't use CEP-121G*, use CEP-121G for the elements in question. They are the same for many elements.

== Symbol XXX not found in Z-matrix ==
;Description of error
At the end of your output, you get lines such as

 Symbol "H3NNN" not found in Z-matrix.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   14 Scr=    1

;Explanation of error
This is an input error. You have typed in a variable name (in the above example, H3NNN) that is not in the z-matrix.

;Fixing the error
Either type the correct symbol, or add it to the z-matrix, as appropriate.

== Variable X has invalid number of steps. ==
;Description of error
At the end of your output, you get lines such as

 Scan the potential surface.
 Variable   Value     No. Steps Step-Size
 -------- ----------- --------- ---------
 Variable  1 has invalid number of steps      -1.
 Error termination via Lnk1e in /disc30/g98/l108.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.7 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=   13 Scr=    1

;Explanation of error
This is an input error. You are attempting to do a generate rigid potential energy scan. Most likely, you have two blank lines instead of one between the z-matrix and the variables.

;Fixing the error
Delete the extra blank line.

== Problem with the distance matrix. ==
;Description of error
At the end of your output, you get lines such as

  Problem with the distance matrix.
 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  9 hours 11 minutes 14.3 seconds.
 File lengths (MBytes):  RWF=  634 Int=    0 D2E=    0 Chk=   10 Scr=    1

;Explanation of error
This can be an input error. At least two atoms are too close together, with the list given above.
Rarely, this can be a program error, particularly when one of the distances is NaN. This can happen when trying to optimize diatomics and you start off with too large a distance.

;Fixing the error
You will need to check variables and the z-matrix of the atoms in question to make sure there are no atoms close together. This can be a result of a missing minus sign in a torsion angle for molecules with planes of symmetry, in which the two atoms related by the plane of symmetry are now coincident, that is, they have distance 0 between them.

== End of file in ZSymb ==
;Description of error
At the end of your output, you get lines such as

 Symbolic Z-matrix:
 Charge =  0 Multiplicity = 1
 End of file in ZSymb.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.6 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    9 Scr=    1

;Explanation of error
This is an input error. Gaussian can't find the z-matrix. There are two common causes:
# You may have omitted the blank line at the end of the geometry specification.
# You may have intended to get the z-matrix and parameters from the checkpoint file, but forgot to type geom=check.

;Fixing the error
Add a blank line at the end or add geom=check.

== Linear search skipped for unknown reason ==
;Description of error
At the end of your output, you get lines such as

 RFO could not converge Lambda in  999 iterations.
 Linear search skipped for unknown reason.
 Error termination via Lnk1e in /disc30/g98/l103.exe.
 Job cpu time:  0 days  7 hours  9 minutes 17.0 seconds.
 File lengths (MBytes):  RWF=   21 Int=    0 D2E=    0 Chk=    6 Scr=    1

;Explanation of error
The rational function optimization was not successful during a linear search. Most likely the Hessian is no longer valid.

;Fixing the error
Restart optimization using Opt=CalcFC.

== Variable index of 3000 on card XXX is out of range, NVar=XX ==
;Description of error
At the end of your output, you get lines such as

 Variable index of 3000 on card  15 is out of range, NVar=  42.
 Termination in UpdVr1.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    8 Scr=    1

;Explanation of error
This is an input error. You forgot to add a variable in your Z-matrix to your list. In the above example, it is a variable which defines atom #15.

;Fixing the error
Add the variable.

== Unknown center XXX ==
;Description of error
At the end of your output, you get lines such as

 Unknown center X
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    8 Scr=    1

;Explanation of error
This is an input error. You are trying to define an atom in a Z-matrix using another non-existent atom (in the above example, X)

;Fixing the error
Fix the atom name.

== Determination of dummy atom variables in z-matrix conversion failed ==
;Description of error
At the end of your output, you get lines such as

 Error termination request processed by link 9999.
 Error termination via Lnk1e in /disc30/g98/l9999.exe.
 Job cpu time:  0 days  1 hours 53 minutes 10.4 seconds.
 File lengths (MBytes):  RWF=   20 Int=    0 D2E=    0 Chk=   11 Scr=    1

 and just before

 Determination of dummy atom variables in z-matrix conversion failed.
 Determination of dummy atom variables in z-matrix conversion failed.
 NNew=      6.03366976D+01 NOld=      5.07835896D+01 Diff= 9.55D+00

;Explanation of error
The conversion from redundant internal coordinates to Z-matrix coordinates failed because of the dummy atoms. You will have to make do with Cartesian coordinates.

;Fixing the error
The geometry optimization converged, but Gaussian couldn't convert back to the input z-matrix.

== malloc failed ==
;Description of error
At the end of your output, you get lines such as

malloc failed.: Resource temporarily unavailable
malloc failed.

;Explanation of error
This is not strictly a Gaussian error. It typically means that the computer has run out of memory, perhaps because you asked for too much memory in the %mem line.

;Fixing the error
* Reduce the amount specified with %mem.
* Or, increase the amount of memory requested in the job script with --mem=.

== Charge and multiplicity card seems defective: ==
;Description of error
At the end of your output, you get lines such as

 ----
 -2 1
 ----
 Z-Matrix taken from the checkpoint file:
 oxalate_2-_Aq1_C2.chk
 Charge and multiplicity card seems defective:
 Charge is bogus.
  WANTED AN INTEGER AS INPUT.
  FOUND A STRING AS INPUT.
 CX      =  0.7995

   ?
 Error termination via Lnk1e in /disc30/g98/l101.exe.

;Explanation of error
This is an input error. If the title line is forgotten when using geom=modify, then it interprets the charge/multiplicity line (in the above example, "-2 1") as the title, and then tries to interpret the variable list as the charge/multiplicity line.

;Fixing the error
Put in a title line.

== Attempt to redefine unrecognized symbol "XXXXX" ==
;Description of error
At the end of your output, you get lines such as

 O2WXC  90.
 Attempt to redefine unrecognized symbol "O2WXC".
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    8 Scr=    1

;Explanation of error
This is an input error. You are attempting a geom=modify, but a variable whose value you are attempting to replace is not in the checkpoint file.

;Fixing the error
Either specify the correct checkpoint file or the correct variable.

== Inconsistency #2 in MakNEB ==
;Description of error
At the end of your output, you get lines such as

 Standard basis: 3-21G (6D, 7F)
 Inconsistency #2 in MakNEB.
 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  3 hours 46 minutes 57.4 seconds.
 File lengths (MBytes):  RWF=  245 Int=    0 D2E=    0 Chk=   11 Scr=    1

;Explanation of error
This is an input error. You have had a change in point group, and you specified iop(2/15=4,2/16=2,2/17=7) to try to avoid program crashing.

;Fixing the error
Be very careful with iop, or remove altogether.

== galloc: could not allocate memory ==
;Description of error
In the output file, you get

  galloc: could not allocate memory

;Explanation of error
This is a memory allocation error due to lack of memory. Gaussian handles memory in such a way that it actually uses about 1GB more than %mem.

;Fixing the error
The value for %mem should be at least 1GB less than the value specified in the job submission script. Conversely, the value specified for --mem in your job script should be at least 1GB greater than the amount specified in the %mem directive in your Gaussian input file. The exact increment needed seems to depend on the job type and input details; 1GB is a conservative value determined empirically.

== No such file or directory ==
;Description of error
In the job output file you get something like

PGFIO/stdio: No such file or directory
PGFIO-F-/OPEN/unit=11/error code returned by host stdio - 2.
 File name = /home/johndoe/scratch/Gau-12345.inp
 In source file ml0.f, at line number 181
  0  0x42bb41
Error: segmentation violation, address not mapped to object

;Explanation of error
The file named in the third line doesn't exist, possibly because the containing directory does not exist. This happens, for example, if you set GAUSS_SCRDIR to a directory that doesn't exist.

;Fixing the error
Create the directory with mkdir, or change the definition of GAUSS_SCRDIR to name an existing directory.