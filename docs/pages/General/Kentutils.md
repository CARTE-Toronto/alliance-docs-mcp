---
title: "Kentutils"
url: "https://docs.alliancecan.ca/wiki/Kentutils"
category: "General"
last_modified: "2024-02-09T19:25:39Z"
page_id: 13870
display_title: "Kentutils"
---

KentUtils are the UCSC Genome Bioinformatics Group\'s suite of biological analysis and web display programs as well as some of Jim Kent\'s own tools [1](http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads),[2](http://hgdownload.soe.ucsc.edu/admin/exe/),[3](https://github.com/ucscGenomeBrowser/kent).

# Availability and loading module {#availability_and_loading_module}

In Compute Canada we provide all these tools through the kentutils module:

Which should give you the usual prerequisites:

    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      kentutils: kentutils/20180716
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Description:
          Kent utilities: collection of tools used by the UCSC genome browser.

        Properties:
          Bioinformatic libraries/apps / Logiciels de bioinformatique

        You will need to load all module(s) on any one of the lines below before the "kentutils/20180716" module is available to load.

          nixpkgs/16.09  gcc/5.4.0
          nixpkgs/16.09  gcc/6.4.0
          nixpkgs/16.09  gcc/7.3.0
          nixpkgs/16.09  intel/2016.4
          nixpkgs/16.09  intel/2018.3
     
        Help:
          
          Description
          ===========
          Kent utilities: collection of tools used by the UCSC genome browser.
          
          
          More information
          ================
           - Homepage: http://genome.cse.ucsc.edu/

To load, simply load the prerequisites before or along with the package:

Or

Or

Or

Or

The availability of different pre-requisites allow you to load other modules that you might need in your pipeline without affecting the KentUtils load.

# Tools available {#tools_available}

Kentutils provides the following tools:

<div style="column-count:2;-moz-column-count:2;-webkit-column-count:6">

- addCols
- ameme
- autoDtd
- autoSql
- autoXml
- ave
- aveCols
- axtChain
- axtSort
- axtSwap
- axtToMaf
- axtToPsl
- bamToPsl
- barChartMaxLimit
- bedClip
- bedCommonRegions
- bedCoverage
- bedExtendRanges
- bedGeneParts
- bedGraphPack
- bedGraphToBigWig
- bedIntersect
- bedItemOverlapCount
- bedJoinTabOffset
- bedJoinTabOffset.py
- bedMergeAdjacent
- bedPartition
- bedPileUps
- bedRemoveOverlap
- bedRestrictToPositions
- bedSort
- bedToBigBed
- bedToExons
- bedToGenePred
- bedToPsl
- bedWeedOverlapping
- bigBedInfo
- bigBedNamedItems
- bigBedSummary
- bigBedToBed
- bigMafToMaf
- bigPslToPsl
- bigWigAverageOverBed
- bigWigCat
- bigWigCluster
- bigWigCorrelate
- bigWigInfo
- bigWigMerge
- bigWigSummary
- bigWigToBedGraph
- bigWigToWig
- binFromRange
- blastToPsl
- blastXmlToPsl
- blat
- calc
- catDir
- catUncomment
- chainAntiRepeat
- chainBridge
- chainFilter
- chainMergeSort
- chainNet
- chainPreNet
- chainSort
- chainSplit
- chainStitchId
- chainSwap
- chainToAxt
- chainToPsl
- chainToPslBasic
- checkAgpAndFa
- checkCoverageGaps
- checkHgFindSpec
- checkTableCoords
- chopFaLines
- chromGraphFromBin
- chromGraphToBin
- chromToUcsc
- clusterGenes
- colTransform
- countChars
- cpg_lh
- crTreeIndexBed
- crTreeSearchBed
- dbSnoop
- dbTrash
- endsInLf
- estOrient
- expMatrixToBarchartBed
- faAlign
- faCmp
- faCount
- faFilter
- faFilterN
- faFrag
- faNoise
- faOneRecord
- faPolyASizes
- faRandomize
- faRc
- faSize
- faSomeRecords
- faSplit
- faToFastq
- faToTab
- faToTwoBit
- faTrans
- fastqStatsAndSubsample
- fastqToFa
- featureBits
- fetchChromSizes
- findMotif
- fixStepToBedGraph.pl
- gapToLift
- genePredCheck
- genePredFilter
- genePredHisto
- genePredSingleCover
- genePredToBed
- genePredToBigGenePred
- genePredToFakePsl
- genePredToGtf
- genePredToMafFrames
- genePredToProt
- gensub2
- getRna
- getRnaPred
- gff3ToGenePred
- gff3ToPsl
- gmtime
- gtfToGenePred
- headRest
- hgBbiDbLink
- hgFakeAgp
- hgFindSpec
- hgGcPercent
- hgGoldGapGl
- hgLoadBed
- hgLoadChain
- hgLoadGap
- hgLoadMaf
- hgLoadMafSummary
- hgLoadNet
- hgLoadOut
- hgLoadOutJoined
- hgLoadSqlTab
- hgLoadWiggle
- hgSpeciesRna
- hgTrackDb
- hgWiggle
- hgsql
- hgsqldump
- hgvsToVcf
- hicInfo
- htmlCheck
- hubCheck
- hubClone
- hubPublicCheck
- ixIxx
- lastz-1.04.00
- lastz_D-1.04.00
- lavToAxt
- lavToPsl
- ldHgGene
- liftOver
- liftOverMerge
- liftUp
- linesToRa
- localtime
- mafAddIRows
- mafAddQRows
- mafCoverage
- mafFetch
- mafFilter
- mafFrag
- mafFrags
- mafGene
- mafMeFirst
- mafNoAlign
- mafOrder
- mafRanges
- mafSpeciesList
- mafSpeciesSubset
- mafSplit
- mafSplitPos
- mafToAxt
- mafToBigMaf
- mafToPsl
- mafToSnpBed
- mafsInRegion
- makeTableList
- maskOutFa
- mktime
- mrnaToGene
- netChainSubset
- netClass
- netFilter
- netSplit
- netSyntenic
- netToAxt
- netToBed
- newProg
- newPythonProg
- nibFrag
- nibSize
- oligoMatch
- overlapSelect
- para
- paraFetch
- paraHub
- paraHubStop
- paraNode
- paraNodeStart
- paraNodeStatus
- paraNodeStop
- paraSync
- paraTestJob
- parasol
- positionalTblCheck
- pslCDnaFilter
- pslCat
- pslCheck
- pslDropOverlap
- pslFilter
- pslHisto
- pslLiftSubrangeBlat
- pslMap
- pslMapPostChain
- pslMrnaCover
- pslPairs
- pslPartition
- pslPosTarget
- pslPretty
- pslRc
- pslRecalcMatch
- pslRemoveFrameShifts
- pslReps
- pslScore
- pslSelect
- pslSomeRecords
- pslSort
- pslStats
- pslSwap
- pslToBed
- pslToBigPsl
- pslToChain
- pslToPslx
- pslxToFa
- qaToQac
- qacAgpLift
- qacToQa
- qacToWig
- raSqlQuery
- raToLines
- raToTab
- randomLines
- rmFaDups
- rowsToCols
- sizeof
- spacedToTab
- splitFile
- splitFileByColumn
- sqlToXml
- strexCalc
- stringify
- subChar
- subColumn
- tabQuery
- tailLines
- tdbQuery
- tdbRename
- tdbSort
- textHistogram
- tickToDate
- toLower
- toUpper
- trackDbIndexBb
- transMapPslToGenePred
- trfBig
- twoBitDup
- twoBitInfo
- twoBitMask
- twoBitToFa
- ucscApiClient
- udr
- vai.pl
- validateFiles
- validateManifest
- varStepToBedGraph.pl
- webSync
- wigCorrelate
- wigEncode
- wigToBigWig
- wordLine
- xmlCat
- xmlToSql

</div>

# Versions

The latest version of KentUtils tools in the software stack was released in April 2018 [4](http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/)

# Documentation

Issuing the command without any arguments will give a brief description of the executable in most cases.

For example:

Will result in:

    gtfToGenePred - convert a GTF file to a genePred
    usage:
       gtfToGenePred gtf genePred

    options:
         -genePredExt - create a extended genePred, including frame
          information and gene name
         -allErrors - skip groups with errors rather than aborting.
          Useful for getting infomation about as many errors as possible.
         -ignoreGroupsWithoutExons - skip groups contain no exons rather than
          generate an error.
         -infoOut=file - write a file with information on each transcript
         -sourcePrefix=pre - only process entries where the source name has the
          specified prefix.  May be repeated.
         -impliedStopAfterCds - implied stop codon in after CDS
         -simple    - just check column validity, not hierarchy, resulting genePred may be damaged
         -geneNameAsName2 - if specified, use gene_name for the name2 field
          instead of gene_id.
         -includeVersion - it gene_version and/or transcript_version attributes exist, include the version
          in the corresponding identifiers.

Many of the KentUtils executables are poorly documented. If you require specific help about the executables, you can submit questions to the main UCSC discussion list. See [5](http://genome.ucsc.edu/contacts.html).

# References

[Downloads](http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)

[Genome Browser and Blat application binaries](http://hgdownload.soe.ucsc.edu/admin/exe/)

[UCSC Genome Bioinformatics Group\'s suite](https://github.com/ucscGenomeBrowser/kent)

[Executables](http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/)

[Discussion list](http://genome.ucsc.edu/contacts.html)
