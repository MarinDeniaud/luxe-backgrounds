#!/bin/sh

# source environment modules command
# we did this for the old installation using Environment Modules made by EasyBuild
#source /etc/profile.d/modules.sh

# environment - bdsim, root, geant4 etc
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc11-opt/bdsim-env-develop-g4v11.0.2-boost.sh

# echo inputs
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
INPUTFILE=$1
OUTPUTFILE=$2
SEEDVALUE=$(($3+$4))
ANALFILE=$5
NGEN=$6
echo "input file    = "${INPUTFILE}
echo "output file   = "${OUTPUTFILE}
echo "seed value    = "${SEEDVALUE}
echo "analysis file = "${ANALFILE}
echo "ngenerate     = "${NGEN}

date
echo ""
hostname
echo ""

# run bdsim
# a common trick is to use the seed value as the output file name so we know what it was easily
bdsim  --file=${INPUTFILE} --outfile=${OUTPUTFILE} --batch --seed=${SEEDVALUE} --ngenerate=${NGEN}
echo "job finished"
date

export PYTHONPATH="/scratch2/mdeniaud/phd/pybdsim:$PYTHONPATH"
export PYTHONPATH="/scratch2/mdeniaud/phd/pymad8:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/scratch2/mdeniaud/phd/luxe-backgrounds/06_analysis"

# run python analysis script to create the histogram root file
/scratch2/mdeniaud/phd/luxe-backgrounds/03_bdsimModel/bdsimAnalysisJob.py -i ${OUTPUTFILE}.root
