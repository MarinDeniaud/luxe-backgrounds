#!/bin/sh

# source environment modules command
# we did this for the old installation using Environment Modules made by EasyBuild
#source /etc/profile.d/modules.sh

# environment - bdsim, root, geant4 etc
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc8-opt/bdsim-env-v1.6.0-g4v10.7.2.3.sh

# echo inputs
echo $1
echo $2
echo $3
echo $4
echo $5
INPUTFILE=$1
OUTPUTDIR=$2
SEEDVALUE=$3
ANALFILE=$4
NGEN=$5
echo "input file    = "${INPUTFILE}
echo "output dir    = "${OUTPUTDIR}
echo "seed value    = "${SEEDVALUE}
echo "analysis file = "${ANALFILE}
echo "ngenerate     = "${NGEN}

date
echo ""
hostname
echo ""

# run bdsim
# a common trick is to use the seed value as the output file name so we know what it was easily
bdsim  --file=${INPUTFILE} --outfile=${SEEDVALUE}_raw --batch --seed=${SEEDVALUE} --ngenerate=${NGEN}
echo "job finished"
date

# run rebdsim analysis on output from bdsim
#rebdsim ${ANALFILE} ${SEEDVALUE}_raw.root ${SEEDVALUE}_ana.root

# copy generated output to desired output directory
mkdir -p ${OUTPUTDIR}
cp *.root ${OUTPUTDIR}/.

# clean up
rm *.root

