#!/bin/sh

# environment - bdsim, root, geant4 etc
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc11-opt/bdsim-env-v1.7.0-rc-g4v10.7.2.3-ftfp-boost.sh

# echo inputs
INPUTFILE=$1
OUTPUTFILE=$2
SEEDVALUE=$(($3+$4))
NGEN=$5
echo "input file    = "${INPUTFILE}
echo "output file   = "${OUTPUTFILE}
echo "seed value    = "${SEEDVALUE}
echo "ngenerate     = "${NGEN}

date
echo ""
hostname
echo ""

export PYTHONPATH="/scratch2/mdeniaud/phd/pybdsim/src:$PYTHONPATH"
export PYTHONPATH="/scratch2/mdeniaud/phd/pymad8/src:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/scratch2/mdeniaud/phd/luxe-backgrounds/06_analysis"

# run bdsim and python analysis script to create the histogram root file
/scratch2/mdeniaud/phd/luxe-backgrounds/03_bdsimModel/bdsimAnalysisJob_wire.py -i ${INPUTFILE} -o ${OUTPUTFILE} -p ${NGEN} -s ${SEEDVALUE}

echo "job finished"
date