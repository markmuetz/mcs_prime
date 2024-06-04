#!/bin/bash
# QUOTES BELOW ARE IMPORTANT!
# E.g. ./monsoon_ssh_tar_cp.sh /projects/mcsprime/mamue/cylc-run/u-dg135/ "share/cycle/20200701T0000Z/engl/um/em*/*.pp"
# See https://code.metoffice.gov.uk/doc/monsoon2/dataTransfer.html#tar-method
set -x # echo on

# Must be set up in ~/.ssh/config
SERVER=Monsoon
# MONSOON_ROOT_PATH=/projects/mcsprime/mamue/cylc-run/u-dg135/
# MONSOON_REL_PATH=share/cycle/20200701T0000Z/engl/um/em*/
MONSOON_ROOT_PATH="$1"
MONSOON_REL_PATH="$2"

# Might be needed.
# ssh -oHostKeyAlgorithms=+ssh-dss ${SERVER} "ssh xcsc \" cd ${MONSOON_ROOT_PATH} && tar cf - ${MONSOON_REL_PATH} \" " | tar xvf -
ssh ${SERVER} "ssh xcsc \" cd ${MONSOON_ROOT_PATH} && tar cf - ${MONSOON_REL_PATH} \" " | tar xvf -
