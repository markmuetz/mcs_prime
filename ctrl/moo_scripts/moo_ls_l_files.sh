source suite.sh
for var in a b c d e f g j k m; do moo ls -l moose:/crum/${SUITE}/ap${var}.pp|tee -a ${SUITE}_ls_l_files.txt; done
