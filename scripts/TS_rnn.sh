#!/usr/bin/bash

#setup anaconda
. /etc/profile.d/anaconda.sh
setup-anaconda

source activate data_mining

python /smartdata/hj7422/Documents/Workplace/Trumpf/code/TS_rnn3.py
