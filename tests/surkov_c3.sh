#!/bin/bash

echo "Surkov (2009) table C.3 p.106"
echo "Parameters EUR-E and VG-A p.103-104"
echo "Reference price 7.49639670"
params="--payoff call --exercise european --S 100 --K 100 --T 0.46575
        --sigma 0.19071 --r 0.0549 --dividend 0.011
        --vg --vgmu 0.49083 --vgdrift -0.28113"
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
./bin/option $1 $params --resolution 8192
./bin/option $1 $params --resolution 16384
./bin/option $1 $params --resolution 32768
