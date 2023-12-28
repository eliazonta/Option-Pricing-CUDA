#!/bin/bash

echo "Surkov (2009) table C.2 p.106"
echo "Parameters EUR-D and KJD-A p.103-104"
echo "Reference price 0.0426761 and 0.0426478"
params="--payoff call --exercise european --S 1 --K 1 --T 0.2
        --sigma 0.2 --r 0.0
        --koujumps --lambda 0.2 --p 0.5 --etaUp 3 --etaDown 2"
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
./bin/option $1 $params --resolution 8192
./bin/option $1 $params --resolution 16384
./bin/option $1 $params --resolution 32768
