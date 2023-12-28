#!/bin/bash

echo "Surkov (2009) table C.1 p.106"
echo "Parameters EUR-C and MJD-C p.103-104"
echo "Reference price 4.391243"
params="--payoff call --exercise european --S 100 --K 100 --T 0.25
        --sigma 0.15 --r 0.05
        --mertonjumps --lambda 0.1 --mertonmu -0.9 --mertongamma 0.45"
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
./bin/option $1 $params --resolution 8192
./bin/option $1 $params --resolution 16384
./bin/option $1 $params --resolution 32768
