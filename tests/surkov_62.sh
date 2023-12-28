#!/bin/bash

echo ""
echo "Lippa (2013), table C.2 p.55"
echo "Surkov (2009), table 6.2 p.92"
echo "Reference price 7.27993383"
params="--payoff call --exercise european --S 100 --K 110 --r 0 --T 1 --sigma 0.2
        --koujumps --lambda 0.2 --p 0.5 --etaUp 3 --etaDown 2"
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
./bin/option $1 $params --resolution 8192
./bin/option $1 $params --resolution 16384
./bin/option $1 $params --resolution 32768
