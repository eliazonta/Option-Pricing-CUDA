#!/bin/bash

echo "Wang, Wan, Forsynth (2007) table 2 p.18"
echo "Reference price 0.61337338"
params="--payoff call --exercise european --S 90 --K 98 --T 0.5
        --r 0 --sigma 0
        --CGMY --C 5.9311 --G 20.2648 --M 39.784 --Y 0"
./bin/option $1 $params --resolution 512
./bin/option $1 $params --resolution 1024
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
