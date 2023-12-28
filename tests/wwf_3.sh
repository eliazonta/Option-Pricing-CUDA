#!/bin/bash

echo "Wang, Wan, Forsynth (2007) table 2 p.18"
echo "Reference price 16.212478"
params="--payoff call --exercise european --S 90 --K 98 --T 0.25
        --r 0.06 --sigma 0
        --CGMY --C 16.97 --G 7.08 --M 29.97 --Y 0.6442"
./bin/option $1 $params --resolution 512
./bin/option $1 $params --resolution 1024
./bin/option $1 $params --resolution 2048
./bin/option $1 $params --resolution 4096
