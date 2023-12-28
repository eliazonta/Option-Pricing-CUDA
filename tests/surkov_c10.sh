#!/bin/bash

echo "Surkov (2009) table C.10 p.109"
echo "Parameters AMR-D MJD-C p.103-104"
echo "Reference price 4.537366137"
params="--payoff put --exercise american --S 100 --K 95 --T 0.75
        --sigma 0.15 --r 0.05
        --mertonjumps --lambda 0.1 --mertonmu -0.9 --mertongamma 0.45"
./bin/option $1 $params --resolution 4096 --timesteps 512
./bin/option $1 $params --resolution 8192 --timesteps 1024
./bin/option $1 $params --resolution 16384 --timesteps 2048
