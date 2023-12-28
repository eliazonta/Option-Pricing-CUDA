#!/bin/bash

echo "Surkov (2009) table C.5 p.107"
echo "Parameters AMR-B MJD-C p.103-104"
echo "Reference price 3.2412435"
params="--payoff put --exercise american --S 100 --K 100 --T 0.25
        --sigma 0.15 --r 0.05
        --mertonjumps --lambda 0.1 --mertonmu -0.9 --mertongamma 0.45"
./bin/option $1 $params --resolution 2048 --timesteps 128
./bin/option $1 $params --resolution 4096 --timesteps 512
./bin/option $1 $params --resolution 8192 --timesteps 2048
./bin/option $1 $params --resolution 16384 --timesteps 8192
./bin/option $1 $params --resolution 32768 --timesteps 32768
