#!/bin/bash

echo "Surkov (2009) table C.7 p.108"
echo "Parameters AMR-C and VG-B p.103-104"
echo "Reference price 35.5301"
params="--payoff put --exercise american --S 1369.1 --K 1200 --T 0.5616
        --sigma 0.20722 --r 0.0541 --dividend 0.012
        --vg --vgmu 0.50215 --vgdrift -0.22898"
./option $1 $params --resolution 2048 --timesteps 128
./option $1 $params --resolution 4096 --timesteps 512
./option $1 $params --resolution 8192 --timesteps 2048
./option $1 $params --resolution 16384 --timesteps 8192
./option $1 $params --resolution 32768 --timesteps 32768
