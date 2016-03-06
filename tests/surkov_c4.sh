#!/bin/bash

echo "Surkov (2009) table C.4 p.107"
echo "Parameters EUR-F and CGMY-B p.103-104"
echo "Reference price 4.3714972 and 4.38984331"
params="--payoff put --exercise european --S 10 --K 10 --T 0.25
        --CGMY --C 1.0 --G 8.8 --M 9.2 --Y 1.8 --r 0.1"
./option $1 $params --resolution 2048
./option $1 $params --resolution 4096
./option $1 $params --resolution 8192
./option $1 $params --resolution 16384
./option $1 $params --resolution 32768
