#!/bin/bash

echo ""
echo "Lippa (2013), table 2.1 p.14"
echo "Andersen (2000) closed-form reference price: 18.0034"
echo "Surkov (2009) table 2.2 p.33 reference price: 18.00362936"
params="--payoff put --exercise european --S 100 --K 100
        --mertonjumps --mertonmu=-1.08 --mertongamma 0.4 --lambda 0.1 --dividend 0.02"
./option $1 $params --resolution 2048
./option $1 $params --resolution 4096
./option $1 $params --resolution 8192
./option $1 $params --resolution 16384
./option $1 $params --resolution 32768
