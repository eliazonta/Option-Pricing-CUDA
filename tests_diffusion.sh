#!/bin/bash

# No-jump and jump-diffusion (Merton, Kou) test cases.

echo "Matlab Closed Form"
echo "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility)"
echo "[42.0707, 2.7238] = blsprice(100, 100, 0.05, 10, 0.15)"
./bin/option $1 --payoff call --resolution 2048
./bin/option $1 --payoff call --resolution 8192
./bin/option $1 --payoff call --resolution 32768
./bin/option $1 --payoff put --resolution 2048
./bin/option $1 --payoff put --resolution 8192
./bin/option $1 --payoff put --resolution 32768

echo ""
echo "Matlab American Option (based on lattice implementation)"
echo "Expected Call price: 42.0707"
./bin/option $1 --payoff call --exercise american --resolution 4096 --timesteps 512
./bin/option $1 --payoff call --exercise american --resolution 8192 --timesteps 1024
./bin/option $1 --payoff call --exercise american --resolution 16384 --timesteps 2048

echo "Expected Put price: 7.0639"
./bin/option $1 --payoff put --exercise american --resolution 4096 --timesteps 512
./bin/option $1 --payoff put --exercise american --resolution 8192 --timesteps 1024
./bin/option $1 --payoff put --exercise american --resolution 16384 --timesteps 2048

echo ""
./tests/surkov_c1.sh $1
echo ""
./tests/surkov_c2.sh $1
echo ""
./tests/surkov_c5.sh $1
echo ""
./tests/surkov_c10.sh $1
echo ""
./tests/surkov_22.sh $1
echo ""
./tests/surkov_62.sh $1
