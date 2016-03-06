echo "Matlab Closed Form"
echo "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility)"
echo "[42.0707, 2.7238] = blsprice(100, 100, 0.05, 10, 0.15)"
./option --payoff call --resolution 2048
./option --payoff call --resolution 8192
./option --payoff call --resolution 32768
./option --payoff put --resolution 2048
./option --payoff put --resolution 8192
./option --payoff put --resolution 32768

echo ""
echo "Matlab American Option (based on lattice implementation)"
echo "Expected Call price: 42.0707"
./option --payoff call --exercise american --resolution 4096 --timesteps 512
./option --payoff call --exercise american --resolution 8192 --timesteps 1024
./option --payoff call --exercise american --resolution 16384 --timesteps 2048

echo "Expected Put price: 7.0639"
./option --payoff put --exercise american --resolution 4096 --timesteps 512
./option --payoff put --exercise american --resolution 8192 --timesteps 1024
./option --payoff put --exercise american --resolution 16384 --timesteps 2048

echo ""
echo "Lippa (2013), table 2.1 p.14"
echo "Andersen (2000) closed-form reference price: 18.0034"
echo "Surkov (2009) table 2.2 p.33 reference price: 18.00362936"
params="--payoff put --exercise european --S 100 --K 100
        --mertonjumps --gamma 0.4 --lambda 0.1 --dividend 0.02"
./option $params --resolution 2048
./option $params --resolution 4096
./option $params --resolution 8192
./option $params --resolution 16384
./option $params --resolution 32768

echo ""
echo "Lippa (2013), table C.2 p.55"
echo "Surkov (2009), table 6.2 p.92"
echo "Reference price 7.27993383"
params="--payoff call --exercise european --S 100 --K 110 --r 0 --T 1 --sigma 0.2
        --koujumps --lambda 0.2 --p 0.5 --etaUp 3 --etaDown 2"
./option $params --resolution 2048
./option $params --resolution 4096
./option $params --resolution 8192
./option $params --resolution 16384
./option $params --resolution 32768

echo ""
./tests/surkov_c1.sh
echo ""
./tests/surkov_c2.sh
echo ""
./tests/surkov_c4.sh
echo ""
./tests/surkov_c5.sh
echo ""
./tests/surkov_c10.sh
echo ""
./tests/surkov_c11.sh
echo ""
./tests/surkov_24.sh
