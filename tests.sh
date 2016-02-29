echo "Matlab Closed Form"
echo "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility)"
echo "[42.0707, 2.7238] = blsprice(100, 100, 0.05, 10, 0.15)"
./option --payoff call --resolution 512
./option --payoff call --resolution 2048
./option --payoff call --resolution 8192
./option --payoff put --resolution 512
./option --payoff put --resolution 2048
./option --payoff put --resolution 8192

echo ""
echo "Matlab American Option (based on lattice implementation)"
echo "Call price: 42.0707"
./option --payoff call --exercise american --resolution 2048 --timesteps 10
./option --payoff call --exercise american --resolution 2048 --timesteps 100
./option --payoff call --exercise american --resolution 2048 --timesteps 1000
./option --payoff call --exercise american --resolution 8192 --timesteps 2000

echo "Put price: 7.0639"
./option --payoff put --exercise american --resolution 2048 --timesteps 10
./option --payoff put --exercise american --resolution 2048 --timesteps 100
./option --payoff put --exercise american --resolution 2048 --timesteps 1000
./option --payoff put --exercise american --resolution 8192 --timesteps 2000

echo ""
echo "Lippa (2013), table 2.1 p.14"
echo "Andersen (2000) closed-form reference price: 18.0034"
echo "Surkov (2009) reference price: 18.00362936"

params="--payoff put --exercise european --S 100 --K 100
        --mertonjumps --gamma 0.4 --lambda 0.1 --dividend 0.02"
./option $params --resolution 512
./option $params --resolution 1024
./option $params --resolution 2048
./option $params --resolution 4096
./option $params --resolution 8192

echo ""
echo "Lippa (2013), table C.2 p.55"
echo "Surkov (2009) reference price: 7.27993383"
params="--payoff call --exercise european --S 100 --K 110 --r 0 --T 1 --sigma 0.2
        --koujumps --lambda 0.2 --p 0.5 --eta1 3 --eta2 2"
./option $params --resolution 512
./option $params --resolution 1024
./option $params --resolution 2048
./option $params --resolution 4096
./option $params --resolution 8192
