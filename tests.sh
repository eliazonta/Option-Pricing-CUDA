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

echo "Lippa (2013), table C.2 p.55"
echo "Surkov (2009) reference price: 7.27993383"
params="--payoff call --exercise european --S 100 --K 110 --r 0 --T 1 --sigma 0.2
        --koujumps --lambda 0.2 --p 0.5 --eta1 3 --eta2 2"
./option $params --resolution 512
./option $params --resolution 1024
./option $params --resolution 2048
./option $params --resolution 4096
./option $params --resolution 8192
