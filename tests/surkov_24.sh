echo "Surkov (2009) table 2.4 p.43"
echo "Parameters AMR-A CGMY-A p.103-104"
echo "Reference price 9.2254803"
params="--payoff put --exercise american --S 90 --K 98 --T 0.25
        --CGMY --C 0.42 --G 4.37 --M 191.2 --Y 1.9102 --r 0.06"
./option $params --resolution 4096 --timesteps 512
./option $params --resolution 8192 --timesteps 1024
./option $params --resolution 16384 --timesteps 2048
