echo "Surkov (2009) table C.11 p.109"
echo "Parameters AMR-C CGMY-A p.103-104"
echo "Reference price 47.113217736"
params="--payoff put --exercise american --S 1369.41 --K 1200 --T 0.5616
        --CGMY --C 0.42 --G 4.37 --M 191.2 --Y 1.0102 --r 0.06"
./option $params --resolution 4096 --timesteps 512
./option $params --resolution 8192 --timesteps 1024
./option $params --resolution 16384 --timesteps 2048
