echo "Should match data on p.14"
./option --payoff put --exercise european --jumps --dividend 0.02 --resolution 512
./option --payoff put --exercise european --jumps --dividend 0.02 --resolution 1024
./option --payoff put --exercise european --jumps --dividend 0.02 --resolution 2048
./option --payoff put --exercise european --jumps --dividend 0.02 --resolution 4096
./option --payoff put --exercise european --jumps --dividend 0.02 --resolution 8192
