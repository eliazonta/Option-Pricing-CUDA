#!/bin/bash

# Pure jump (Variance Gamma, CGMY) test cases.
echo ""
./tests/surkov_c3.sh $1
echo ""
./tests/surkov_c4.sh $1
echo ""
./tests/surkov_c7.sh $1
echo ""
./tests/surkov_c11.sh $1
echo ""
./tests/surkov_24.sh $1
echo ""
./tests/wwf_2.sh $1
echo ""
./tests/wwf_3.sh $1
