#!/bin/bash

# usage:
# ./createAll

rm -rf bin pylib
b2 && mv pylib/py_ped*.so /usr/local/home/xiong/00git/research/lib/boostPython/
echo "compile pyPED"
echo 


