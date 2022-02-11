#!/bin/bash
for i in {9..13}
do
   python3 Q4data.py $((2**$i)) 
done
