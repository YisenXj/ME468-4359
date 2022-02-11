#!/bin/bash
for i in {9..13}
do
   python3 mat-vecMult.py $((2**$i)) 
   echo "" 
done
