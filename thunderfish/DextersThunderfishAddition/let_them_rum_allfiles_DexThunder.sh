#!/bin/bash

cat allfiles.txt | while read -r line
    do
	python3 -W ignore DextersThunderfishAddition.py "${line:0:-4}/$line"
    done
