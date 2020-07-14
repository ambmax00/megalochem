#! /bin/bash

molname=$1

cp template.in "$molname".json 
sed -i "s/xxx/$molname/g" "$molname".json
