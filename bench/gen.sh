#! /bin/bash

molname=$1

cp template.json "$molname".json 
sed -i "s/xxx/$molname/g" "$molname".json
