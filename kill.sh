#!/bin/bash
echo -n "please input pid number from:"
read from
echo -n "to:"
read to
for p in $(seq $from $to)
do
    kill $p
done