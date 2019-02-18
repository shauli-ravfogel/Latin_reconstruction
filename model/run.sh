#!/bin/sh

for i in `seq 1 5`;
do
	python3 split_train_test.py $i
	python3 main.py $i #--dynet-autobatch 1
done
echo "Done."
