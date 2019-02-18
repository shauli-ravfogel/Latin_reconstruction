python calc_acc.py 
mkdir -p results
scp log.txt results
for i in `seq 1 5`;
do
	scp preds$i.txt results
	scp train$i.txt results
	scp dev$i.txt results
	scp test$i.txt results
done
