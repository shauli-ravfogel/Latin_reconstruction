#!/bin/sh

for dataset in orto-no-diac ipa-no-diac;
do
	python3 create_dataset.py 1 $dataset
	
    for model_type in gru lstm;
	do
	
	for model_size in 75 100 150 200;
	
	do
		for optimizer in adam cyclic adagrad;
		
		do
		
			for batch in 1 10 20 50;
			
			do
			
				for dropout in 0 0.1 0.2 0.3 ;
				do
					NUMBER=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 6)
					if [ "$NUMBER" == "" ]; then
  						NUMBER=0
					fi
					
					for i in `seq 1 4`;
					
					do
					
						python3 main.py --running_id $i --model_size $model_size --batch_size $batch --dropout $dropout --optimizer $optimizer --network $model_type --dynet-autobatch 1

					done
					
					
					python3 calc_acc.py
					mkdir -p results/$dataset/$model_type/$model_size/$optimizer/$batch/$dropout
					for i in `seq 1 4`;
						do
							scp preds$i.txt results/$dataset/$model_type/$model_size/$optimizer/$batch/$dropout
							scp model.m.$i results/$dataset/$model_type/$model_size/$optimizer/$batch/$dropout
						done
						
						echo -e "\ndataset="  $dataset  "\nmodel_size = "  $model_size  "\nbatch = "  $batch  "\noptimizer = "  $optimizer  "\ndropout = "  $dropout >> log.txt 
						scp log.txt results/$dataset/$model_type/$model_size/$optimizer/$batch/$dropout
				
				done
			
			done
		
		done
	done
    done
done

