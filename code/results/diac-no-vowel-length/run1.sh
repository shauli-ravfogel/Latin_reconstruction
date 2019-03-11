#!/bin/bash          

# trained with linear = False
         
dataset="orto-no-diac"
batch=5
optimizer="cyclic"
model_type="lstm"
model_size=100
dropout=0.1
running_id=1;

python3 create_dataset.py 1 $dataset;

sleep 5;
python3 main.py --running_id $running_id --model_size $model_size --batch_size $batch --dropout $dropout --optimizer $optimizer --network $model_type --dynet-autobatch 1
