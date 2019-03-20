#!/bin/bash          
            
dataset="orto"
batch=10
optimizer="adam"
model_type="lstm"
model_size=150
dropout=0.2
running_id=4;

python3 create_dataset.py $running_id $dataset;

sleep 5;
python3 main.py --running_id $running_id --model_size $model_size --batch_size $batch --dropout $dropout --optimizer $optimizer --network $model_type --dynet-autobatch 1 --dynet-mem 1024
