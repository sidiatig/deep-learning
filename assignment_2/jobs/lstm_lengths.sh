#!/bin/bash

for i in `seq 4 4 24 `; do
  python -u train.py --model_type="LSTM" --input_length=$i --learning_rate=0.0001 &
done
wait
