#/bin/bash

home=$HOME
datap=/data/xzjiang/tree-convert-2018.2.2/data
train=$datap/hlt-cdt-train.conll
train2=$datap/cdt2-train-ctb-pos.conll
dev=$datap/hlt-cdt-dev.conll
test=$datap/hlt-cdt-test.conll

out=test.out.conll

exe=/data/xzjiang/tree-conversion/code

parser="$exe/biaffine-parser config.txt pat-c crf-loss"
#$parser --train=1 --test=0 --dictionary-exist=0 --train-file=$train --dev-file=$dev > log.create-model 2>&1 
$parser  --thread-num=1  --task=1 --train=1 --test=0 --dictionary-exist=1 --train-file=$train  --dev-file=$dev --test-file=$test > log.train 2>&1  
#$parser --train=0 --test=1 --thread-num=1 --task=0 --inst-max-num-eval=-1  --test-file=$test --output-file=$out  > log.test 2>&1 
