# README.txt
---
* 编译源代码
mkdir build
cd build
cmake .. -DMKL=1（使用MKL加速， 否则-DMKL=0）


* 运行文件
run.sh
    * 创建词典  
./biaffine-parser config.txt --train=1 --test=0 --dictionary-exist=0 --train-file= --dev-file=
    * 训练  
./biaffine-parser config.txt --train=1 --test=0 --dictionary-exist=1 --train-file= --dev-file= --test-file --thread-num=1 --task=0  
    
    task=0 use-train2=0 是单个Parser训练  
    task=0 use-train2=1 是multi-learning Parser训练  
    task=1 use-train2=0 是转化模型训练   

    * 测试  
./biaffine-parser config.txt --train=0 --test=1   
--thread-num=1 --task=0 --inst-max-num-eval=-1   
--test-file=  -output-file=   


* 配置文件  
config.txt 主要放控制程序流程的选项，命令行的选项值可以覆盖文件中的选项值  
option.txt 主要放神经网络的参数选项  




