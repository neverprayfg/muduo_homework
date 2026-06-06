
模型后端默认是cpu
./muduoXinyu  ./data/stories110M.bin ./data/tokenizer.bin "once upon a time," 

./muduoXinyu  ./data/stories110M.bin ./data/tokenizer.bin "once upon a time,"  --backend cpu --skipValidation

./muduoXinyu  ./data/stories110M.bin ./data/tokenizer.bin "once upon a time,"  --backend npu --enableDeviceOpt 

./muduoXinyu  ./data/stories110M.bin ./data/tokenizer.bin "once upon a time,"  --backend npu --skipValidation --enableDeviceOpt 