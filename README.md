# muduoXinyu推理框架
muduoXinyu是轻量级模块化的LLM推理引擎，支持CPU、GPU设备并适配华为NPU和海光DCU两款国产处理器上的高效推理
## 编译：  
### CPU
编译方式：
```bash 
cd muduoXinyu  
make
```
### GPU
编译方式：
```bash   
cd muduoXinyu  
make cuda
```
### DCU
编译方式：
```bash   
cd muduoXinyu  
make dcu
```
如果在超算互联网平台（https://www.scnet.cn/） 中提交，可以直接执行
```bash   
sbatch muduoXinyu.sh
```
注意需修改`muduoXinyu.sh`脚本文件中
```bash   
#SBATCH -p 队列名称

......

cd /work/home/用户名/muduoXinyu
```
替换为自己的队列名称和用户名  
查看当前可用队列名称可使用命令
```bash   
whichpartition
```
### NPU
编译方式：
```bash   
cd muduoXinyu  
make npu
```
## 运行
1. Usage
```bash
./muduo 模型路径 分词器路径 提示词/提示词文件路径
```
可选参数：  
--backend cpu|cuda|dcu|npu  指定运行后端  
--skipValidation            跳过验证  
--enableDeviceOpt           启用针对设备优化  
*如使用npu后端请打开`--enableDeviceOpt`选项，且由于计算精度问题，验证时和校验文件有所差别，推荐打开`--skipValidation`选项跳过验证  
2. 示例 
```bash
#提示词
./muduoXinyu  data/stories110M.bin data/tokenizer.bin "once upon a time,"
#提示词文件路径
./muduoXinyu  data/stories110M.bin data/tokenizer.bin data/input_prompt.txt
#启用cuda后端
./muduoXinyu  data/stories110M.bin data/tokenizer.bin data/input_prompt.txt --backend cuda
```