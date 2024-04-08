# 实验平台

## 平台配置信息

人工智能平台 PAI（原机器学习平台）DSW，1 * NVIDIA V100

## 硬件配置查询

```shell
import subprocess

try:
    subprocess.run('nvidia-smi', shell=True)
except Exception as e:
    print('No NVIDIA GPU found or nvidia-smi is not installed.')
```

# 更新镜像源

此步可省，有时可能出现依赖包安装不上等问题

```shell
cd /etc/apt
sudo cp sources.list sources.list.backup

# 修改镜像源
sudo vim sources.list
```

**注：**[镜像源网址](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)，将该网址覆盖 *sources.list* 文件

```shell
sudo apt update
# 检查可更新列表
sudo apt upgrade
# 执行更新
```

# ChatGLM3代码下载

```shell
git clone https://github.com/THUDM/ChatGLM3
cd ChatGLM3
```

# 隔离环境搭建

```shell
conda create -n chatglm3 python==3.10
```

可能会出现如下报错

```shell
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): failed

UnavailableInvalidChannel: HTTP 404 NOT FOUND for channel anaconda/pkgs/main <http://mirrors.aliyun.com/anaconda/pkgs/main>

The channel is not accessible or is invalid.

You will need to adjust your conda configuration to proceed.
Use `conda config --show channels` to view your configuration's current state,
and use `conda config --show-sources` to view config file locations.
```

可通过如下代码增加通道

```shell
conda config --show channels
conda config --remove channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```

若依旧报错（如下）

```shell
Collecting package metadata (current_repodata.json): / Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='mirrors.tuna.tsinghua.edu.cn', port=443): Read timed out. (read timeout=9.15)")': /anaconda/pkgs/main/noarch/current_repodata.json
```

则删除 */root/.condarc* 文件当中所有内容，并保存退出

随后再次创建环境，并安装模型依赖包

```shell
conda activate chatglm3
python -m pip install --upgrade pip

cd /mnt/workspace/ChatGLM3/
pip install -r requirements.txt
```

# ChatGML3模型下载

**严重注意：**阿里云不支持huggingface！！！所以此处使用的是魔搭

```shell
cd /mnt/workspace
git lfs install
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

# 模型验证

```shell
cd basic_demo/
cp cli_demo.py cli_demo.py.bak
```

修改文件内容

```python
vi cli_demo.py

# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
MODEL_PATH = os.environ.get('MODEL_PATH', '/mnt/workspace/chatglm3-6b')
```

使用GPU进行计算，需在cli_demo.py中添加如下code

```python
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()
model = model.to(device)
```

# Jupyter启动模型

```shell
pip install modelscope
conda install ipykernel
python -m ipykernel install --user --name=chatglm3 --display-name="python(chatglm3)"
jupyter lab
```

试用如下code进行测试

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/chatglm3-6b', trust_remote_code=True)
model = AutoModel.from_pretrained('/mnt/workspace/chatglm3-6b', trust_remote_code=True, device='cuda')
model = model.eval()

response, history = model.chat(tokenizer, "晚上好", history=[])
print(response)

response, history = model.chat(tokenizer, "压力太大了怎么办", history=history)
print(response)
```

# 高效微调参考

## 数据集准备

此处首先使用一个官方示例的数据集，[下载网址](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view)此为广告数据集

```shell
# 在/mnt/workspace/ChatGLM3/finetune_demo目录下创建data目录
mkdir data
# 将tar.gz文件上传至此然后解压
tar -xzvf AdvertiseGen.tar.gz
# 安装相关依赖包
pip install -r requirements.txt
```

在当前目录运行如下py code，将数据集进行切割

```python
import json
from typing import Union
from pathlib import Path


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


convert_adgen('data/AdvertiseGen', 'data/AdvertiseGen_fix')
```

## 修改相关配置文件

该模型的官方文档中的 *lora.yaml* 使用后会导致爆显卡，所以使用下面的内容平替该 *yaml* 文件

```yaml
data_config:
  train_file: train.json
  val_file: dev.json
  test_file: dev.json
  num_proc: 16
max_input_length: 128
max_output_length: 256
training_args:
  # see `transformers.Seq2SeqTrainingArguments`
  output_dir: ./output
  max_steps: 3000
  # settings for data loading
  per_device_train_batch_size: 1
  dataloader_num_workers: 16
  remove_unused_columns: false
  # settings for saving checkpoints
  save_strategy: steps
  save_steps: 500
  # settings for logging
  log_level: info
  logging_strategy: steps
  logging_steps: 10
  # settings for evaluation
  per_device_eval_batch_size: 16
  evaluation_strategy: steps
  eval_steps: 500
  # settings for optimizer
  # adam_epsilon: 1e-6
  # uncomment the following line to detect nan or inf values
  # debug: underflow_overflow
  predict_with_generate: true
  # see `transformers.GenerationConfig`
  generation_config:
    max_new_tokens: 256
  # set your absolute deepspeed path here
  #deepspeed: ds_zero_2.json
  # set to true if train with cpu.
  use_cpu: false
peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
```

## 执行微调

查看本项目所创建虚拟环境python的绝对路径

```shell
which python
```

在jupyter中执行下面code，需要将上述shell结果以参数形式放入下面的cell中

```python
!CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/chatglm3/bin/python finetune_hf.py  data/AdvertiseGen_fix  /mnt/workspace/chatglm3-6b  configs/lora.yamlti
```

**注：**如果只有一块显卡，要将 *CUDA_VISIBLE_DEVICES* 置为0（1时仅使用cpu进行计算）

# 使用心里健康数据微调模型

## 数据集准备

通过[下载链接](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/tree/main)可下载本节使用的数据集，由于阿里云不支持访问 *hugging face* ，所以需要下载到本地，然后通过上传的方式到服务器

使用以下code将一份完整的心理健康数据json文件分成两个部分 *dev.json train.json* 

```python
import json


data_list = [
    json.loads(line) for line in open('./MeatalHealth/combined_dataset.json', 'r', encoding='utf-8')
]

# 此处比例参考前面的广告数据集比例
split_ratio = 0.00934
split_point = int(len(data_list) * split_ratio)

data_part1 = data_list[:split_point]
data_part2 = data_list[split_point:]
print(len(data_part1), len(data_part2))

with open('./MeatalHealth/dev.json', 'w', encoding='utf-8') as file:
    json.dump(data_part1, file, ensure_ascii=False, indent=4)
with open('./MeatalHealth/train.json', 'w', encoding='utf-8') as file:
    json.dump(data_part2, file, ensure_ascii=False, indent=4)
```

将 *lora_finetune.ipynb* 中的切割数据集修改为如下code

```python
import json
from typing import Union
from pathlib import Path


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['Context']},
                                                {'role': 'assistant', 'content': dct['Response']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


convert_adgen('data/MeatalHealth', 'data/MeatalHealth_fix')
```

**注：**主要注意更换*json*文件目录，以及该文件的用户和回答的*keys*是否更换

## 执行微调

使用如下程序进行微调

```python
!CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/chatglm3/bin/python finetune_hf.py  data/MeatalHealth_fix  /mnt/workspace/chatglm3-6b  configs/lora.yaml
```

**注：**需要更改*lora.yaml*当中的部分参数，否则可能运行困难

