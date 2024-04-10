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

**注：** [镜像源网址](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)，将该网址覆盖 *sources.list* 文件

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

**严重注意：** 阿里云不支持huggingface！！！所以此处使用的是魔搭

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

**注：** 如果只有一块显卡，要将 *CUDA_VISIBLE_DEVICES* 置为0（1时仅使用cpu进行计算）

# 使用心里健康数据微调模型

## 数据集准备

通过[下载链接](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/tree/main)可下载本节使用的数据集，由于阿里云不支持访问 *hugging face* ，所以需要下载到本地，然后通过上传的方式到服务器

使用以下code将一份完整的心理健康数据json文件分成两个部分 *dev.json train.json* 

```python
import json


with open('./MeatalHealth/combined_dataset.json', 'r') as f:
    lines = f.readlines()

first_file_lines = lines[:300]
remaining_lines = lines[300:]

with open('./MeatalHealth/dev.json', 'w') as f:
    for line in first_file_lines:
        f.write(line)

with open('./MeatalHealth/train.json', 'w') as f:
    for line in remaining_lines:
        f.write(line)
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

**注：** 主要注意更换*json*文件目录，以及该文件的用户和回答的*keys*是否更换

## 执行微调

使用如下程序进行微调

```python
!CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/chatglm3/bin/python finetune_hf.py  data/MeatalHealth_fix  /mnt/workspace/chatglm3-6b  configs/lora.yaml
```

**注：** 需要更改*lora.yaml*当中的部分参数，否则可能运行困难

## 使用微调数据集进行推理

双引号中内容可进行适当更改

```python
!CUDA_VISIBLE_DEVICES=0  /opt/conda/envs/chatglm3/bin/python inference_hf.py output/checkpoint-3000/ --prompt "Due to exam preparation causing high stress, I have insomnia at night."
```

# 微调结果对比

此处未进行模型合并，为测试微调前和微调后的模型差异。微调前chat效果通过 *terminal* 中调用 */mnt/workspace/ChatGLM3/basic_demo/cli_demo.py* 的方式进行提问，微调后则通过 *prompt* 推理的方式进行提问测试。

通过如下方式在 *terminal* 其中基础预训练模型，然后直接输入进行测试即可

```shell
cd /mnt/workspace/ChatGLM3/basic_demo
python cli_demo.py
```

通过如下方式在 *jupyter* 中通过 *prompt* 推理的方式，测试预训练后的模型

```python
!CUDA_VISIBLE_DEVICES=0  /opt/conda/envs/chatglm3/bin/python inference_hf.py output/checkpoint-3000/ --prompt "Due to exam preparation causing high stress, I have insomnia at night."
```

**注：** 更换双引号内中的内容即可；另外，每次在 *terminal* 中询问结束后需要关闭后再使用 *prompt* 推理

## 心理学的问题

### 预训练前

提问

```context
I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here.\n   I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.\n   How can I change my feeling of being worthless to everyone?
```

回答

```context
I'm sorry to hear that you're going through a tough time. It's important to remember that you are valuable and worthy of love and respect. Here are some suggestions that may help you change your feeling of being worthless:

1. Practice self-compassion: Treat yourself with kindness and understanding. Recognize that everyone makes mistakes and experiences setbacks. Remind yourself that it's okay to feel the way you do, and that you can learn and grow from these experiences.

2. Focus on your strengths: Identify your positive qualities and accomplishments. Acknowledge the things that you do well and that make a difference. Try to build on these strengths rather than dwelling on your weaknesses.

3. Challenge negative thoughts: Be aware of when you're thinking negative thoughts about yourself. Try to reframe these thoughts in a more positive light. For example, instead of thinking "I'm a failure," try to think "I made a mistake, but I'm learning from it and getting better."

4. Seek support: Talk to people you trust about how you're feeling. This could be a friend, family member, or mental health professional. They can provide you with support, encouragement, and valuable advice.

5. Practice self-care: Take care of yourself physically, emotionally, and mentally. This might include getting enough sleep, eating a healthy diet, exercising regularly, and finding activities that bring you joy and relaxation.

Remember that changing your feelings takes time and effort. Be patient with yourself and keep in mind that small steps can lead to big changes over time.
```

提问

```context
我正在经历一些与我的感受和自己有关的事情。我几乎不睡觉，除了想着我是多么的毫无价值，以及我不应该存在。我从未尝试过或思考过自杀。我一直想要解决我的问题，但我总是没能付诸实践。我该如何改变我对每个人都觉得自己毫无价值的感觉？
```

回答

```context
ChatGLM：首先，我想告诉你，你并不孤独，许多人都可能有过类似的感受。要改变这种感觉，你可以尝试以下几个方法：

1. 自我接纳：接受自己的优点和不足，认识到每个人都有自己的价值。不要过于苛责自己，要明白每个人都有不完美的地方。

2. 增强自信：多关注自己的成就和优点，不要总是拿自己的缺点和别人的优点比较。可以尝试学习新技能、参加社交活动来提升自信。

3. 与他人沟通：和亲朋好友分享你的感受，寻求他们的支持和建议。有时候，倾诉可以让你感到轻松，甚至能帮助你找到解决问题的方法。

4. 专业帮助：如果你觉得自己无法摆脱这种感觉，建议寻求心理专业人士的帮助，如心理咨询师或心理医生。他们可以为你提供专业的意见和指导。

5. 健康生活：保持良好的作息习惯，保证充足的睡眠，注意饮食和锻炼。身体健康很大程度上会影响到心理状态。

6. 找到自己的兴趣和目标：确定一个自己感兴趣的领域或目标，投身于其中，这会让你更有价值感。

请记住，改变这种感觉需要时间和耐心，不要期望立竿见影。尝试以上方法，并保持积极的心态，逐步改善自己的心理状态。
```

### 预训练后

提问

```context
I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here.\n   I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.\n   How can I change my feeling of being worthless to everyone?
```

回答

```context
First of all, you should not feel like you should not be here. I don't know what kind of "things" you are going through, but there is a good chance that your life is not as difficult as you think it is. You are not alone in this. I know of many people who feel this way. There are many people who have a lot of things to do and are successful in their lives. You are one of them.
```

提问

```context
我正在经历一些与我的感受和自己有关的事情。我几乎不睡觉，除了想着我是多么的毫无价值，以及我不应该存在。我从未尝试过或思考过自杀。我一直想要解决我的问题，但我总是没能付诸实践。我该如何改变我对每个人都觉得自己毫无价值的感觉？
```

回答

```context
首先，你能够意识到自己有这种感觉，已经是一个好的开始。 在考虑自杀的时候，先尝试去了解你自己的内心，了解你真正想要什么，以及你想要什么。 有时候，我们想要自杀，是因为我们害怕死亡，害怕失去我们的亲人和朋友，害怕失去我们自己的身份和角色。 有时候，我们想要自杀，是因为我们害怕失去自己的价值，害怕自己不再被接受，被拒绝。 对于你所说的问题，我建议你尝试以下方法： 1. 在你能够安静下来的时候，试着去了解你自己的内心，了解你真正想要什么。 2. 尝试和一些你信任的朋友，家人，同事交流。 3. 如果你还是不能解决你的问题，可以考虑找心理医生，心理治疗师，他们可以提供更好的帮助。 希望这些能够对你有所帮助，祝你好运！
```

## 心理学交叉的问题

### 预训练前

提问

```context
How to optimize teaching methods and educational environments to promote students' learning and development?
```

回答

```context
Optimizing teaching methods and educational environments can have a significant impact on promoting students' learning and development. Here are some strategies that can be effective:

1. Make learning engaging: Students are more likely to be engaged in the learning process when it is interactive, hands-on, and relevant to their lives. Incorporate technology, multimedia, and real-world examples to make learning more engaging.
2. Personalize learning: Every student learns differently, so it is important to personalize the learning experience for each student. This can be achieved by providing different learning paths, allowing for student choice and voice in their learning, and differentiating instruction based on student needs.
3. Use active learning strategies: Encourage students to take an active role in their own learning by using strategies such as problem-solving, critical thinking, and collaborative learning.
4. Foster a positive learning environment: A positive learning environment can promote student engagement and motivation. This can be achieved by creating a culture of respect, kindness, and support, and by fostering a sense of community among students.
5. Provide ongoing feedback: Regular feedback can help students understand their progress, identify areas for improvement, and build confidence. It is important to provide both formative and summative feedback to help students learn and grow.
6. Use assessment to support learning: Assessment should be used to support learning and not just to evaluate students. It can be used to identify student strengths and weaknesses, provide opportunities for student reflection, and inform instruction.
7. Continuously improve teaching methods and environments: Finally, it is important to continuously evaluate and improve teaching methods and educational environments. This can be done by seeking feedback from students and colleagues, trying new strategies, and using data to inform decision-making.
```

提问

```context
如何优化教学方法和教育环境以促进学生的学习和发展？
```

回答

```context
优化教学方法和教育环境对于促进学生的学习和发展非常重要。以下是一些可以采取的策略：

1. 提高学习参与度：通过采用互动性、实践性、相关性的学习方式，使学生更有可能参与学习过程。可以运用技术、多媒体和实际例子来使学习更加吸引人。
2. 个性化学习：每个学生都以不同的方式学习，因此重要的是为学生提供个性化的学习体验。这可以通过为学生提供不同的学习路径、允许学生在自己的学习过程中拥有选择权和发言权，以及根据学生的需要进行差异化的教学来实现。
3. 使用主动学习策略：鼓励学生积极参与自己的学习过程，通过采用诸如问题解决、批判性思考和合作学习等策略来实现。
4. 创造积极的学习环境：一个积极的学习环境可以促进学生的参与和动力。这可以通过创建尊重、善意和支持的文化，以及培养学生之间的社区意识来实现。
5. 提供持续的反馈：定期的反馈可以帮助学生了解自己的进展情况、确定需要改进的领域，并建立自信。重要的是要提供形式性和总结性反馈，以帮助学生学习和成长。
6. 使用评估支持学习：评估应该是为了支持学习而进行的，而不是为了评估学生。它可以用来识别学生的强项和弱点、为学生提供反思的机会，并指导教学。
7. 不断改进教学方法和环境：最后，不断评估和改进教学方法和环境非常重要。这可以通过向学生和同事寻求反馈、尝试新的策略，并使用数据来指导决策来实现。
```

### 预训练后

提问

```context
How to optimize teaching methods and educational environments to promote students' learning and development?
```

回答

```context
The key to promoting students' learning and development is to create an environment that is conducive to their learning. This means creating a physical environment that is well lit, well ventilated, and free of distractions. The learning environment should also be organized and structured to promote engagement. The learning environment should be designed to encourage collaboration, communication, and critical thinking. This can be achieved by creating opportunities for group work, encouraging participation, and providing opportunities for students to explore different ways of thinking. Additionally, providing opportunities for hands-on learning can help students understand concepts more effectively. The use of technology can also be beneficial. Technology can help students connect with other students, provide opportunities for hands-on learning, and enhance their understanding of concepts.
```

提问

```context
如何优化教学方法和教育环境以促进学生的学习和发展？
```

回答

```context
优化教学方法和教育环境以促进学生的学习和发展是教育工作者和学者们长期探讨的问题。以下是一些建议:

1. 采用多元化的教学方法:根据不同的学科和年级,采用不同的教学方法,如小组讨论、案例分析、实验、游戏等,让学生在不同的情境下学习,提高学生的学习兴趣和参与度。

2. 注重个性化教学:了解每个学生的学习风格和特点,针对不同的学生提供个性化的教学支持和指导,提高学生的学习效果和自信心。

3. 创造积极的学习氛围:为学生提供良好的学习环境和资源,鼓励学生互相合作和学习,创造积极的学习氛围,促进学生的学习和发展。

4. 培养学生的思维能力:通过提问、挑战和解决问题的方法,培养学生的批判性思维和创新能力,让学生学会独立思考和解决问题,提高学生的学习效果和适应能力。

5. 注重实践和应用:将教学内容与实际生活和社会实践相结合,让学生通过实践和应用来理解和掌握知识,提高学生的学习效果和实用能力。

6. 采用科技手段:利用科技手段,如在线教学平台、多媒体教学工具等,提高教学效率和效果,促进学生的学习和发展。

7. 注重教师的专业素养:提高教师的专业素养和教育水平,让学生得到更好的教育和支持,促进学生的学习和发展。

以上是一些建议,希望能对你有所帮助。
```

## 与心理学无关的问题

### 预训练前

提问

```context
How is Porsche?
```

回答

```context
Porsche is a German sports car manufacturer that is known for producing high-performance cars, sedans, SUVs, and hybrid vehicles. The company was founded in 1900 and is headquartered in Stuttgart, Germany. Porsche's products are known for their exceptional performance, advanced technology, and elegant design. The company's product line includes sports cars, sedans, SUVs, and hybrid vehicles.

Porsche is highly regarded globally and its vehicles are usually outstanding in various car evaluations. The company has also been actively involved in developing innovative technology and design, such as hybrid systems and autonomous driving technology.

Overall, Porsche is a reputable car manufacturer with a long history, excellent reputation, and high-quality products. If you are interested in high-performance cars, Porsche may be a brand worth considering.
```

提问

```context
保时捷怎么样？
```

回答

```context
保时捷（Porsche）是一家德国汽车制造商，以生产高性能的跑车、轿车和SUV而闻名。该公司成立于1900年，总部位于德国斯图加特。保时捷的产品以其卓越的性能、先进的技术和精美的设计而著称。它的车型系列包括跑车、轿车、SUV和混合动力汽车等。

保时捷汽车在全球范围内享有很高的声誉，其产品在各种汽车评测中通常表现优异。该公司也一直致力于开发创新的技术和设计，如混合动力系统和自动驾驶技术等。

总的来说，保时捷是一家具有悠久历史、卓越声誉和优质产品的汽车制造商。如果您对高性能汽车感兴趣，保时捷可能是您值得考虑的品牌之一。
```

### 预训练后

提问

```context
How is Porsche?
```

回答

```context
Porsche is a luxury car brand that is well known for its high-performance sports cars. The company was founded in Germany in 1945 and has since become one of the most successful and prestigious car brands in the world. Porsche is known for its innovative technology and attention to detail, and its cars are highly sought after by collectors and enthusiasts. Do you have any other questions about Porsche?
```

提问

```context
保时捷怎么样？
```

回答

```context
保时捷（Porsche）是一家德国汽车制造商，成立于1900年，总部位于斯图加特。保时捷以生产高性能跑车、轿车和SUV而闻名，其产品以驾驶体验、科技配置和优雅的外观而受到消费者的喜爱。

保时捷汽车在性能、操控和安全性方面表现出色，其生产的车辆在各种赛事中屡获佳绩。此外，保时捷还推出了多款混合动力和纯电动车型，为消费者提供了更多的选择。

在我国，保时捷汽车也有一定的市场地位，很多消费者对保时捷品牌有着较高的认可度。然而，与其他高档汽车品牌一样，保时捷的维修保养费用相对较高，保养和维修成本较高。

总的来说，保时捷是一家有着良好口碑的汽车制造商，如果你喜欢驾驶体验和科技配置，且预算充足，那么保时捷汽车是一个不错的选择。
```

## 小节

由于微调数据集中全为英文数据集，所以对以上回答中的英文提问的回答差异程度较为明显。另外，由于预训练数据集中的回答数据集可能多为未进分段，故在微调后模型的英文回答中，也大概率为一个自然段(如果多尝试几次也会有很长的分段回答，但是内容巨多，模型分了30点回答 *心理学交叉* 的问题)。

# 总结

通过该项目，简单复习了一下 *ubuntu* 的相关操作，*conda* 命令相关报错的处理，*json* 文件格式的保存与读取中存在的坑， *LLM* (此处使用ChatGLM3-6b)的本地部署，*LLM* 的微调，相关NLP开源数据集的开放网址，*GPU* 资源不够时减小训练参数量的方法，如何辨别模型是否进行了微调，以及如何使用微调后的文件继续推理，另外简单参考了一下模型合并(即将预训练好的模型更新本地已部署好的模型)

