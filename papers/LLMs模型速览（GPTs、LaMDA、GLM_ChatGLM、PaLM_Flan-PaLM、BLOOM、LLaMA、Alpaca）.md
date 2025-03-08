@[toc]

>参考:
>- [《总结从T5、GPT-3、Chinchilla、PaLM、LLaMA、Alpaca等近30个最新模型》](https://blog.csdn.net/qq_27590277/article/details/130256877)
>- [LLaMA、Palm、GLM、BLOOM、GPT模型结构对比](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490555&idx=2&sn=ff61b482d34095877f83ee213dbb4724&chksm=9bc5f3a9acb27abff5fd1e55dc6e55c810957fb0d2af5b254127a8866c305e39ff4bdb035a3e&scene=178&cur_album_id=2878066965444362241#rd)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dd6e36453bf04d7545cba1fc6d9d501c.png#pic_center =600x)
基础模型：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/309036a8e0e9a117bd960bac8cb09617.png#pic_center =600x)
下表是在上述基础模型上进行指令微调的大模型：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0207a0f2d922857d992bedaa8624d123.png#pic_center =600x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3d405995d6bf5539ae29aeee7ad235c0.png#pic_center =600x)
在[datalearner.com](https://www.datalearner.com/ai-models/pretrained-models?&aiArea=-1&language=-1&contextLength=-1&openSource=-1&publisher=-1)上，可以查看所有已发布的AI大模型：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7a489bed74c08202109e0a4d61da4829.png)

## 一、  GPT系列
| 模型 | 发布日期 |
|---|---|
| GPT | 2018-11-14 |
| GPT-2 | 2019-11-27 |
| GPT-3 | 2020-6-11 |
| InstructGPT | 2022-3-4 |
| ChatGPT | 2022-11-30 |
| GPT-4 | 2023-3-14 |
| ChatGPT Plugin | 2023-5-12 |
### 1.1 GPTs（OpenAI，2018——2020）
>- [《【LLM系列之GPT】》](https://mp.weixin.qq.com/s/1Bpt5MG6mbZCYAXDJmIr3A)、视频[《GPT，GPT-2，GPT-3 论文精读》](https://www.bilibili.com/video/BV1AF411b7xQ/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>- [《多图详解attention和mask。从循环神经网络、transformer到GPT2》](https://blog.csdn.net/qq_56591814/article/details/119759105)、[《datawhale课程《transformers入门》笔记3：图解GPT-2》](https://blog.csdn.net/qq_56591814/article/details/119833831)

&#8195;&#8195;GPT是自回归模型（auto-regression），使用 Transformer 的 Decoder 模块构建。原始的Transformer Decoder 比Encoder 多了一个encoder-decoder-attention层（第二个自注意力层，k和v来自encoder层最后的输出memory），使得它可以关注来自 Encoder 的信息。在GPT中，使用的decoder去掉了这一层。
>自回归：生每个 token 之后，将这个 token 添加到输入的序列中，形成一个新序列。然后这个新序列成为模型在下一个时间步的输入。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d3a713253c33ee1ab69f8b8c3ddc3d00.png#pic_center =350x)
&#8195;&#8195;Mask 操作是在 Self-Attention 进行 Softmax 之前进行的，具体做法是将要 Mask 的位置用一个无穷小的数替换 -inf，然后再 Softmax，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/594f7e5ca32db34d8ab828245cd9b470.png#pic_center =600x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0f62e3e02fb4dfea886e5c2aacb6ac00.png#pic_center =600x)
下图是 GPT 整体模型图，其中包含了 12 个 Decoder：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f710439bd7de0805db23d3892ee53ccf.png#pic_center =200x)
GPT提出来“生成式预训练（无监督）+判别式任务精调（有监督）”的范式来处理NLP任务。

- 生成式预训练：在大规模无监督语料上进行预训练一个高容量的语言模型，学习丰富的上下文信息，掌握文本的通用语义。
- 判别式任务精调：在通用语义基础上根据下游任务进行领域适配。具体的在预训练好的模型上增加一个与任务相关的神经网络层，比如一个全连接层，预测最终的标签。并在该任务的监督数据上进行微调训练（微调的一种理解：学习率较小，训练epoch数量较少，对模型整体参数进行轻微调整）

`GPT-2（2019-2）`和`GPT-3（2020-6）`的区别：
- GPT-3使用了更深的网络层数和更宽的Transformer网络结构，模型更大，参数更多，表达能力和语言理解能力更强；
- GPT-3在预训练阶段使用了更大规模的数据集，并采用了更多样化的预训练任务
- GPT-3的微调阶段采用了zero-shot学习和few-shot的方法，使得GPT-3具备更强的泛化能力和迁移学习能力。
### 1.2 InstructGPT（2022-3）
>[《李沐论文精度系列之九：InstructGPT》](https://blog.csdn.net/qq_56591814/article/details/130588064)、bilibili视频[《InstructGPT 论文精读》](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
#### 1.2.1 算法

&#8195;&#8195;语言模型扩大并不能代表它们会更好地按照用户的意图进行工作，大语言模型很可能会生成一些不真实的、有害的或者是没有帮助的答案。换句话说，这些模型和用户的意图并不一致（not aligned with their users）。**由此OpenAI提出了“align”的概念，即希望模型的输出与人类意图“对齐”，符合人类真实偏好。**

InstructGPT提出了语言模型的三个目标：
- helpful——帮助用户解决问题
- honest——不能伪造信息或误导用户
- harmless——不会令人反感，也不会对他人或社会有害

&#8195;&#8195;InstructGPT使用来自人类反馈的强化学习（利用人类的偏好作为奖励信号，让模型仿照人来生成答案），对GPT-3进行微调，实现以上目标。具体实现步骤如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/42d548867de6876c5918ef30d9394476.png#pic_center =700x)
1. 收集**示范数据**，进行有监督微调`SFT`。
	- 标注数据：根据prompts（提示，这里就是写的各种各样的问题），人类会撰写一系列demonstrations（演示）作为模型的期望输出（主要是英文）；
	- 模型微调：**将prompts和人类标注的答案拼在一起，作为人工标注的数据集**，然后使用这部分数据集对预训练的GPT-3进行监督微调，得到第一个模型`SFT`（supervised fine-tuning，有监督微调）
	- **因为问题和答案是拼在一起的，所以在 GPT 眼中都是一样的，都是给定一段话然后预测下一个词，所以在微调上跟之前的在别的地方做微调或者是做预训练没有任何区别。**
2. 收集**比较数据**，训练奖励模型`RM`。
	- 生成式标注是很贵的一件事，所以第二步是进行排序式/判别式标注。用上一步得到的`SFT`模型生成各种问题的答案，标注者（labelers）会对这些输出进行比较和排序（由好到坏，比如下图D>C>A=B）。
	- 基于这个数据集，用强化学习训练一个`RM`（reward model）。训练好了之后这个RM模型就可以对生成的答案进行打分，且打出的分数能够满足人工排序的关系。
3. 使用强化学习的机制，优化`SFT`模型，得到最终的`RL`模型（InstructGPT）。
将`SFT`模型的输出输入`RM`进行打分，通过强化学习来优化`SFT`模型的参数，详见本文4.3节。

&#8195;&#8195;步骤2和步骤3可以连续迭代。第二步可以使得在同样的标注成本下得到更多的数据，模型的性能会更好一些，最终得到的模型就是InstructGPT。
#### 1.2.2 损失函数
&#8195;&#8195;语言模型通过预测下一个词的方式进行训练，其目标函数是最大化给定语言序列的条件概率，而不是“有帮助且安全地遵循用户的指示”，所以当前的语言模型训练的目标函数有问题。这部分在第三步`RL`模型（InstructGPT）中体现。简单来说就是新的损失函数包括以下几个部分：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/84c60e7309c67384f9c8da4486a950a5.png#pic_center =600x)

-  在标注数据集上训练，期望RL模型的输出在RM模型里打分尽可能的高
- $\beta log(\pi_{\phi }^{RL}(y|x)/\pi^{SFT} (y|x) )$：正则项。
随着模型的更新，`RL`产生的输出$y$和原始的`SFT`模型输出的$y$会逐渐不一样，所以作者在loss里加入了一个KL散度（评估两个概率分布的差异），希望`RL`在`SFT`模型的基础上优化一些就行，但是不要偏太远，即相当于加入了一个正则项。
- $\gamma E_{x}\sim D_{pretrain}[log(\pi_{\phi }^{RL}(x)]$：GPT-3模型原来的的目标函数
	- 如果只使用上述两项进行训练，会导致该模型仅仅对人类的排序结果较好，而在通用NLP任务上，性能可能会大幅下降。文章通过在loss中加入了GPT-3预训练模型的目标函数来规避这一问题。
### 1.3 ChatGPT（2022.11.30）
&#8195;&#8195;`ChatGPT`和`InstructGPT`在模型结构，训练方式上都完全一致，即都使用了指示学习（Instruction Learning）和人工反馈的强化学习（RLHF，Reinforcement Learning from Human Feedback）来指导模型的训练。区别是`InstructGPT`是在`GPT3`上微调，`ChatGPT`是在`GPT3.5`上微调的。

### 1.4 ChatGPT plugin
>- [《Introducing OpenAI》](https://openai.com/blog/introducing-openai)、 [《Introducing ChatGPT》](https://openai.com/blog/chatgpt)、 [《ChatGPT plugins》](https://openai.com/blog/chatgpt-plugins)
>- 知乎贴[《chatgpt插件(ChatGPT plugins)功能详解》](https://zhuanlan.zhihu.com/p/618024606)


&#8195;&#8195;为了能够更加灵活的扩展 ChatGPT 的现有功能，OpenAI 正式上线了以安全为核心的 `ChatGPT plugin`，在保障数据安全性的前提下，让 ChatGPT 功能再度提升一整个数量级！plugin（插件）可以允许 ChatGPT 执行以下操作：

- 检索实时信息: 例如，体育比分、股票价格、最新消息等。
- 检索知识库信息: 例如，公司文件、个人笔记等。
- 代表用户执行操作；例如，订机票、订餐等。

&#8195;&#8195;`ChatGPT plugin`，其实就是类似`Toolformer`技术的应用，使得模型可以连接成百上千个API，这样大语言模型只是一个交互的工具，真正完成任务的还是之前的各种工具。这样不仅准确度可以提升，而且3月24`ChatGPT plugin`开通联网后，还可以更新自己的知识库，开启了无限可能。



>&#8195;&#8195;比如用计算器进行计算肯定是可以算对的，而不需要像之前一样进行推理了。
### 1.5 GPT-4（2023.3.14）
>[《李沐论文精度系列之十：GPT-4》](https://blog.csdn.net/qq_56591814/article/details/130542583?spm=1001.2014.3001.5501)

&#8195;&#8195;`GPT-4` 是 OpenAI 继 ChatGPT 之后发布的一个大规模的多模态模型，之前的 GPT 系列模型都是只支持纯文本输入输出的语言模型，而 GPT-4 可以接受图像和文本作为输入，并产生文本输出。
&#8195;&#8195;`GPT-4` 仍然是基于 Transformer 的自回归结构的预训练模型。OpenAI 的博客中表示在随意的对话中，GPT-3.5 和 GPT-4 之间的区别可能很微妙，当任务的复杂性达到足够的阈值时，差异就会出现，即 GPT-4 比 GPT-3.5 更可靠、更有创意，并且能够处理更细微的指令。
&#8195;&#8195;虽然在许多现实场景中的能力不如人类，但 `GPT-4` 在各种专业和学术基准测试中表现出人类水平的表现，包括通过模拟律师考试，得分在应试者的前 10% 左右。和 ChatGPT RLHF 的方法类似，alignment（对齐）训练过程可以提高模型事实性和对期望行为遵循度的表现，具有强大的意图理解能力，并且对 GPT-4 的安全性问题做了很大的优化和提升。
&#8195;&#8195;`GPT-4` 的基础模型其实于 2022 年 8 月就已完成训练。OpenAI 对于基础理解和推理能力越来越强的 LLM 采取了更为谨慎的态度，花 6 个月时间重点针对 Alignment、安全性和事实性等问题进行大量测试和补丁。2023 年 3 月 14 日，OpenAI 发布 `GPT-4` 及相关文章。文章中几乎没有披露任何技术细节。同时当前公开的 `GPT-4` API 是限制了 few-shot 能力的版本，并没有将完整能力的基础模型开放给公众（这个版本会维护到6月14号）。
## 二、 LaMDA系列
### 2.1 LaMDA（Google 2021.5）
>- 论文[《LaMDA: Language Models for Dialog Applications》](https://paperswithcode.com/paper/lamda-language-models-for-dialog-applications)
>- 官网[《LaMDA: our breakthrough conversation technology》](https://blog.google/technology/ai/lamda/)
>- 博客：[《1370亿参数、接近人类水平，谷歌对话AI模型LaMDA放出论文》](https://zhuanlan.zhihu.com/p/461285733)、[《LaMDA：用于对话应用程序的语言模型》](https://blog.csdn.net/bqw18744018044/article/details/130352868)

#### 2.1.1 简介

&#8195;&#8195;`LaMDA` 是谷歌在2021年开发者大会上公布的，**专用于对话**的大语言模型。模型基于Transformer架构，并在具有1.56T单词的公开对话数据和其他网页文档上预训练，最终尺寸从2B到137B。

&#8195;&#8195;论文中提出三个指导模型更好训练的指标，并概括了如何在这三个方面取得进展：
- 质量：
	- 合理性/Sensibleness：生成在对话上下文中有意义的响应
	- 特异性/Specificity：通过判断系统的响应是否特定于前面的对话上下文来衡量的，而不是适用于大多数上下文的通用回应；
	- 趣味性/Interestingness，SSI）：衡量模型是否产生了富有洞察力、出乎意料或机智的回应，因此更有可能创造更好的对话。
- 安全性：
- 根基性，Groundedness ：生成的响应中包含的声明能够被参考和与已知来源进行核实的程度。当前这一代语言模型通常会生成看似合理但实际上与已知外部事实相矛盾的陈述。


#### 2.1.2 LaMDA 预训练与微调
在定义了对话模型训练的指导指标之后，LaMDA 讲过预训练与微调两个阶段的训练。
- 预训练：从公共对话数据和其他公共网页文档中收集并创建了一个具有 1.56T 单词的数据集，是用于训练以往对话模型的单词量的近 40 倍
- 微调阶段做两个工作：
	- LaMDA 生成器：执行混合生成任务，以生成对给定上下文的自然语言响应（模式是预测两个角色来回对话的对话数据集中下一个token）
	- LaMDA 分类器：预测LaMDA 生成器生成的响应的安全与质量（SSI）分数，安全分数低的候选响应首先被过滤掉，剩下的候选响应根据 SSI 分数重新排名，并选择分数最高的作为最终响应。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6a8a92e9dc3d437bfdaa31f1a382d491.png#pic_center)<center> 谷歌使用 LaMDA 分类器进一步过滤掉用于生成任务的训练数据，以增加高质量候选响应的密度</center>
#### 2.1.3 事实根基（真实性、可靠性）
&#8195;&#8195;人们能够使用工具并参考已建立的知识库来检测事实，但是很多语言模型仅利用内部模型参数来获取知识。谷歌通过与人的对话数据集进行微调，让LaMDA模型能够更好地利用外部知识来提供更可靠的回应。
&#8195;&#8195;具体来说，为了提高LaMDA模型原始回应的可靠性，谷歌采集了人与LaMDA之间的对话数据集。**这些对话数据集在适当的情况下使用了搜索查询和搜索结果进行注释**。谷歌通过对这个数据集进行微调，让LaMDA模型的生成器和分类器能够学习在与用户交互时如何调用外部信息检索系统，以增强回应的可靠性。虽然这项工作仍处于早期阶段，但谷歌已经看到了一些有希望的结果。


#### 2.1.4 实验&结论
&#8195;&#8195;谷歌对预训练模型（PT）、微调模型（LaMDA）、人类评估者在多轮双作者对话上的响应进行评估，指标是质量、安全性和根基性，结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1dd797c114e4dc9f328d6a02c28e14fb.png#pic_center =600x)
- 质量：LaMDA 在每个维度和所有模型大小情况下都显著优于预训练模型，合理性、特异性和趣味性等**质量度量通常会随模型参数量提升**；
- 安全性：可以通过微调提升，但是无法仅从模型缩放中得到收益；
- 根基性：随着模型大小的增加，根基性也提升，这或许是因为更大的模型具备更大的记住不常见知识的能力

&#8195;&#8195;微调使模型可以访问外部知识源并有效地将记住知识的负载转移到外部知识源。微调还可以缩小与人类水平的质量差距，尽管该模型在安全性和根基性方面的性能依然低于人类。



### 2.2 Bard（Google 2023.3.21）
>[Bard体验版](https://bard.google.com/chat)
>
&#8195;&#8195;Bard 是谷歌基于 `LaMDA` 研制的对标 ChatGPT 的对话语言模型，目前应该只支持英文对话，限美国和英国用户预约访问。

### 2.3 LLaMA2
- [完整版 LLaMA2 大模型全流程方案，开源了](https://zhuanlan.zhihu.com/p/654187299)：
[Colossal-AI](https://github.com/hpcaitech/ColossalAI/blob/main/docs/README-zhHans.md)是全球规模最大、最活跃的大模型开发工具与社区，提供开箱即用的 8 到 512 卡 LLaMA2 训练、微调、推理方案，对 700 亿参数训练加速 195%，并提供一站式云平台解决方案，极大降低大模型开发和落地应用成本。
## 三、GLM
### 3.1 GLM生态
- `GLM`：一种基于Transformer架构进行改进的通用预训练框架，GLM将不同任务的预训练目标统一为自回归填空任务(Autoregressive Blank Infilling)，使得模型在自然语言理解和文本生成方面性能都有所改善。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1ad2227fdfe6dfd6428b951ae5f6f839.png#pic_center =600x)

- `GLM-130B`：于2022年8月由清华智谱AI开源放出。该大语言模型基于之前提出的GLM(General Language Model)，在Norm处理、激活函数、Mask机制等方面进行了调整，目的是训练出开源开放的高精度千亿中英双语稠密模型，能够让更多研发者用上千亿模型。
- `ChatGLM`: 基于`GLM-130B`，引入面向对话的用户反馈，进行指令微调后得到的对话机器人。`ChatGLM`解决了大基座模型在复杂问题、动态知识、人类对齐场景的不足。`ChatGLM`于2023年3月开启申请内测，目前暂停了公开申请。
- `ChatGLM-6B`：于2023年3月开源。在进行ChatGLM千亿模型内测的同时，清华团队也开放出了同样技术小参数量的版本，方便研发者们进行学习和开发（非商用）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5b9a71bddd6b28c9f00aff63da1d67c6.png#pic_center =600x)
### 3.2 GLM（清华等，2022.3.17）
>- 论文[《GLM: General Language Model Pretraining with Autoregressive Blank Infilling》](https://paperswithcode.com/paper/all-nlp-tasks-are-generation-tasks-a-general)、[github地址](https://github.com/THUDM/GLM)
>- 参考：[《ChatGLM基座：GLM（General Language Model）》](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490547&idx=2&sn=37f1fb81039f5c0c9c644e2bae81c9ca&chksm=9bc5f3a1acb27ab75f5157d51e6668c96e6b6eaf5c6ee497764cefdbc9654caf9155bbb9e3ba&scene=178&cur_album_id=2878066965444362241#rd)、知乎[《GLM: General Language Model Pretraining with Autoregressive Blank Infilling》](https://zhuanlan.zhihu.com/p/579645487)

#### 3.2.1 背景

&#8195;&#8195;NLP任务分为NLU(文本分类、分词、句法分析、信息抽取等)、有条件生成任务（seq-seq，如翻译任务、QA）、无条件生成任务（用预训练模型直接生成内容）三大类。基础的预训练模型也分为三种：
| 预训练模式 | 代表模型 | 说明                                                                                           |
|------------|----------|------------------------------------------------------------------------------------------------|
| 自编码     | BERT     | 双向的transformer作为编码器，在语言理解相关的文本表示效果很好。缺点是不能直接用于文本生成。         |
| 自回归     | GPT      | 从左往右学习的模型，在长文本的生成能力很强。缺点是单向的注意力机制在NLU任务中，不能完全捕捉token的内在联系。|
| 编码解码   | T5       | 编码器使用双向注意力，解码器使用单向注意力，并且有交叉注意力连接两者。在有条件生成任务中表现良好(文本摘要，回答生成)。  |

所以用一张表格简单总结就是：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3c71e4c6873fbcb080b6a69adee3a448.png#pic_center =500x)
>注：✅表示擅长，x表示无法直接应用，— 表示可以做

&#8195;&#8195;目前这些训练前框架都不足以在所有NLP中具有竞争力任务。以往的工作（T5）试图通过多任务学习统一不同的框架。然而，由于自编码和自回归的目标性质不同，简单的统一不能完全继承这两个框架的优点。

#### 3.2.2 主要贡献
- 提出了一种基于自回归空白填充的通用语言模型（GLM）来应对上述三种任务。
- GLM通过添加2D位置编码并允许任意顺序预测跨度来改进空白填充预训练，从而在NLU任务上比BERT和T5获得了性能提升。
- 通过变化空白数量和长度，可以针对不同类型的任务对GLM进行预训练。
- 在跨NLU、有条件和无条件生成的广泛任务范围内，GLM在相同的模型大小和数据情况下优于BERT、T5和GPT，并且使用BERTLarge的1.25×参数的单个预训练模型实现了最佳性能，展示了其对不同下游任务的通用性。

#### 3.2.3 预训练
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/50010511eb936e87a1af7ef7da3f4fbf.png#pic_center =700x)<center>GLM 将 NLU 任务制定为包含任务描述的完形填空问题，并通过自回归生成来回答 </center>

##### 3.2.3.1 模型输入
&#8195;&#8195;GLM通过优化自回归空白填充目标进行训练。给定输入文本`x =[x 1 ,··· ,x n ]`，对多个文本跨度`spans {s 1 ,··· ,s m }`  进行采样，然后将这些span进行mask（用[mask]标记替换），形成损坏的文本`xcorrupt`。span的长度服从泊松分布(λ=3)，与BART一样，重复采样，直到15%的token被mask（根据经验，15% 的比率对于下游 NLU 任务的良好性能至关重要）。

下面举例说明。对于input=[x1,x2,x3,x4,x5,x6]，假设mask 掉 [x3] 和 [x5,x6]。然后输入x包括两部分：
-  part A：损坏的文本`xcorrupt`，例子中是`[x1,x2,mask,x4,mask]`
- part B ：mask掉的span部分，例子中是 `[x5,x6],[x3]`。为了完全捕捉不同跨度之间的相互依赖关系，会随机排列跨度的顺序，类似于置换语言模型`XLNet`。

##### 3.2.3.2 预训练目标&Mask矩阵
&#8195;&#8195;预训练的目标是：**通过自回归方式从损坏的文本`xcorrupt`中预测跨度`span`中被mask的部分**，即从part A预测part B。下图显示了mask矩阵，可以看出：
- Part A部分采用双向注意力，可以关注它们自己（蓝框）前后的信息，但不能关注 B；
- Part B采用单向注意力，可以关注 A 部分及 B 部分中的前文。

&#8195;&#8195;为了启用自回归生成，每个span都自动填充了特殊标记 [S] 和 [E] ，表示预测从start到end跨度的部分。通过这种方式，**GLM在统一模型中自动学习双向编码器（Part A）和单向解码器（Part B）**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b4d36b5fa650c381cba3e12cae1e7a55.png#pic_center =350x)<center> [M] := [MASK], [S] := [START], [E] := [END]</center>
##### 3.2.3.3 二维位置编码
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ace897b3068b96955414a75555d16b09.png#pic_center =350x)

&#8195;&#8195;如上图所示，Part A与PartB拼接成一个sequence，每个token都用两个位置编码 ids（ two positional ids）：
- positional id1：表示损坏的文本xcorrupt中的位置，PartB以span被mask的位置表示
- positional id2：表示跨度内的位置，所以Part A统一以0表示。PartB中的token，以从开始到此位置的span长度表示。

最终两个位置编码都会加入到输入token 的embedding向量中。

##### 3.2.3.4 多任务预训练
&#8195;&#8195;前面的介绍中，span都比较短，适用于NLU任务。然而，我们希望模型能同时处理NLU任务和文本生成任务是，所以我们设置了第二个预训练任务——长文本生成，分两个级别：

- 文档级别（gMASK）。我们随机抽样一个跨度，其长度从原始长度的50％到100％的均匀分布中抽样。该目标旨在进行长文本生成。
-  句子级别（sMASK）。我们限制掩蔽跨度必须是完整的句子。我们随机抽样多个跨度（句子）以覆盖15％的原始令牌。此目标旨在进行序列到序列任务，其预测通常为完整的句子或段落。

&#8195;&#8195;这两个级别的生成任务和NLU任务相同，唯一的区别在于跨度数量和跨度长度。在实际使用中，可以根据不同的任务需要，设置不同mask方式的比例。例如，如果希望模型有更强的生成能力，可以把文档级别的gMASK的比例设置地比较高。在`GLM-130B`中，采用了70%文档级别的gMASK和30%单词级别的MASK。

#### 3.2.4 模型结构
>- [Pre-LN：On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745)
>- [Sandwich-LN: Cogview: Mastering text-to-image generation via transformers](https://arxiv.org/abs/2105.13290)

&#8195;&#8195;GLM 使用单个Transformer ，并对架构进行了多项修改：
1. 采用`Sandwich-LN`。LayerNorm会影响训练的稳定性，目前认为认为稳定性上: Sandwich-LN > Pre-LN > Post-LN（原始的BERT）
2. 使用单个线性层来进行输出Token预测
3. `ReLU`激活函数替换为`GELU`


![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7fda047d3b6135335fd77526592eef9e.png#pic_center =600x)


#### 3.2.5 下游任务微调
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/45709ffc029125e3255f7f60d3e73d01.png#pic_center =350x)
&#8195;&#8195;对于下游NLU任务，我们通常会在模型之上添加线性分类器，以前层的输出作为输入来预测正确的标签，但这会导致预训练和微调之间的不一致。
&#8195;&#8195;GLM微调时，分类任务转换为完形填空，类似PET。如上图示例，原本的“positive”和“negative”二分类任务，转换为预测[mask]的任务（映射到单词“good”和“bad”）。
>&#8195;&#8195;其实这部分就是`Prompt Tuning`，有三种主要算法：PET、P-Tuning和EFL。有兴趣的可以参考[《PaddleNLP系列课程一：Taskflow、小样本学习、FasterTransformer》](https://blog.csdn.net/qq_56591814/article/details/128215142?spm=1001.2014.3001.5501)第二章。

#### 3.2.6 实验结果
1. SuperGLUE：NLU任务上，GLM在大多数具有基础架构或大型架构的任务上始终优于BERT。平均而言，`GLMBase` 得分比BERT Base 高 4.6%，`GLMLarge` 得分比BERT Large 高 5.0%。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eb4d824ce69c0f92ea94e36d8a5831cf.png#pic_center =600x)
2. Sequence-to-Sequence：GLM RoBERTa可以实现匹配Seq2Seq BART模型的性能，并且优于T5和UniLMv2

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7e7cf887813d6c99d31310e1d34aab8d.png#pic_center =600x)
3. 有条件生成和无条件生成
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/73e1c7dd4090030a59812dd50ba1da56.png#pic_center =600x)
其它结果请看论文。
#### 3.2.7 结论
&#8195;&#8195;**GLM是一种用于自然语言理解和生成的通用预训练框架。论文展示了NLU任务可以被形式化为条件生成任务，因此可以由自回归模型解决**。GLM将不同任务的预训练目标统一为自回归空白填充，具有混合的注意力掩码和新颖的二维位置编码。我们的实验证明GLM在NLU任务中优于先前的方法，并且可以有效地共享参数以用于不同的任务。

### 3.3 GLM-130B
>- 论文[《GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL》](https://openreview.net/pdf?id=-Aw0rrrPUF)、[github项目](https://github.com/THUDM/GLM-130B/)、[官方博客](https://chatglm.cn/blog)
>- 官方视频[《从GLM-130B到ChatGLM：大模型预训练与微调》](https://www.bilibili.com/video/BV1iu4y1Z7bv/?vd_source=8d00c2c0cdbe325ba3b959e4aea901ea)、[视频笔记](https://zhuanlan.zhihu.com/p/636329188)、[《论文阅读-GLM-130B：一种开放的双语预训练模型》](https://zhuanlan.zhihu.com/p/618690572)

#### 3.3.1 背景&模型优势
&#8195;&#8195;`GPT-3`是一款强大的语言模型，但由于未公开，存在技术瓶颈。目前的语言模型规模庞大，训练需要数百张A100以上的显卡，非常困难。`GLM-130B`是2022年8月由清华AI向研究界和工业界开放的拥有1300亿参数的中英双语稠密模型。本文介绍了GLM-130B的训练过程，包括设计选择、高效稳定的训练策略和工程努力。
&#8195;&#8195;在广泛的英语测试中，`GLM-130B`的性能明显优于`GPT-175B`，但在OPT-175B和BLOOM-176B上并未观察到性能优势。在相关测试中，GLM-130B也始终明显优于最大的中文模型`ERNIE TITAN 3.0 260B`。最后，利用GLM-130B独特的缩放特性，实现了INT4量化使其成为100B缩放模型中的先驱，可进行快速推理（小型多任务模型成为一种趋势）。
>- 可复现性： 所有结果（超过 30 个任务）均可通过我们的开源代码和模型参数复现。
>- 跨平台： 支持在国产的海光 DCU、华为昇腾 910 和申威处理器及美国的英伟达芯片上进行训练与推理。

&#8195;&#8195;2022年11月，在斯坦福大学大模型中心对全球30个主流大模型的评测报告中，**GLM-130B 在准确性和恶意性指标上与 GPT-3 175B (davinci) 接近或持平**，鲁棒性和校准误差在所有千亿规模的基座大模型（作为公平对比，只对比无指令提示微调模型）中表现不错（下图）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/53c313a5b2b02db79b98aff4dd357c62.png#pic_center =600x)


| ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/afa5bc539f5edb7a6f8e43c2ae6486a6.png) | ![(image2.jpg)\]](https://i-blog.csdnimg.cn/blog_migrate/ebe2f4aebc398100c6f29280babd5e77.png) |
|:---:|:---:|

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d515826353c9507cf2670700c4ca9f8.png#pic_center =600x)
#### 3.3.2 Deep Layer Norm
>[DeepNorm：Deepnet: Scaling transformers to 1,000 layers](https://arxiv.org/abs/2203.00555)
>
&#8195;&#8195;训练不稳定性是训练LLMs的一个主要挑战，适当选择LNs有助于稳定LLM的训练。作者发现GLM的训非常不稳定，于是使用了`Deep Layer Norm`机制，公式为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a6d27b2abd31f41ac437bd4b50127727.png#pic_center =400x)
&#8195;&#8195;此外，所有偏置项都被初始化为零。下图显示`Deep Layer Norm`显著有利于GLM-130B的训练稳定性，比`Sandwich-LN`更稳定。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eba917ac4e47bbca4c128a908cbf2da9.png#pic_center =800x)<center> GLM-130B训练不同层次规范的试验。事实证明，DeepNorm是最稳定的一种，因为它具有较小的梯度范数，并且在早期训练中不会出现尖峰</center>
#### 3.3.3 位置编码
&#8195;&#8195;位置编码分为绝对位置编码和相对位置编码。一些较新的在大模型中应用较多的位置编码有[ALiBi](https://arxiv.org/abs/2108.12409)和[RoPE](https://arxiv.org/abs/2104.09864)，`GLM-130B`采用的是后者。`GLM-130B`团队的观点是虽然`RoPE`外推性能有限，但是并不应该把长文本的处理问题完全依赖于位置编码的外推，而是需要什么样的长度就在什么样的context length上做训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a5e53e35b2db1f7be98d4a20de19b3e5.png#pic_center =600x)
#### 3.3.4 大模型训练系列技术（混合精度训练、激活函数重演、数据并行、流水线气泡）
这部分内容请参考官方视频[《从GLM-130B到ChatGLM：大模型预训练与微调》](https://www.bilibili.com/video/BV1iu4y1Z7bv/?vd_source=8d00c2c0cdbe325ba3b959e4aea901ea)、[视频笔记](https://zhuanlan.zhihu.com/p/636329188)。
### 3.4 ChatGLM（2023.3.22）
>[官网（内测申请）](https://chatglm.cn/)、[《ChatGLM 的 Prompt 工程实践》](https://www.bilibili.com/video/BV1ic411c7gE)及[视频ppt文件](https://pan.baidu.com/s/1T5vBCAPG2ahrI_H2jKnihw?pwd=mwmr)，提取码: mwmr 



&#8195;&#8195;由于`GLM-130B`的动态知识欠缺、知识陈旧、缺乏可解释性，同时缺少高效“Prompt工程”，在对话场景中使用时很难尽人意。所以清华大学参考了 `ChatGPT` 的设计思路，在 `GLM-130B` 中注入了代码预训练，通过有监督微调（Supervised Fine-Tuning）、反馈自助（Feedback Bootstrap）、人类反馈强化学习（Reinforcement Learning from Human Feedback） 等技术实现人类意图对齐。

&#8195;&#8195;ChatGLM千亿参数版本由于还处于内测，没有太多的公开信息，报告中给出了目前的一些成绩对比：

- 在`MMLU`评测基准上，较`GLM-130B`有了有更大提升，超过`GPT3 davinci`版本30%，达到了`ChatGPT`(GPT-3.5-turbo)的81%
- 在非数学知识场景达到了`ChatGPT`(GPT-3.5-turbo)的95%
- 在非数学推理场景达到了`ChatGPT`(GPT-3.5-turbo)的96%
- 在高考、SAT、LSAT等考试的综合成绩上，达到了`ChatGPT`(GPT-3.5-turbo)的90%。
### 3.5  ChatGLM-6B
>- [项目地址](https://github.com/THUDM/ChatGLM-6B)、[官方博客](https://chatglm.cn/blog)
>- 微调心得：
>	- [ChatGLM-6B 在 ModelWhale 平台的部署与微调教程（直接运行）](https://www.heywhale.com/mw/project/6436d82948f7da1fee2be59e)
> 	 - [ChatGLM-6B保姆教程](https://www.zhihu.com/question/595670355/answer/3015099216)、[微调实战](https://zhuanlan.zhihu.com/p/625468667)、	[ChatGLM-6B指令微调](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490455&idx=1&sn=7625fc9379c1b973178a347efadccc8c&chksm=9bc5f3c5acb27ad3703d2468043a0cea2d4c701c680521a7e863dd1a7157be67b4382a4d77bc&scene=178&cur_album_id=2878066965444362241#rd)

#### 3.5.1 简介
&#8195;&#8195;由于`ChatGLM`千亿参数版本暂未公开，为了与社区一起更好地推动大模型技术的发展，清华团队开源了62亿参数版本的`ChatGLM-6B`。结合模型量化技术，用户可以在消费级的显卡上进行本地部署。

该版本具有以下特点：

- 充分的中英双语预训练： ChatGLM-6B 在 1:1 比例的中英语料上训练了 1T 的 token 量，兼具双语能力。
- 优化的模型架构和大小： 吸取 GLM-130B 训练经验，修正了二维 RoPE 位置编码实现，使用传统FFN结构。6B（62亿）的参数大小，也使得研究者和个人开发者自己微调和部署 ChatGLM-6B 成为可能。
- 较低的部署门槛： FP16 半精度下，ChatGLM-6B 需要至少 13GB 的显存进行推理，结合模型量化技术，这一需求可以进一步降低到 10GB（INT8） 和 6GB（INT4）， 使得 ChatGLM-6B 可以部署在消费级显卡上。
- 更长的序列长度： 相比 GLM-10B（序列长度1024），ChatGLM-6B 序列长度达 2048，支持更长对话和应用。
- 人类意图对齐训练： 使用了监督微调（Supervised Fine-Tuning）、反馈自助（Feedback Bootstrap）、人类反馈强化学习（Reinforcement Learning from Human Feedback） 等方式，使模型初具理解人类指令意图的能力。输出格式为 markdown，方便展示。

#### 3.5.2 局限性

- 模型容量较小： 6B 的小容量，决定了其相对较弱的模型记忆和语言能力。在面对许多事实性知识任务时，ChatGLM-6B 可能会生成不正确的信息；她也不擅长逻辑类问题（如数学、编程）的解答。
- 偏见：ChatGLM-6B 只是一个初步与人类意图对齐的语言模型，可能会生成有害、有偏见的内容。
- 多轮对话能力较弱：ChatGLM-6B 的上下文理解能力还不够充分，在面对长答案生成，以及多轮对话的场景时，可能会出现上下文丢失和理解错误的情况。
- 英文能力不足：训练时使用的指示大部分都是中文的，只有一小部分指示是英文的。因此在使用英文指示时，回复的质量可能不如中文指示的回复，甚至与中文指示下的回复矛盾。
- 易被误导：ChatGLM-6B 的“自我认知”可能存在问题，很容易被误导并产生错误的言论。例如当前版本模型在被误导的情况下，会在自我认知上发生偏差。即使该模型经过了1万亿标识符（token）左右的双语预训练，并且进行了指令微调和人类反馈强化学习（RLHF），但是因为模型容量较小，所以在某些指示下可能会产生有误导性的内容。
#### 3.5.3 环境配置
>[如何评价智谱 AI 发布的 ChatGLM，以及开源支持单卡推理的 ChatGLM-6B 模型？](https://www.zhihu.com/question/589484629/answer/2979328794)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/31dd101df11fba29aabb1aa8ed579644.png#pic_center =600x)
&#8195;&#8195;`ChatGLM-6B`所有模型文件，总共13G左右，显存不够时可以使用量化模型的方式加载，4-bit量化后可以加载到显存，占用5.2G显存左右，但是量化加载需要13G的内存，就是无论无何这13G的模型文件要么直接加载到显存，要么加载到内存量化后再加载到显存

下面官方直接提供了量化后的模型文件，也就避免了上述处理13G模型文件的操作。
- 4-bit量化后的模型文件下载：`GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b-int4`
- 进一步提对Embedding量化后的模型，模型参数仅占用4.3 GB显存：`GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b-int4-qe`

#### 3.5.4 相关开源项目

- [Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain):中文langchain项目，基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成，增加web search功能、知识库选择功能和支持知识增量更新
- [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM):基于 langchain 的 ChatGLM 应用，实现基于可扩展知识库的问答
- [ChatGLM-6B-Engineering](https://github.com/LemonQu-GIT/ChatGLM-6B-Engineering)：基于 ChatGLM-6B 后期调教，网络爬虫及 Stable Diffusion 实现的网络搜索及图片生成
- [ChatGLM-OpenAI-API](https://github.com/ninehills/chatglm-openai-api):将 ChatGLM-6B 封装为 OpenAI API 风格，并通过 ngrok/cloudflare 对外提供服务，从而将 ChatGLM 快速集成到 OpenAI 的各种生态中。

对 ChatGLM-6B 进行微调的开源项目：
- [InstructGLM](https://github.com/yanqiangmiffy/InstructGLM):基于ChatGLM-6B进行指令学习，汇总开源中英文指令数据，基于Lora进行指令数据微调，开放了Alpaca、Belle微调后的Lora权重，修复web_demo重复问题
- [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning):一种平价的chatgpt实现方案，基于清华的ChatGLM-6B+ LoRA 进行finetune。

- [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)：基于ChatGLM-6B模型进行定制化微调，汇总10余种指令数据集和3种微调方案，实现了4/8比特量化和模型权重融合，提供微调模型快速部署方法。
- [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)：基于 LoRA 对 ChatGLM-6B 进行微调。
### 3.6 ChatGLM2-6B部署与微调
详见我的另一篇博客[《ChatGLM2-6B部署与微调》](https://blog.csdn.net/qq_56591814/article/details/133049185?spm=1001.2014.3001.5502)
### 3.7 ChatGLM3-6B（2023.10.27）
&#8195;&#8195;2023年的10月27日，智谱AI联合清华大学再次发布第三代基础大语言模型ChatGLM3系列。本次发布的第三代模型共包含3个：基础大语言模型ChatGLM3-6B-Base、对话调优大语言模型ChatGLM3-6B和长文本对话大语言模型ChatGLM3-6B-32K。

&#8195;&#8195;ChatGLM3的性能比第二大有大幅的提高。在各项评测中的得分均有大幅提升。官方甚至宣称：ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能。

| 模型版本 | 评测任务 | 评测方向 | 得分 | 相比第二代提升 |
| --- | --- | --- | --- | --- |
| ChatGLM2-6B-Base | MMLU | 自然语言理解等 | 47.9 | - |
| ChatGLM2-6B-Base | GSM8K | 数学能力 | 32.4 | - |
| ChatGLM2-6B-Base | C-Eval | 中文能力 | 51.7 | - |
| ChatGLM3-6B-Base | MMLU | 自然语言理解等 | 61.4 | 36% |
| ChatGLM3-6B-Base | GSM8K | 数学能力 | 72.3 | 179% |
| ChatGLM3-6B-Base | C-Eval | 中文能力 | 69 | 33.5% |

更多介绍详见[《ChatGLM3：6B版本的ChatGLM3能力大幅增强，依然免费商用授权！》](https://www.datalearner.com/blog/1051698397994641)
## 四、 PaLM（Google Research 2022.4 ）
>- [《LLM系列之PaLM》](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490532&idx=1&sn=d3c77bd67ed0043ee72409e45874eed6&chksm=9bc5f3b6acb27aa04ba23b0975b934fc1facf2a6d822f332a1a373de38aa9c48a025c8587e91&scene=178&cur_album_id=2878066965444362241#rd)
>- 论文[《PaLM: Scaling Language Modeling with Pathways》](https://paperswithcode.com/paper/palm-scaling-language-modeling-with-pathways-1)
>- [github1（PaLM-pytorch）](https://github.com/lucidrains/PaLM-pytorch/tree/main)、[github2](https://github.com/conceptofmind/PaLM)、[Hugging Face](https://huggingface.co/conceptofmind/palm-1b)

### 4.1 简介

`PaLM`（Pathways Language Model ）是谷歌2022年提出的 540B 参数规模的大语言模型，论文主要贡献有：

- PaLM 使用 谷歌提出的Pathways系统 在 6144 TPU v4 芯片上进行训练（Pathways 是一种新的 ML 系统，可以跨多个 TPU Pod 进行高效训练，详情可参考李沐的[Pathways论文精读](https://www.bilibili.com/video/BV1xB4y1m7Xi/?spm_id_from=333.999.0.0)）
- 它通过在数百种语言理解和生成基准上实现小样本学习sota结果，证明了scaling的良好效果。

### 4.2 模型结构
PaLM 使用Transformer  decoder架构，但是做了一些修改：
- 采用`SwiGLU`激活函数，提供更好的性能和梯度流动，提高模型效果
- 提出`Parallel Layers`，并行处理多个输入，训练速度提高约 15%
- Multi-Query Attention共享key/query的映射，自回归时解码更快
- 位置嵌入使用`RoPE embeddings`，在长文本上性能更好
- 采用`Shared Input-Output Embeddings`，输入、输出embedding矩阵是共享
- 不使用偏置项：在dense kernel或layer norm中都没有使用偏差，这种操作提高了大模型的训练稳定性

#### 4.2.1 SwiGLU层
>- 论文[《GLU Variants Improve Transformer》](https://paperswithcode.com/paper/glu-variants-improve-transformer)
>- 知乎贴[《GLU代替Transformer中的FFN(Feed-Forward Networks)》](https://zhuanlan.zhihu.com/p/486798318)

&#8195;&#8195;在论文《GLU Variants Improve Transformer》中提到，使用`SwiGLU`替换transformer中FFN的第一层，得到的FFNSwiGLU，已被证明可以显著提高模型效果，下面进行简单的介绍。
##### 4.2.1.1 FFN
&#8195;&#8195;一个Transformer Bolck中主要包含三部分：MultiheadAttention(多头注意力)、FFN(前馈神经网络)和Add&Norm（残差连接和LayerNorm），其中FFN是由两个线性变换层和激活函数组成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd0a5069085d92c360dedc6c0d653916.png#pic_center =200x)<center>Transformer Block </center>
所以Transformer中的FFN层可表示为：
$$FFN(x, W_1, W_2, b_1, b_2) = W_{2}(relu(xW_{1}+b_{1}))+b_{2}=max(0, xW_1 + b_1)W_2 + b_2$$
- relu的优点：神经元只需要进行加、乘和比较这些简单的计算操作，而且有很好的稀疏性，大约有50%的神经元会处于激活状态
- relu的缺点：输出是非零中心化的，会给后面的计算引入偏置转移的问题，影响梯度下降的效率。
 神经元在训练时容易死亡，不恰当的更新会导致参数梯度一直为0，永远无法被激活。

在T5模型中去除了FFN的偏置项，所以T5中的FFN表示为：
$$FFN_{ReLU}(x, W_1, W_2) =max(0, xW_1 )W_2 $$
##### 4.2.1.2 swish激活函数及FFN变体
&#8195;&#8195;后面的一些工作也使用了其它的激活函数替换`ReLU`，例如Hendrycks等人就使用了`GELU`来进行替换，Ramachandran等人使用了`Swish`来进行替换 ：
 
$$\begin{align*}
\text{FFN}_{\text{GELU}}(x, W_1, W_2) &= \text{GELU}(xW_1)W_2 \\
\text{FFN}_{\text{Swish}}(x, W_1, W_2) &= \text{Swish}_1(xW_1)W_2 \\
\end{align*}$$

其中，GELU（高斯误差线性单元）公式为：(erf 表示误差函数)
$$\text{GELU}(x) = \frac{1}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) \cdot x$$
Swish：
$$\text{Swish}(x) = x \cdot \text{sigmoid}(\beta \cdot x)$$

Swish具有以下特性：
- 平滑性：Swish函数在整个实数域上是连续且可微的，没有突变点或不连续的部分，这有助于提高梯度的稳定性和训练的效果。（比ReLU更平滑）

- 渐进饱和性：Swish函数在输入为正或负的大值时，会趋向于饱和，即输出值接近于输入值。这有助于抑制大幅度的激活响应，减轻梯度爆炸的问题。

- 自适应性：Swish函数具有自适应的特性，它的形状和曲线根据输入值的变化而变化。在较大的负值范围内，Swish函数趋向于线性变换；在较大的正值范围内，Swish函数趋向于饱和（Sigmoid函数的特性），保持输入的大部分信息。Swish函数结合了ReLU的线性增长特性，和Sigmoid函数的平滑特性，使得处理复杂的非线性关系时更具表达能力。

- 较低的计算复杂度：相比于其他激活函数（如ReLU），Swish函数的计算复杂度较低，可以更高效地进行前向传播和反向传播。
##### 4.2.1.3 GLU一般形式及变体
&#8195;&#8195;`GLU`，门控线性单元（Gated Linear Units），是一种神经网络层，其核心思想是通过门控机制来控制激活函数的输出，由线性变换和门控机制组成：
- 输入$x$通过线性变换得到两个输出向量，分别称为"门"向量（下式中的$xW + b$）和"中间"向量（下式中的$xV + c$）
- 门向量通过一个激活函数（通常是sigmoid函数）进行门控，产生一个介于0和1之间的值，表示在给定位置上的输入是否应该被过滤或保留
- 中间向量与门向量进行Hadamard乘积，从而对输入进行控制和加权。

<font color='deeppink'>**GLU的一般形式**</font>可表示为：
$$GLU(x, W, V, b, c) = σ(xW + b) ⊗ (xV + c)$$

如果将激活函数省略，就可以得到一个双线性变换函数(Bilinear)，可表示为：
$$Bilinear(x, W, V, b, c) = (xW + b) ⊗ (xV + c)$$
<font color='deeppink'>**GLU变体**</font>：

$$\begin{align*}
\text{ReGLU}(x, W, V, b, c) &= \max(0, xW + b) \odot (xV + c) \\
\text{GEGLU}(x, W, V, b, c) &= \text{GELU}(xW + b) \odot (xV + c) \\
\text{SwiGLU}(x, W, V, b, c, \beta) &= \text{Swish}_{\beta}(xW + b) \odot (xV + c)
\end{align*}$$

&#8195;&#8195;从GLU的变体中我们不难发现ReGLU的表达式与FFN的表达式是相似的，所以考虑将FFN的第一次线性和激活函数替换为GLU，所以有：
$$\begin{align*}
\text{FFNGLU}(x, W, V, W2) &= (\sigma(xW) \odot xV)W_2 \\
\text{FFNBilinear}(x, W, V, W2) &= (xW \odot xV)W_2 \\
\text{FFNReGLU}(x, W, V, W2) &= (\max(0, xW) \odot xV)W_2 \\
\text{FFNGEGLU}(x, W, V, W2) &= (\text{GELU}(xW) \odot xV)W_2 \\
\text{FFNSwiGLU}(x, W, V, W2) &= (\text{Swish}1(xW) \odot xV)W_2 \\
\end{align*}$$

&#8195;&#8195;不难看出，替换操作是FFN原先第一层的$xW_1$替换为GLU的$(xW) \odot xV$，所以对比于原始的FFN来说，多了一项线性变换$xV$。作者为了保持参数数量和计算量不变，将hidden unit减少2/3，即$W,V$的第二维和$W_2$的第一维减少2/3。
##### 4.2.1.4 实验结果
&#8195;&#8195;使用[T5模型](https://arxiv.org/abs/1910.10683)作为baseline，设置编码器和解码器各由12层组成，维度768，注意力层采用 
$head=12,d_k = d_v = 64$ ，FFN层及其变体采用3072个神经元，而GLU及其变体则采用2048个神经元，实验结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/99a135968c19caf61e0668975857777e.png#pic_center =400x)
在GLUE语言理解基准数据集的任务上进行对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f721594b2b8ddaaa47b122783d19d955.png#pic_center =700x)
#### 4.2.2 Parallel Layers
&#8195;&#8195;在传统的语言模型中，模型的每一层负责特定的任务，每一层都必须等待前一层完成后才能开始处理。`Parallel Layers`是PaLM的一个关键创新，通过并行层，可以同时处理多个任务，显著提高模型的速度和准确性（并行层可以同时从多个示例中进行学习）。
&#8195;&#8195;标准的transformer block中，输出公式可以写成：
$$y = x + MLP(LayerNorm(x + Attention(LayerNorm(x)))$$
在Parallel Layers中，可以写成：（代码可参考[LLM系列之PaLM](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490532&idx=1&sn=d3c77bd67ed0043ee72409e45874eed6&chksm=9bc5f3b6acb27aa04ba23b0975b934fc1facf2a6d822f332a1a373de38aa9c48a025c8587e91&scene=178&cur_album_id=2878066965444362241#rd)）
$$y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))$$

&#8195;&#8195;并行公式使大规模训练速度提高了大约 `15%`。消融实验显示在 8B 参数量下模型效果下降很小，但在 62B 参数量下没有模型效果下降的现象。

#### 4.2.3 共享键/值的映射
&#8195;&#8195;标准的transformer block中，假设有k个注意力头，则计算过程为：
- 使用矩阵$W$将Q、K、V映射到k个不同的语义空间中，形状为[k, h]；
- 进行Attention计算，得到k个head矩阵。
- 将这k个矩阵串联拼接起来，乘以矩阵$W^o$（保持维度一致）得到多头注意力结果。

故多头注意力模型公式可以写成：

$$Multi-Head（Q ,K , V )=concat(head_1....head_c)W^{O}=\begin{bmatrix}
concat(h_{1,1}...h_{n,1})W^{O}\\ 
...\\ 
concat(h_{1,m}...h_{n,m}W^{O}\end{bmatrix}$$

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/976eb1da41df902477fb7cee1c1a9c27.png#pic_center =500x)
&#8195;&#8195;而在PaLM中，每个head的key/value权值共享，即key和value被映射为`[1,h]`，但query仍然被映射为`shape[k,h]`。论文发现这种操作对模型质量和训练速度没有影响，但在自回归解码时间上有效节省了成本。（标准的多头注意力在自回归解码过程中，键（key）和值（value）张量在不同样本之间不共享，并且每次只解码一个单词，所以在加速器硬件上的效率较低）

#### 4.2.4 RoPE embeddings
&#8195;&#8195;RoPE 嵌入在长文本上具有更好的性能 ，具体原理可看苏神文章《Transformer升级之路：2、博采众长的旋转式位置编码》
#### 4.2.5 Shared Input-Output Embeddings
&#8195;&#8195;在自然语言处理任务中，输入序列和输出序列都需要经过嵌入层来获取对应的嵌入向量。而在PaLM中，输入序列和输出序列共享相同的嵌入层参数矩阵，即输入序列中的单词通过嵌入层获得其嵌入向量，同时输出序列中的单词也通过相同的嵌入层获得对应的嵌入向量。

&#8195;&#8195;这样做的目的是为了让输入和输出之间共享语义信息，表示更加一致和相互关联，使得模型能够更好地理解输入和输出之间的语义关系，并更准确地进行预测和生成。

&#8195;&#8195;需要注意的是，共享嵌入层并不意味着输入和输出之间的嵌入是完全相同的，而是共享参数矩阵，通过参数的共享来实现输入和输出之间的信息传递和一致性。

### 4.3 模型尺度和训练数据
考虑了三种不同的模型尺度：540B、62B 和 8B 参数：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/981efca8af2aa637f226c87d18077078.png)

PaLM 预训练数据集：
- 包含 7800 亿个标记的高质量语料库，代表了广泛的自然语言用例。该数据集是经过过滤的网页、书籍、维基百科、新闻文章、源代码和社交媒体对话的混合体。该数据集基于用于训练  `LaMDA`（Thoppilan 等人，2022 年）和 `GLaM`（Du 等人，2021 年）的数据集。
- 所有三个模型都只在一个时期的数据上进行训练（所有模型的数据清洗方式都相同）。
- 除了自然语言数据，预训练数据集还包含 196GB 代码，从 GitHub 上的开源存储库获取，包括 Java、HTML、Javascript、Python、PHP、C#、XML、C++ 和 C。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d9df0b5ad9fc5bddfdf91f1e51837b3d.png#pic_center)
### 4.4 训练硬件资源
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d689f7ee62c31d6489890414dfecbea.png#pic_center =800x)
&#8195;&#8195;总体来说，该程序包含：
- 组件 A：用于 pod 内前向+反向计算（包括 pod 内梯度减少）
- 组件 B：用于跨 pod 梯度传输的传输子图，以及用于优化器更新的（包括本地和远程梯度的求和） 

&#8195;&#8195;Pathways 程序在每个 pod 上执行组件 A，然后将输出梯度传输到另一个 pod，最后在每个 pod 上执行组件 B。因此，它掩盖了延迟，还分摊了管理数据传输的成本，PaLM 代表了 LLM 训练效率向前迈出的重要一步。（PaLM的硬件FLOPs利用率为57.8%，模型FLOPs利用率见下表）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ff1a2bbe51cf9c3905b8d8f5f86cfa4d.png#pic_center =600x)
>&#8195;&#8195;`FLOPS`表示每秒钟可以执行的浮点运算次数，`Model FLOPS utilization`（模型FLOPS利用率）是指在机器学习模型中使用的浮点运算数（FLOPS）的有效利用程度，表示实际执行的浮点运算与模型的理论计算能力之间的关系。一个高的Model FLOPS utilization意味着模型能够有效地利用计算资源，并将其转化为有意义的计算任务。这意味着模型的计算效率较高，能够更快地完成训练或推断任务。

### 4.5 实验
&#8195;&#8195;PaLM 540B 在所有基准测试中都优于类似尺寸的模型（Megatron-Turing NLG 530B）。这表明预训练数据集、训练策略和训练期间观察到的标记数量在实现这些结果方面也起着重要作用。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2328316374923aa707a43269c0493719.png#pic_center =700x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/72f909f3c85057d680fd4e3e849a1dd2.png#pic_center =600x)

&#8195;&#8195;`PaLM-Coder 540B` 的性能进一步提高，在 HumanEval 上达到 88.4% pass@100，在 MBPP 上达到 80.8% pass@80。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0547b2604d89dd99c7de355b7b0a1928.png#pic_center =600x)
其它实验效果详见论文。
### 4.6 PaLM2（Google Research 2022.3 ）
详见[《Introducing PaLM 2》](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/)
## 五、 BLOOM（Google AI，2022.7）
> - 论文[《BLOOM: A 176B-Parameter Open-Access Multilingual Language Model》](https://paperswithcode.com/paper/bloom-a-176b-parameter-open-access)
> - [《BLOOM: 多语言大模型》](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490550&idx=1&sn=9b30122076f455759cfaf79465907653&chksm=9bc5f3a4acb27ab271a133fc31831fa3a9676be678ad8f8fb5eaf2be7d6a05c814d6904955e8&scene=178&cur_album_id=2878066965444362241#rd)
### 5.1 背景
&#8195;&#8195;大型语言模型（LLMs）已被证明能够根据少数示例或指令微调就可以执行新任务（zero-shot），但大多数LLMs是由资源充裕的组织开发的，且没有公开发布。因此，大多数的研究社区都被排除在LLMs的开发之外，也导致了一些结果，例如大多数LLMs主要是在英文文本上训练的。

&#8195;&#8195;为了推广这一技术，我们发布了`BLOOM`模型。`BLOOM`是一个在包含46种自然语言和13种编程语言的数百个数据源上训练的1760亿参数的多语言模型，由数百名研究人员合作开发和发布的。`BLOOM`使用了Transformer-decoder结构，在各种基准测试中取得了竞争性的性能。

&#8195;&#8195;为构建`BLOOM`，我们对其各个组成部分进行了彻底的设计过程，包括训练数据集（第3.1节）、模型架构和训练目标（第3.2节）以及分布式学习的工程策略（第3.4节）。我们还对模型的能力进行了分析（第4节）。我们的总体目标不仅是公开发布一个具有与最近开发的系统相当性能的大规模多语言语言模型，还要记录开发过程中采取的协调步骤（第2.2节）。本文的目的是提供对这些设计步骤的高级概述，并引用我们在开发BLOOM过程中产生的各个报告。
>&#8195;&#8195;`BLOOM`：BigScience Large Open-science Open-access Multilingual Language Model，大型开放多语言模型。
>&#8195;&#8195;`BigScience`：是一个开放的研究合作组织，其目标是公开发布LLM。
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/813ec14474b2a3655573b41da9436da7.png#pic_center =600x)
### 5.2 训练数据
#### 5.2.1 多语言语料库 ROOTS
&#8195;&#8195;`BLOOM`是在`ROOTS`语料库上进行训练的。该语料库是由498个Hugging Face数据集组成的综合集合，涵盖了1.61TB的文本，覆盖了46种自然语言和13种编程语言，其分布可见下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/90a7d5ef2c5eddc11ff88699ef8db1bc.png#pic_center =750x)<center> 左侧：46种自然语言的语系树状图，其中表面积与字节数成比例。印欧语系和汉藏语系占主导（1321.89 GB）。橙色的薄表面代表18GB的印度尼西亚语数据，绿色的矩形代表0.4GB的尼日尔-刚果语系子集。右侧：按大小分布的13种编程语言的华夫饼图，其中一个方块表示约200MB的数据量</center>

#### 5.2.2 xP3和指令数据集xP3mt
&#8195;&#8195;在公共提示池（P3）的子集上训练的T0证明了，在多任务提示数据集的混合上微调的语言模型具有强大的零样本任务泛化能力。在预训练BLOOM之后，我们采用了同样的大规模多任务微调方法，为BLOOM赋予了多语言零样本任务泛化能力。我们将结果模型称为`BLOOMZ`。下面是模型用到的几个数据集：

- `P3`：各种现有的和开源的英文自然语言数据集的提示集合，涵盖了各种自然语言任务，包括情感分析、问答和自然语言推理，并排除了有害内容或非自然语言，如编程语言。

- `xP3`：为了训练BLOOMZ，我们扩展了P3，得到了xP3，一个包含83个数据集的提示集合，涵盖46种语言和16个任务。xP3中的任务既可以是跨语言的（例如翻译），也可以是单语言的（例如摘要、问答）。我们使用PromptSource来收集这些提示，并为提示添加了额外的元数据
- `xP3mt`：为了研究多语言提示的重要性，我们还将xP3中的英语提示机器翻译为各自的数据集语言，从而生成了一个称为xP3mt的提示集合。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29cd199d65979a0c286a38cc98f8923a.png#pic_center =800x)<center> 图4：提示数据集`xP3`的语言分布与`ROOTS`紧密相关。</center>
### 5.3 模型结构
&#8195;&#8195;我们评估了各种模型结构，发现因果解码器模型表现最佳。我们进行了一系列实验，评估了位置编码和激活函数等对因果解码器模型的影响，最终我们在`BLOOM`中采用了两种改进：

- `ALiBi`位置嵌入：与将位置信息添加到嵌入层不同，ALiBi直接根据键和查询的距离减弱注意力得分。尽管ALiBi最初是基于对更长序列的外推能力的动机，但我们发现它在原始序列长度上也导致了更平滑的训练和更好的下游性能，优于原始transformer的和rotary embeddings这两种。

- `Embedding Layer Norm`：嵌入层归一化。在嵌入层后添加了额外的层归一化层，显著提高了训练的稳定性。
>&#8195;&#8195;请注意，初步的104B实验是在float16精度下进行的，而最终的训练是在bfloat16精度下进行的。自那时以来，float16被认为是导致训练LLM时观察到的许多不稳定性的原因之一（Zhang等人，2022；Zeng等人，2022）。bfloat16可能减轻了对嵌入层归一化的需求。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b82543d07fcee2e6db86d3420053b16a.png)

### 5.3 工程实现

BLOOM使用`Megatron-DeepSpeed`进行训练的，这是一个用于大规模分布式训练的框架。它由两个部分组成：
- Megatron-LM：提供了Transformer的实现，张量并行性和数据加载原语
- DeepSpeed提供了ZeRO优化器，模型流水线和通用的分布式训练组件。

&#8195;&#8195;这个框架使我们能够以高效的方式进行3D并行训练（三种互补的分布式训练方法的融合）。下面将介绍这些方法：
- `DP`：Data parallelism，数据并行。将模型复制多次，每个副本放置在不同的设备上，并提供数据的切片进行并行处理。在每个训练步骤结束时，所有模型副本都会进行同步。

- `TP`：Tensor parallelism，张量并行。将模型的各个层在多个设备上进行分区。这样，不再将整个激活或梯度张量放置在单个GPU上，而是将该张量的分片放置在不同的GPU上。这种技术有时被称为水平并行或层内模型并行。

- `PP`：Pipeline parallelism，流水线并行。将模型的层分割到多个GPU上，这样模型的每个GPU上只放置部分层，这有时被称为垂直并行。 使用bfloat16 混合精度，使用融合的 CUDA 内核

&#8195;&#8195;最后，ZeRO（零冗余优化器）允许不同的进程仅保存数据的一部分（参数、梯度和优化器状态）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9d1b5a8f8b287f32110a4e329084f0f2.png#pic_center =600x)

<center> 图6：DP+PP+TP的组合导致3D并行化。</center>

>&#8195;&#8195;该模型是在Jean Zay上进行训练的，Jean Zay是由法国国家计算中心（CNRS）的IDRIS运营的，由法国政府资助的超级计算机。
>&#8195;&#8195;训练BLOOM大约耗时`3.5`个月，共计消耗了1,082,990个计算小时。训练过程在48个节点上进行，每个节点配备8个NVIDIA A100 80GB的GPU（总共`384个GPU`）
>&#8195;&#8195;由于在训练过程中可能出现硬件故障，我们还保留了4个备用节点。每个节点配备2个AMD EPYC 7543 32核的CPU和512 GB的内存，存储由使用SpectrumScale（GPFS）并行文件系统的全闪存和硬盘驱动器混合处理，该文件系统在超级计算机的所有节点和用户之间共享。
>&#8195;&#8195;每个节点有4个NVLink GPU到GPU的互连通道用于节点内通信，每个节点有4个Omni-Path 100 Gbps的链路，按照增强的8D超立方体全局拓扑结构排列，用于节点间通信

### 5.4 实验&评测
1. SuperGLUE

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3cab27971060f19902576d86d6227dce.png#pic_center =800x)<center> 图7：各种LLM在`SuperGLUE`基准上zero-shot and one-shot prompt-based 效果</center>

2. HELM基准
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/15ee7e2474e02aa34bb7161c967b7fdc.png#pic_center =800x)

<center> 图10：各种语言模型在5次提示的HELM基准测试上的结果</center>

### 5.5  总结

BLOOM主要提升LLM的多语言能力，采用因果解码器的结构，但是做了AIBI位置编码和层归一化两方面的改进。文在还有很详细的数据集采集即过滤、训练调试等细节，感兴趣的可以看看。
## 六、 FLAN（Google 2022.10）
>- [《Flan-T5: One Model for ALL Tasks》](https://zhuanlan.zhihu.com/p/580468546)、[《LLM系列之FLAN-T5/PaLM》](https://blog.csdn.net/yanqianglifei/article/details/130568753)
>- 论文[《Scaling Instruction-Finetuned Language Models》](https://paperswithcode.com/paper/scaling-instruction-finetuned-language-models)、[FLAN-T5（Hugging Face）](https://huggingface.co/docs/transformers/model_doc/flan-t5)

&#8195;&#8195;`FLAN` 指的是（Instruction finetuning ），即"基于指令的微调"。通过在超大规模的任务上进行微调，可以大大提高语言模型的泛化性能，做到单个模型就可以在1800多个NLP任务上都能有很好的表现。这意味着模型一旦训练完毕，可以直接在几乎全部的NLP任务上直接使用，实现`One model for ALL tasks`。随后谷歌公开发布了大幅优于基线 T5 模型的 [Flan-T5模型](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)。
>基础模型可以进行任意的替换（需要有Decoder部分，所以不包括BERT这类纯Encoder语言模型）
### 6.1 摘要
&#8195;&#8195;通过使用一系列以指令形式表达的数据集来对语言模型进行微调，已经被证明可以提高模型的性能并提高其对未见任务的泛化能力。在本论文中，我们探讨了指令微调的几个关键方面：（1）扩展任务数量，（2）扩大模型规模，（3）链式思维数据的微调（CoT，chain-of-thought data）。我们的实验表明，指令微调在任务数量和模型规模方面都具有良好的扩展性。其次，经过CoT的指令微调会极大提高在CoT任务上的性能。
&#8195;&#8195;我们发现，使用上述方法进行指令微调显著改善了各种模型类别（PaLM、T5、U-PaLM）、启动设置（零样本、少样本、连续思考）和评估基准（MMLU、BBH、TyDiQA、MGSM、开放式生成）的性能。
&#8195;&#8195;例如，经过1.8K个任务的指令微调后，`Flan-PaLM 540B`在性能上大幅超过了`PALM 540B`（平均提高了9.4%），并在一些基准测试中取得了SOTA性能，如在五样本MMLU上达到了75.2%。我们还公开发布了`Flan-T5`的checkpoints，即使与规模更大的模型（如PaLM 62B）相比，它们在少样本情况下也具有强大的性能。总体而言，**指令微调是改善预训练语言模型性能和可用性的通用方法**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/498aa168c724afcc2df1bab86c346701.png#pic_center =600x)<center> 图1：我们对1.8K个以指令形式表达的任务进行了各种语言模型的微调，并在未见任务上对其进行评估。我们同时进行了带有示例和不带示例（即零样本和少样本）以及带有链式思维和不带链式思维的微调，从而在多个评估场景中实现了泛化。</center>
### 6.2 介绍
#### 6.2.1 Task mixtures
&#8195;&#8195;混合任务包括总共 1836 种指令任务，包括 473个 数据集，146 个任务类别，包括Muffin3、T0-SF、NIV2和CoT以及一些对话、程序合成和链式思维推理任务。所有数据源都是公开的，详情可见附录F。
>&#8195;&#8195;`CoT`微调混合：涉及CoT注释，我们使用它来探索在CoT注释上微调是否能够提高对未见推理任务的性能。我们从先前的工作中选择了九个数据集，人工评估者为这些数据集手动编写了CoT注释作为训练语料库。这九个数据集包括算术推理（Cobbe等，2021年）、多跳推理（Geva等，2021年）和自然语言推理（Camburu等，2020年）等任务。我们为每个任务手动编写了十个指令模板。数据卡片详见附录F。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9513467b301ad41e9cfe04a5940752ca.png#pic_center =600x)<center> 图2：我们的微调数据包括473个数据集、146个任务类别和1836个总任务。本文附录F中提供了用于本文的任务的详细信息。</center>
#### 6.2.2 模板和格式
&#8195;&#8195;因为需要用单个语言模型来完成超过1800+种不同的任务，所以需要将任务都转换成相同的“输入格式”喂给模型训练，同时这些任务的输出也需要是统一的“输出格式”。根据 “是否需要进行推理 （CoT）” 以及 “是否需要提供示例（Few-shot）” 可将输入输出划分成四种类型：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/90bd995853da88a4cdf3bb00c621aad7.png#pic_center =600x)<center>图3：本研究中的微调数据格式组合。分为是否带示例和是否使用CoT数据集。 </center>

| chain-of-thought | few-shot | 输入                                    | 输出        |
|------------------|----------|-----------------------------------------|-------------|
| ❎                | ❎        | 指令 + 问题                             | 答案        |
| ✅                | ❎        | 指令 + CoT引导（by reasoning step by step） + 问题 | 理由 + 答案 |
| ❎                | ✅        | 指令 + 示例问题 + 示例问题回答 + 指令 + 问题   | 答案        |
| ✅                | ✅        | 指令 + CoT引导 + 示例问题 + 示例问题理由 + 示例问题回答 + 指令 + CoT引导 + 问题 | 理由 + 答案 |

#### 6.2.3 微调过程
&#8195;&#8195;文本中，我们实验了`T5`（80M到11B）、`PaLM`（8B到540B）和`U-PaLM`（540B）等各种规模的模型，训练过程都是相同的。
&#8195;&#8195;我们采用恒定的学习率以及[Adafactor](https://arxiv.org/abs/1804.04235)优化器进行训练；同时会将多个训练样本“打包”成单个序列，这些训练样本直接会通过一个特殊的“解释。token”进行分割，每个模型的微调步骤数、学习率、批次大小和丢弃率详见附录E。
&#8195;&#8195;对于每个模型，我们选择一个检查点用于所有评估。我们通过定期评估保留任务的性能（每2k到10k步骤，具体取决于模型大小）来确定最佳步骤，并在所有消融运行中使用相同数量的检查点步骤。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a6f936c6f6152090aa93987754d1b2f8.png)<center> 表2：微调过程所需的计算量非常小。例如指令微调`Flan-PaLM 540B`时，只使用了预训练计算量的`0.2%`</center>

### 6.3 实验结果
1. 增加微调数据中的任务数量可以提升`Flan-PaLM`在大多数评估基准上的性能

&#8195;&#8195;如下表所示，评估基准包括MMLU（57个任务）、BBH（23个任务）、TyDiQA（8种语言）和MGSM（10种语言）。所有四个评估基准的评估指标都是few-shot提示准确率（完全匹配），我们对所有任务进行了平均值计算（不考虑权重）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2414ddc20bcaeda9b89f60d0969be703.png#pic_center =600x)<center> 表3：评估结果是MMLU-direct、MMLU-CoT、BBH-direct、BBH-CoT、TyDiQA和MGSM这6个的归一化平均值。。</center>

T5模型应用Flan效果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e30ff983d8bafc091b62a375eea20280.png#pic_center =600x)


2. 模型越大效果越好、任务越多效果越好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6ff3a1fdb79eb89384a7b873dc7a2802.png#pic_center =600x)
3. 、CoT数据可显著提高模型的推理能力（包括零样本）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a1c7b40f713a1d3e381acb70d97d83cb.png#pic_center =500x)<center>Figure 6: `PaLM`和`Flan-PaLM`在23个具有挑战性的BIG-Bench任务（BBH）上的zero-shot表现。Flan-PaLM通过“Let’s think step-by-step”激活了链式思维（CoT）生成。
 </center>

可以看到：

- 只有加入Flan训练之后的PaLM模型，CoT文本的加入才会带来效果提升；
- Flan本身也能够给模型带来足够的效果提升

### 6.4 总结
&#8195;&#8195;这篇工作提出了Flan的微调框架，核心有四点：统一的输入输出格式（4种类型），引入chain-of-thought，大幅提高任务数量，大幅提高模型体积；实现了用一个模型来解决超过1800种几乎全部的NLP任务，通过较低的成本，极大发掘了现有语言模型的泛化性能，让大家看到了通用模型的希望，即One Model for ALL Tasks。
## 七、 LLaMA系列（Meta AI） 
### 7.1 LLaMA（Meta AI 2023.2.24）
>- 论文：[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)、[Model Card（GitHub）](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
>- 官网博客[《Introducing LLaMA: A foundational, 65-billion-parameter large language model》](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) 
>- [LLM系列之LLaMA](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490547&idx=1&sn=f9c17e84481acf59b1a311d7d9e91c38&chksm=9bc5f3a1acb27ab778648272865f63ceaa9c49dca979f4efcf981540e5754b082d235d0414eb&scene=178&cur_album_id=2878066965444362241#rd)、[使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://mp.weixin.qq.com/s/r04BzCzYf29sxxqiLM84pg)
#### 7.1.1 背景

&#8195;&#8195;Hoffmann等人的[最新研究](https://arxiv.org/abs/2203.15556)表明，在给定的计算预算下，最佳性能并不是由最大的模型实现的，而是由在更多数据上训练的较小模型实现的。但是<font color='deeppink'>**在大模型时代，推理成本至关重要**</font >。在这种情况下，对于给定的性能水平，首选的模型不是训练速度最快的模型，而是推理速度最快的模型。尽管训练一个大模型以达到一定性能水平可能更便宜，<font color='deeppink'>但训练时间更长的较小模型最终在推理方面更经济</font >。

&#8195;&#8195;本研究的重点是训练一系列语言模型，在各种推理预算下实现最佳性能。通过在更多的token上进行训练，得到的模型称为`LLaMA`，参数范围从7B到65B，与现在最好的LLM相当。
- `LLaMA-13B` 仅以 1/10 规模的参数在多数的 benchmarks 上性能优于 GPT-3(175B)
- `LLaMA-65B` 与业内最好的模型 `Chinchilla-70B` 和 `PaLM-540B` 实力相当。
- 仅使用**公开数据集**即可部分复现最先进的性能（86%左右的效果）



#### 7.1.2. 预训练数据
&#8195;&#8195;我们的训练数据集是多个来源的混合，涵盖了不同的领域，但仅限于使用公开可用且与开源兼容的数据。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d61c8185501d6f28975abb017e14150a.png#pic_center =400x)<center> 表1：预训练数据（`1.4T tokens`）。表中列出了每个子集的采样比例、epoch数量和磁盘大小。在预训练1T个token时，采样比例相同</center>
#### 7.1.3 模型结构
整体结构仍然是`Transformer decoder`，但是做了三点改进：
- `RMSNorm Pre-normalizatio(GPT3)`：为了提高训练的稳定性，我们对每个Transformer子层的输入进行归一化，而不是对输出进行归一化，并使用了[RMSNorm](https://paperswithcode.com/paper/root-mean-square-layer-normalization)归一化函数；
- `SwiGLU(PaLM)`：将ReLU非线性激活函数替换为SwiGLU激活函数，详情参考本文4.2.1节；
- `Rotary Embeddings (GPTNeo)`：去除了绝对位置嵌入，并在网络的每一层中添加了旋转位置嵌入`RoPE`

相关代码（[llama/model.py](https://github.com/facebookresearch/llama/blob/main/llama/model.py)）：

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
    	# 作者对每个Transformer子层的输入进行归一化，而不是对输出进行归一化
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
```

#### 7.1.4 高效训练
1. 算法
我们的训练方法与Brown的`Flan`和Chowdhery的`PaLM`这些工作相似，并受到了[Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)的启发。
>&#8195;&#8195;	`DeepMind`发表的《Training Compute-Optimal Large Language Models》这篇论文，研究了在给定计算预算下，训练Transformer语言模型的最佳模型大小和训练tokens数量。通过在5亿到5000亿个tokens上训练超过400个语言模型，作者发现模型大小和训练令牌数应该等比例扩展：模型大小每翻倍，训练令牌数也应翻倍。而现有的趋势是是增加模型大小，通常不增加训练令牌数。例如`MT-NLG 530B`比`GPT-3 170B`大了三倍，但是训练令牌数大致相同，导致训练不足，性能明显低于相同计算预算下可能实现的水平。

>&#8195;&#8195;作者认为，在相同的计算预算下，一个在更多数据上训练的较小模型将表现更好，并训练了一个计算优化模型`Chinchilla(70B)`验证了这一假设。`Chinchilla(70B)`其在微调和推理时使用的计算资源大大减少，极大地方便了下游应用，但是在各种下游评估任务中，性能上显著优于`Gopher(280B)`、`GPT-3(175B)`、`Jurassic-1(178B)`和`Megatron-Turing NLG(530B)`这些更大规模的模型。
2. 训练参数
作者使用了`AdamW`优化器，并使用`cosine learning rate schedule`，使得最终学习率等于最大学习率的10%，设置0.1的权重衰减和1.0的梯度裁剪。warmup的step为2000，并根据模型的大小改变学习率和批处理大小：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1100013f5a7981e531d255778206490a.png#pic_center =600x)
3. 高效优化
	- 使用高效的因果多头注意力机制的实现来减少内存使用和运行时间，该实现可在[xformers](https://github.com/facebookresearch/xformers)库中获得
	- gradient checkpointing：使用checkpoint技术来减少在反向传播过程中需要重新计算的激活值数量。具体来说，我们保存了计算成本较高的激活值，例如线性层的输出，这是通过手动实现transformer层的backward函数来实现的，而不是依赖于PyTorch的autograd库。
	- 尽可能地重叠激活值的计算和GPU之间的通信(基于all_reduce操作)

&#8195;&#8195;当训练一个65B参数的模型时，我们的代码在具有80GB RAM的2048个A100 GPU上每秒处理大约380个标记。这意味着在包含1.4T个标记的数据集上进行训练需要大约21天的时间。训练损失的变化参考下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/23d28ba59c5cb437bc11013580494ae5.png#pic_center =400x)<center> 图1：7B、13B、33B和65B模型在训练过程中的训练损失，LLaMA-33B和LLaMA-65B模型是在1.4T个训练标记上训练的，较小的模型是在1.0T个训练标记上训练的。所有模型的batch size均为4M tokens</center>




#### 7.1.5 主要结果
1. 常识推理（Common Sense Reasoning）
`LLaMA-65B`在大多数任务中都优于`Chinchilla-70B`、`PaLM-540B`和`GPT-3`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4687a394135b025ef68d0108fbe23181.png#pic_center =600x)<center> 表3：在常识推理任务上的zero-shot表现</center>

2. 闭卷问答（Closed-book Question Answering）

| ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9d137ffeb01dd494fd5019402d997d20.png#pic_center =300x)| ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1b1c871199d01c8aefc4115d027a3616.png#pic_center =300x)|
|:---:|:---:|

3. 其它任务

| ![Image 1](https://i-blog.csdnimg.cn/blog_migrate/31416be89c4dc2bc457e60e13443dc02.png) | ![Image 2](https://i-blog.csdnimg.cn/blog_migrate/f97741849cc33e2f60f5b13e3d8da775.png) | ![Image 3](https://i-blog.csdnimg.cn/blog_migrate/775a8f0d02a9ba795b2da1672c752506.png) |
|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|
|阅读理解|数学推理|代码生成

4. 训练过程中的性能演变（Evolution of performance during training）
在训练过程中，我们对几个问题回答和常识推理基准进行了模型性能的跟踪，并在图2中进行了报告。在大多数基准测试中，性能稳步提高，并与模型的训练困惑度相关（参见图1）。不过，SIQA和WinoGrande是例外。特别值得注意的是，在`SIQA`上，我们观察到性能存在较大的变异，这可能表明该基准测试不够可靠。在`WinoGrande`上，性能与训练困惑度的相关性不太明显：LLaMA-33B和LLaMA-65B在训练过程中的性能相似。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cf7a98d7c82f1c8d1d777cef9a5c5f9e.png#pic_center =700x)
5. 指令调优
尽管`LLaMA-65B`的非微调版本已经能够遵循基本指令，但我们观察到非常少量的指令微调可以提高在MMLU上的性能，并进一步提高模型遵循指令的能力。指令微调得到的模型为`LLaMA-I`，在MMLU上达到了68.9%，但仍远未达到最先进的水平（MMLU上的GPT code-davincii-002为77.4）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/49a0d91b26955cf42ae0384520f8ea47.png#pic_center =300x)<center> 表10：指令微调 - MMLU（5-shot）。中等模型在MMLU上进行指令微调和不进行指令微调的比较</center>

#### 7.1.6 结论
&#8195;&#8195;本文中提出了一系列公开发布的语言模型，并实现与最先进的基础模型相竞争的结果。最值得注意的是，LLaMA-13B的性能优于GPT-3，但体积比GPT-3小10倍以上，LLaMA-65B与Chinchilla-70B和PaLM-540B竞争。

&#8195;&#8195;与之前的研究不同，论文的研究表明，不使用专有数据集，而只使用公开可用的数据集进行训练，可以达到最先进的性能。作者希望向研究界发布这些模型将加速大型语言模型的发展，并有助于提高它们的鲁棒性，减轻已知的问题，如毒性和偏见。

&#8195;&#8195;此外，作者像Chung等人一样观察到，根据指令对这些模型进行微调会产生有希望的结果计划在未来的工作中进一步研究这一点。

&#8195;&#8195;最后，作者计划在未来发布在更大的预训练语料库上训练的更大的模型，因为作者在扩展语料时已经看到了性能的不断提高
### 7.2 Alpaca（2023.3.13）
>- 博客[《Alpaca: A Strong, Replicable Instruction-Following Model》](https://crfm.stanford.edu/2023/03/13/alpaca.html)、 [stanford alpaca](https://github.com/tatsu-lab/stanford_alpaca)
>- 论文[《Self-Instruct: Aligning Language Models with Self-Generated Instructions》](https://paperswithcode.com/paper/self-instruct-aligning-language-model-with)、[知乎-论文解读贴](https://zhuanlan.zhihu.com/p/614916562)


&#8195;&#8195;`Alpaca 7B`是斯坦福大学在`LLaMA 7B`模型上经过52K个指令跟踪示范进行微调的模型，其性能比肩GPT-3.5（text-davinci-003），但是整个训练成本不到600美元。
>在8个80GB A100上训练了3个小时，不到100美元；使用OpenAI的API自动生成指令集，不到500美元

#### 7.2.1 背景
&#8195;&#8195;诸如`GPT-3.5`（text-davinci-003）、`ChatGPT`、`Claude`和`Bing Chat`等指令跟踪模型（Instruction-following models）的功能越来越强大，但它们仍然存在许多问题：它们可能生成虚假信息、传播社会刻板印象并产生有害语言。为了解决这些紧迫的问题，学术界的参与非常重要。不幸的是，由于没有能够与OpenAI的text-davinci-003等闭源模型相媲美的易于获取的模型，学术界在指令跟踪模型上进行研究一直很困难。

&#8195;&#8195;我们发布了一种名为Alpaca的指令跟踪语言模型，该模型基于Meta的`LLaMA 7B`模型，其微调的52K指令集是在text-davinci-003上自我指导式生成的（generated in the style of self-instruct  ）。在self-instruct evaluation set上，Alpaca表现出许多与OpenAI的text-davinci-003相似的行为，但其尺寸也惊人地小，易于复制且成本低廉。

&#8195;&#8195;我们发布了训练配方（training recipe）和数据，并计划在将来发布模型权重。我们还提供一个交互式演示，以便研究界更好地了解Alpaca的行为。Alpaca仅供学术研究使用，禁止任何商业用途。
- [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)：包含了我们用于对Alpaca模型进行微调的52K个指令跟随数据。这个JSON文件是一个字典列表，每个字典包含以下字段：
	- instruction: str，描述模型应执行的任务。这52K个指令中的每个指令都是独特的。
	- input: str，任务的可选上下文或输入。例如，当指令是“总结以下文章”时，输入是文章内容。大约40%的示例有一个输入。
	- output: str，由text-davinci-003生成的指令答案。
- [Data generation process](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process)：数据生成代码
- [Training code](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)：使用Hugging Face API进行微调的代码。

&#8195;&#8195;总结起来就是构建了self-instruct方法并通过指令微调实验证明了其有效性，发布了[alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)指令数据集，发布了`Alpaca`模型。评估显示，使用 `SELF -INSTRUCT` 对 GPT3 进行调整的性能明显优于使用现有的公共指令数据集，并且与 `InstructGPT 001` 的性能表现接近。
#### 7.2.2 self-instruct 概述
&#8195;&#8195;在一定的预算下训练高质量的指令跟踪模型面临两个重要挑战：一个是强大的预训练语言模型，另一个是高质量的指令跟踪数据。前者通过Meta最近发布的`LLaMA`模型得到解决，后者可以通过现有的强大语言模型自动生成指令数据来解决，即`self-instruct`半自动化方法。

&#8195;&#8195;该方法通过使用一个小的任务种子集作为任务池，从中随机选择任务来生成指令数据集。生成的数据经过过滤和筛选后，可以用于对语言模型进行指令调优，以提高其遵循指令的能力。

&#8195;&#8195;本文采用175个人工编写的instruction-output pairs作为种子池，指示模型（text-davinci-003）创造更广泛的指令集，每个pais都包含[指令, 输入, 输出]或[指令, 输出]。生成流程如下：

1. 使用模型生成新的指令，用于定义任务（通常是新的）；
2. 对该模型生成的指令判断是否分类任务；
3. 使用模型生成输入-输出实例，这些实例可以后续用于指令微调；
4. 对上述模型生成的数据进行过滤和后处理；

&#8195;&#8195;最后将经过过滤和后处理的数据添加到种子池中；一直重复上述步骤直到种子池有足够多的数据。通过简化生成流程（详见GitHub）大大降低成本，使得用OpenAI API生成52K指令集的成本不到500美元。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7e4e6f9649e88e855534884f7672be3d.png#pic_center =800x)<center> 图2：SELF-INSTRUCT的高级概述。</center>

&#8195;&#8195;在这个生成的数据上，我们对GPT3进行微调（即使用生成指令数据时使用的相同模型）构建了`GPT3 SELF-INST`。评估结果表明，在`SUPER NI`（典型NLP任务）上`GPT3 SELF-INST`的性能大幅超过GPT3（原始模型）（+33.1%），并几乎与`InstructGPT 001`的性能相当；在广泛的指令遵循能力上，与InstructGPT 001之间存在5%的差距。

&#8195;&#8195;我们还使用Hugging Face的训练框架对`LLaMA`模型进行了微调，利用了Fully Sharded Data Parallel和混合精度训练等技术，在8个80GB的A100上花费了3个小时，成本不到100美元。


#### 7.2.3 self-instruct 方法

&#8195;&#8195;一般的，指令数据集中的一条数据包含（指令, 输入, 输出）这三部分，例如：

```python
指令：写一篇关于以下主题的文章
输入：校园安全
输出：校园安全是学校必须高度重视的问题，涉及到师生的生命财产安全。建立安全意识，制定严格的规章制度，加强安全巡查和应急预案，是确保校园安全的关键。
```
&#8195;&#8195;不过很多时候"指令"和"输入"划分的不是那么清晰，例如上面也可以写成：

```python
指令：写一篇以校园安全为主题的文章
输出：校园安全是学校必须高度重视的问题，涉及到师生的生命财产安全。建立安全意识，制定严格的规章制度，加强安全巡查和应急预案，是确保校园安全的关键。
```
&#8195;&#8195;所以指令数据集中的一条数据可能包含三部分（指令, 输入, 输出），也可能只有两部分（指令, 输出）。

1. 生成指令
生成指令时，先从种子池中随机抽取6个人工编写的指令，再随机抽取2个之前步骤中模型生成的指令，总共8个指令。以如下表的模版格式组织之后，输入给模型，让模型输出一个新的指令。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3f4e3ff198650e3ba95b3c8668b6aeea.png#pic_center =400x)

2. 判断指令是否属于分类任务：在根据指令生成实例时，分类任务与非分类任务使用的prompt模版是不同的，下一点会详细讲解。
判断方法：在种子池中随机挑选12条分类指令和19条非分类指令，然后加上新生成的指令，以下表7的模版格式组织之后，输入给模型，让模型输出新生成的指令是否分类任务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/be02bf37df84fa5274be0322c2330f67.png#pic_center =600x)
3. 生成实例

在给定指令之后，生成（输入, 输出）这个实例对时还有两种策略：
- `Input-first`：输入优先策略，先生成输入，后生成输出。
- `Output-first`：输出优先策略，常用于分类任务。
输入优先的方式在生成输入时，偏向于只生成一个标签，尤其是指令对应着分类任务时，其输入里面偏向于只生成一个类别。输出优先就是为了一定程度上缓解该问题。

指令数据集的丰富度我们是希望越丰富越好，所以允许出现一个指令，多个输入的数据。

- **输入优先**：在种子池中随机抽取 k 条数据，以如下的prompt模版的形式组合之后，输入给模型，让模型为最后的指令生成相应的实例。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/46843944d61e66ca2ddfeaee44c1ae92.png#pic_center =600x)

- **输出优先**：在种子池中随机抽取 k 条在之前的步骤中已经标记为分类的数据，以如下的prompt模版的形式组合之后，输入给模型，让模型为最后的指令生成相应的实例。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c0d17bf482125315132e4a68961bb6df.png#pic_center =600x)
4. 过滤及后处理
- 为了数据的多样性，新生成的指令只有与种子池中的指令的 ROUGE-L 小于0.7时才会添加进入种子池；
- 排除一些无法被语言模型处理的指令，比如涉及图像、图片、图形的指令；
- 在给指令生成实例时，会过滤掉输入相同但是输出不同的实例

#### 7.2.5 alpaca_data.json指令集分析
1. 统计信息
下表描述了生成数据的基本统计信息。在过滤后，我们生成了总计超过52K个指令和82K多个对应这些指令的实例。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0ca02f7871204e65c1efe825d640f8a9.png#pic_center =400x)
2. 数据质量
评估方式：随机抽取200条指令，并给每个指令随机抽取一个实例，然后人工对该指令和实例进行标注评估，评估结果如下表2所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/815de9fc81d5928c230236aea6d54365.png#pic_center =400x)

- 生成的指令有含义，能表示一个任务的占比为92%；
- 给每个指令生成合适的输入的占比为79%；
- 生成的输出是指令和输入的正确结果的占比为58%；
- 指令、输入、输出，这三个字段全对的占比为54%；

#### 7.2.6 实验结果
1. SUPER NI benchmark：Self-Instruct能够给GPT3模型带来33.1%的巨大的提升，效果接近`InstructGPT 001`
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d44aa807cf33878605ccf128566a3fc4.png#pic_center =400x)<center> 表格3：在来自SUPER NI）的未见任务上的评估结果</center>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3e70626a4bee5295cdc2da6c4a6fedec.png)
2. 新新测评数据集SUPERNI。


&#8195;&#8195;为了更好的测试本文中提出的方法训练出的模型在给用户使用时的效果。本文设计了一份新的更贴近普通用户的数据集（252条指令，每个指令一个实例），在该数据集上测试Self-Instruct的效果。在设计这个数据集时考虑到的有：

- 不同的领域：邮件写作、社交媒体、生产力工具、娱乐、编程等；
- 形式上：可以是（指令, 输入, 输出），也可以是（指令, 输出）；
- 指令有的长、有的短，输入/输出中包含项目符号、表格、代码、方程等；

&#8195;&#8195;评估方式是人工对模型的输出结果做打分，评分A最好，评分D最差。在下图5中的颜色对应着绿色最好，红色最差。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f84b580a0ac3c859a995f68f69ff92dc.png#pic_center =600x)
结果显示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/88de8eb39a05d7cd787ef8c323ab39df.png)
### 7.3 Llama 2
>- 论文：[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
>- [Llama 2官网](https://ai.meta.com/llama/)、[Llama 2开源项目](https://github.com/facebookresearch/llama/tree/main)、[HuggingFace仓库](https://huggingface.co/meta-llama)
>- 博客：官方博客[Meta and Microsoft Introduce the Next Generation of Llama](https://ai.meta.com/blog/llama-2/)、[LLaMA 2技术细节详细介绍](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490574&idx=1&sn=4fdac3807e218e6f26ef0a4c1cb6841e&chksm=9bc5f45cacb27d4aef58278d9df9a28869b7baf01430db59dd4bddaa08471620429cd32c688a&scene=178&cur_album_id=2878066965930901510#rd) 、[《更强的Llama 2开源，可直接商用》](https://mp.weixin.qq.com/s/PJyFoLP7IBxjbswq-NBEkA)

&#8195;&#8195; Llama 可以说是 AI 社区内最强大的开源大模型，但因为开源协议问题，一直不可免费商用。近期，Meta 终于发布了大家期待已久的免费可商用版本 `Llama 2`。

&#8195;&#8195;此次 Meta 发布的 Llama 2 模型系列包含 7B、13B & 70B 三种参数变体。此外还训练了 340 亿参数变体，但并没有发布，只在技术报告中提到了。Llama 2相比Llama有以下升级：
- `Llama 2` 模型接受了 2 万亿个标记的训练，训练语料相比LLaMA多出40%
- `Llama 2`上下文长度是 `Llama 1` 的两倍（2048升→4096），可以理解和生成更长的文本。
- 发布了`LLaMA-2-chat` ，使用来自人类反馈的强化学习（超过 100 万个新的人类注释的训练）来确保安全性和帮助性。
-  70B模型采用分组查询注意力（GQA）


&#8195;&#8195;公布的测评结果显示，Llama 2 在包括推理、编码、精通性和知识测试等许多外部基准测试中都优于其他开源语言模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d49b5697c0b8ae690a92c0980d06d54d.png)
更多资源：[《关于 Llama 2 的一切资源，我们都帮你整理好了》](https://zhuanlan.zhihu.com/p/650614370)

### 7.4 Llama2-Chinese
>- [Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese)
>- [《首发！国内最大Llama开源社区发布首个预训练中文版Llama2》](https://mp.weixin.qq.com/s/JXyPAgJaX4GvvohJO_Nlyw)
>
&#8195;&#8195;7月31日，Llama中文社区[Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese)率先完成了国内首个真正意义上的中文版Llama2-13B大模型，从模型底层实现了Llama2中文能力的大幅优化和提升（不是微调！而是基于200B中文语料预训练！）。毋庸置疑，中文版Llama2一经发布将开启国内大模型新时代！

&#8195;&#8195;虽然Llama2的预训练数据相对于第一代扩大了一倍，但是中文预训练数据的比例依然非常少，仅占0.13%，这也导致了原版Llama2的中文能力较弱。

&#8195;&#8195;我们对于一些中文问题进行提问，发现大多数情况下Llama2都不能以中文回答，或者以中英文混杂的形式回答问题。因此，需要基于大规模中文数据对Llama2进行优化，使Llama2具备更好的中文能力。为此国内顶尖高校大模型博士团队创办了Llama中文社区[Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese)，开启了Llama2中文大模型训练征程。

以下是社区主要时间线：

- 2023年7月19日：正式启动Llama2模型的中文预训练，开启Llama2中文社区，国内下载地址正在启动。
- 2023年7月23日：Llama2中文微调参数发布至Hugging Face仓库`FlagAlpha`。
- 2023年7月31日：国内首个真正意义上的Llama2中文大模型发布，详细信息参见社区公众号文章。
- 2023年8月26日：新增`Code Llama`模型。
- 2023年8月28日：发布基于Llama2进行中文预训练的开源大模型`Atom-7B`
- 2023年9月12日：更新预训练版本Atom-7B和对话版本`Atom-7B-Chat`模型参数，最新的中文预训练数据量为100B token，训练进程见[llama.family](https://llama.family/)。



### 7.5 其它Llama 2项目
#### 7.5.1 Vicuna v1.5
>参考[《GPT-4最强平替更新！UC伯克利发布Vicuna v1.5》](https://mp.weixin.qq.com/s/OWPQtFaww1vAqLoQuTxI6Q)
>
&#8195;&#8195;自3月UC伯克利发布`Vicuna`以来，`Vicuna`就已成为最受欢迎的聊天LLM之一，它在多模态、AI安全和评估方面的研究具有开创性。此次基于全新的`Llama 2`，发布了更新版`Vicuna v1.5`，不仅支持4K和16K上下文，并且在几乎所有基准测试中取得了SOTA。目前基于Vicuna的优秀项目有
- [MiniGPT4](https://minigpt-4.github.io)
- [LLaVA](https://llava-vl.github.io)
- [LLM-Attacks](https://mp.weixin.qq.com/s/OWPQtFaww1vAqLoQuTxI6Q)：只要通过附加一系列特定的无意义token，就能生成一个神秘的prompt后缀。由此，任何人都可以轻松破解LLM的安全措施，生成无限量的有害内容。
- [Gorilla](https://github.com/ShishirPatil/gorilla)：Gorilla是一种基于LLaMA架构的大型语言模型，它可以生成合适的API调用，还可以快速添加新的领域知识，包括Kubernetes、GCP、AWS、OpenAPI等。
- [QLoRA](https://github.com/artidoro/qlora)：QLoRA，使用一种新的高精度技术将预训练模型量化为4位，然后添加一小部分可学习的低秩适配器权重。这些适配器权重通过量化权重的反向传播梯度进行调整。QLoRA方法证明了4位量化模型也可以进行有效的微调,达到与全精度模型相当的性能。
- [ToolLLaMA](https://github.com/OpenBMB/ToolBench)：开源LLM能够掌握数千种不同的现实世界API，并通过收集高质量的指令调优数据集来实现这一点。


#### 7.5.2 中文LLaMA-2 & Alpaca-2
>[《哈工大科大讯飞联合推出中文LLaMA-2 & Alpaca-2大语言模型》](https://mp.weixin.qq.com/s/sJ_imBdHCD4NibVy58EO2w)
>
&#8195;&#8195;本项目基于Meta发布的可商用大模型Llama-2开发，是中文LLaMA&Alpaca大模型的第二期项目，开源了中文LLaMA-2基座模型和Alpaca-2指令精调大模型。这些模型在原版Llama-2的基础上扩充并优化了中文词表，使用了大规模中文数据进行增量预训练，进一步提升了中文基础语义和指令理解能力，相比一代相关模型获得了显著性能提升。相关模型支持4K上下文并可通过NTK方法最高扩展至18K+。项目链接：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2。
- 针对Llama-2模型扩充了新版中文词表，开源了中文LLaMA-2和Alpaca-2大模型
- 开源了预训练脚本、指令精调脚本，用户可根据需要进一步训练模型
- 使用个人电脑的CPU/GPU快速在本地进行大模型量化和部署体验
- 支持transformers, llama.cpp, text-generation-webui, LangChain, vLLM等LLaMA生态
- 目前已开源的模型：Chinese-LLaMA-2-7B, Chinese-Alpaca-2-7B

## 八、 LLM系列之底座模型对比
>[LLaMA、Palm、GLM、BLOOM、GPT模型结构对比](https://mp.weixin.qq.com/s?__biz=MzAxOTU5NTU4MQ==&mid=2247490555&idx=2&sn=ff61b482d34095877f83ee213dbb4724&chksm=9bc5f3a9acb27abff5fd1e55dc6e55c810957fb0d2af5b254127a8866c305e39ff4bdb035a3e&scene=178&cur_album_id=2878066965444362241#rd)

1. LLama：

	-  使用`RMSNorm[GPT3]`对输入数据进行标准化，参考论文：Root mean square layer normalization；
	- 使用激活函数`SwiGLU [PaLM]`， 参考PALM论文：Glu variants improve transformer；
	- 使用`Rotary Embeddings`进行位置编码[GPTNeo]，参考论文 Roformer: Enhanced transformer with rotary position embedding；
	- 使用了AdamW优化器，并使用cosine learning rate schedule；
	- 使用因果多头注意的有效实现来减少内存使用和运行时间（xformers库实现）
2. Palm

	- `SwiGLU`激活函数：用于 MLP 中间激活，采用SwiGLU激活函数：用于 MLP 中间激活，因为与标准 ReLU、GELU 或 Swish 激活相比，《GLU Variants Improve Transformer》论文里提到：SwiGLU 已被证明可以显著提高模型效果
	- `Parallel Layers`：每个 Transformer 结构中的“并行”公式：与 GPT-J-6B 中一样，使用的是标准“序列化”公式。并行公式使大规模训练速度提高了大约 15%。消融实验显示在 8B 参数量下模型效果下降很小，但在 62B 参数量下没有模型效果下降的现象。
	- `Multi-Query Attention`：每个头共享键/值的映射，即“key”和“value”被投影到 [1, h]，但“query”仍被投影到形状 [k, h]，这种操作对模型质量和训练速度没有影响，但在自回归解码时间上有效节省了成本。
	- `RoPE embeddings`：使用的不是绝对或相对位置嵌入，而是RoPE，是因为 RoPE 嵌入在长文本上具有更好的性能 ，
	- `Shared Input-Output Embeddings`:输入和输出embedding矩阵是共享
3. GLM

	- Layer Normalization的顺序和残差连接被重新排列，
	-  使用单个线性层来进行输出Token预测
	- `ReLU`激活函数替换为`GELU`
	- 采用二维位置编码

4. BLOOM
	- 使用 `ALiBi` 位置嵌入，它根据键和查询的距离直接衰减注意力分数。 与原始的 Transformer 和 Rotary 嵌入相比，它可以带来更流畅的训练和更好的下游性能。ALiBi不会在词嵌入中添加位置嵌入；相反，它会使用与其距离成比例的惩罚来偏向查询键的注意力评分。图片
	-  `Embedding Layer Norm`：嵌入层归一化。在嵌入层后添加了额外的层归一化层，显著提高了训练的稳定性。
	- 使用了 25 万个标记的词汇表。 使用字节级 BPE，解码时永远不会产生未知标记。

5. GPT
GPT 使用 Transformer Decoder 结构，但是去掉了第二个 Mask Multi-Head Attention层。


## 九、补充：近期发布的其它LLMs
### 9.1 Zephyr-7B论文解析及全量训练、Lora训练
见帖子[《Zephyr-7B论文解析及全量训练、Lora训练》](https://blog.csdn.net/qq_56591814/article/details/134344019?spm=1001.2014.3001.5502)

