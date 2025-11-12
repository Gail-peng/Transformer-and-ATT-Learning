# Transformer-and-ATT-Learning

首先感谢文章：https://zhuanlan.zhihu.com/p/338817680 提供的思路和图片！

本项目用于记录Transformer模型的学习笔记，供以后的学习和参考。记笔记的过程也是思考的过程！

必须学习的论文是 Attention is All You Need：https://arxiv.org/abs/1706.03762

那么，Transformer能拿来干什么呢？ 

这里列举一些例子：

## 1. 自然语言处理（NLP）领域

这是 Transformer 应用最成熟、最广泛的领域，大量预训练模型（如 BERT、GPT、T5 等）均基于 Transformer 架构，支撑了多种细分任务：

- 机器翻译：这是 Transformer 最初的设计目标（论文《Attention Is All You Need》），可实现多语言间的自动翻译（如中英、英法翻译），主流模型包括 Google 的 Transformer 基础模型、T5 等。
- 文本分类：对文本按类别划分，如情感分析（判断文本积极 / 消极）、垃圾邮件检测、新闻主题分类（政治 / 体育 / 娱乐）等，BERT、RoBERTa 等模型在这类任务上表现优异。
- 序列标注：为文本中的每个 token（词 / 字）标注标签，例如命名实体识别（NER，识别人名、地名、机构名）、词性标注（标注名词 / 动词 / 形容词）、语义角色标注（分析句子中词的语义角色，如主语、宾语）等。
- 问答系统：包括抽取式问答（从文本中提取答案，如 SQuAD 数据集任务）和生成式问答（生成自然语言答案，如基于 GPT 的问答），BERT、T5 是常用模型。
- 文本生成：生成符合语法和语义的文本，例如机器写作（新闻、小说）、文本摘要（提炼长文本核心内容）、对话系统（聊天机器人，如 ChatGPT）、诗歌 / 代码生成等，GPT 系列、BART、T5 是核心模型。
- 文本相似度与检索：计算文本间的语义相似度，用于重复文本检测、语义检索（如 “搜索与 query 语义相似的文档”），Sentence-BERT 等模型通过生成句子嵌入实现此类任务。
- 跨语言任务：如跨语言文本分类（用一种语言的模型处理另一种语言的文本）、跨语言检索、多语言翻译等，多语言 BERT（mBERT）、XLM-R 等模型支持数百种语言。

## 2. 语音处理领域

传统语音任务多依赖 RNN，但 Transformer 凭借长距离依赖捕捉能力，逐渐成为主流：

- 语音识别（ASR）：将语音信号转换为文本，例如手机语音输入、会议实时转写，模型如 Google 的 Listen-Attend-Spell（LAS）结合 Transformer 改进，或纯 Transformer 模型（如 Conformer，融合 CNN 与 Transformer）。
- 语音合成（TTS）：将文本转换为自然语音，例如智能助手的语音输出，Tacotron 2 等模型结合 Transformer 提升合成自然度。
- 语音翻译：直接将一种语言的语音转换为另一种语言的文本（如 “英语语音→中文文本”），端到端模型常以 Transformer 为核心。

## 3. 计算机视觉（CV）领域

Transformer 打破了 CV 领域依赖 CNN 的传统，通过将图像视为 “序列”（如分割为 patches）实现高效建模：

- 图像分类：ViT（Vision Transformer）首次证明 Transformer 可直接处理图像分类任务，将图像分割为固定大小的 patches 作为输入序列，性能超越传统 CNN。
- 目标检测与分割：如 DETR（Detection Transformer）用 Transformer 替代传统锚框机制，直接输出目标边界框和类别；Mask Transformer 则用于语义分割（为图像每个像素标注类别）。
- 图像生成：结合扩散模型或自回归机制，生成高质量图像，例如 DALL-E 2、Midjourney 等文本到图像生成模型，核心依赖 Transformer 处理文本与图像的映射关系。
- 视频理解：处理视频的时空信息，如视频分类（判断视频类型，如 “篮球比赛”“演讲”）、动作识别（识别视频中人物动作，如 “跑步”“挥手”），模型如 Video Swin Transformer。

## 4. 多模态任务（跨文本、图像、语音等）

Transformer 擅长建模不同模态数据的关联，是多模态学习的核心架构：

- 图文检索：实现 “文本搜图像” 或 “图像搜文本”（如用 “一只猫坐在沙发上” 检索对应图片），CLIP、ALBEF 等模型通过 Transformer 对齐文本与图像的语义空间。
- 图像描述生成（Image Captioning）：为图像生成自然语言描述（如 “一只狗在草地上追飞盘”），模型如 Show-Attend-Tell 的改进版结合 Transformer。
- 视觉问答（VQA）：根据图像回答自然语言问题（如 “图中有几个人？”），Transformer 用于融合图像特征与问题文本特征。
- 跨模态生成：如 “文本→图像”“图像→文本”“语音→图像” 等，例如 DALL-E 生成图像、PALI（Google）支持多模态指令生成。

## 5. 其他领域

- 推荐系统：处理用户行为序列（如浏览、购买记录），捕捉长距离兴趣依赖，提升推荐准确性，例如 DIN（Deep Interest Network）的改进版引入 Transformer。
- 代码生成与理解：如 GitHub Copilot（基于 GPT）可生成代码、补全代码，或实现代码翻译（如 Python 转 Java），CodeBERT 等模型专注于代码领域。
- 分子生物学：处理 DNA、蛋白质序列，预测分子结构或功能，例如用 Transformer 分析基因序列的关联性。


# Transformer 的结构

<img width="741" height="502" alt="image" src="https://github.com/user-attachments/assets/ac40cea3-80fe-4270-8b99-ec0c1467aa3d" />

这是一张非常经典的 Transformer 模型结构图，各大帖子和网站已经贴烂了。

这个里面最重要的东西就是两个重要结构

- Encoder（编码器）模块： 图中左侧部分
- Decoder（解码器）模块： 图中右侧部分

可以看到，在编码器和解码器模块中，都包含了6个小的单元，见下图，对应的是编码器和解码器结构。这些结构会在后面的部分详细解释。

<img width="519" height="726" alt="image" src="https://github.com/user-attachments/assets/305c181a-984b-4de1-8a25-252f58bba245" />


接下来重要的事情就是先熟悉熟悉 Transformer 整体的工作流程


# Transformer 的工作流程


## 1.先处理输入数据

把输入的文本信息进行分词，之后会得到切分好的词语，这些词语通过使用简单的model进行编码，完成 **词语->向量** 的编码过程。

除此之外，还需要让模型知道词语在句子中的位置信息，因为位置代表了一定的语义关系，而这种关系是需要让模型学习到的。所以，每个词语还有其对应的位置embedding，完成 **位置信息->向量** 的编码过程。

最后就是词语向量信息的拼接，使输入的向量包含词语的表征信息和词语在句子中的位置信息，将其转化为矩阵，**矩阵的每一行都是一个词的向量表征，整个矩阵就是这个句子的向量表征**。整个过程如下图所示。

<img width="727" height="317" alt="image" src="https://github.com/user-attachments/assets/1cd76533-267d-413b-a7bc-c26562115e0f" />


## 2.句子表征输入Encoder模块

将上一步处理好的句子向量矩阵输入到编码器模块，向量矩阵会通过6个编码器结构得到所有词语的编码信息矩阵。

其中若输入向量矩阵为 n x m 维度（n为词语个数，d是向量编码维度），则每个编码器结构的输出，以及整个编码器模块的输出矩阵维度和输入矩阵维度完全相同 （n x m），**论文中的向量编码维度 m 为 512**。

<img width="589" height="817" alt="image" src="https://github.com/user-attachments/assets/6d683546-42ab-4282-af93-1b8dc7a5a6f8" />


## 3.Encoder模块的输出传送到Decoder模块解码

将上一步输出的包含更深层次语义信息的编码矩阵输入到Decoder模块中，Decoder模块会根据上一时刻的解码结果加上来自Encoder的编码矩阵实现对当前时刻信息的解码。

举例：以翻译场景为例，输入上一时刻“我有”的解码结果，结合来自Encoder模块的编码矩阵，Decoder模块会对当前时刻的信息进行解码，得到当前时刻的预测结果“苹果”，最后得到翻译结果是“我有苹果”。

<img width="677" height="395" alt="image" src="https://github.com/user-attachments/assets/f0ce3c2b-2c9f-410b-b9ef-69288d7438a1" />


# Transformer 中的细节


## 如何处理输入信息

1.**单词的Embedding**：可以使用多种模型，例如Word2Vec、Glove 等。

2.**单词的位置Embedding**：又叫做Position Embedding（PE），这个位置编码向量的维度和上面的单词向量的维度是一样的，因为要做向量的加法。Transformer中计算 PE 向量的方式是通过公式进行计算。

<img width="653" height="143" alt="image" src="https://github.com/user-attachments/assets/6a87f55f-859e-4c1f-8cbf-d909c7966bda" />

其中：

- pos为单词在句子中的位置（从 0 开始计数）
- d为向量编码的维度，和单词Embedding的维度相同
- i为位置编码的维度索引（从 0 开始）
- sin函数（2i）用于处理**单词位置编码向量**中的**偶数维度**的数值计算
- cos函数（2i+1）用于处理**单词位置编码向量**中的**奇数维度**的数值计算

举例：

<img width="820" height="88" alt="image" src="https://github.com/user-attachments/assets/be920ec3-cb7d-47cf-97a4-5f39752ca0d2" />

<img width="481" height="479" alt="image" src="https://github.com/user-attachments/assets/5193a73b-8fe3-4330-b498-0260ff453c79" />

<img width="581" height="479" alt="image" src="https://github.com/user-attachments/assets/5ca5bedf-ff40-42c1-bf4f-dd69e3537d72" />


<img width="422" height="138" alt="image" src="https://github.com/user-attachments/assets/e22b5004-f574-4fd5-998c-7fe5327a2a79" />


为什么这么做：

1. 这个算法能够处理不同长度的文本的Embedding，假设训练集里面最长的句子是有 50 个单词，使用时有一个长度为 60 的句子，则使用公式计算的方法可以计算出后续的 Embedding。

2.相对位置清晰，模型能够感知到距离关系：对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。


## 重中之重：Self-ATT （自注意力机制）

先上图，明确一下Self-ATT机制用在了模型中的哪个部分

<img width="519" height="726" alt="image" src="https://github.com/user-attachments/assets/4c8679d9-4a4e-453b-b1b6-17966d99d011" />

可以看到，在Transformer的编码器结构和解码器结构中都使用了 Self-ATT 机制。

其中呢，编码器结构使用了一个 Multi-Head Attention（**多个 Self-Attention组成**），解码层使用了一个 Masked Multi-Head Attention 和一个 Multi-Head Attention。

这些ATT机制是Transformer模型的重点，所以下面详细解释ATT相关的知识信息。


### Self-Attention 机制

在序列建模任务（如 NLP、CV）中，我们需要让序列中的每个元素（如单词、图像 patch）能够 “关注” 其他元素，从而捕捉元素间的语义 / 结构关联（即 “长距离依赖”）。

例如，在句子 “我喜欢吃苹果” 中，“我” 需要关注 “苹果” 才能理解动作的对象；在图像中，“猫” 的特征需要关注 “尾巴”“耳朵” 的特征才能完整建模。

自注意力的核心就是实现这种 **“动态关注”**—— 让每个元素根据自身与其他元素的 “相关性”，自适应地聚合信息。

如下图所示，Self-ATT 的结构是这样的

<img width="504" height="606" alt="image" src="https://github.com/user-attachments/assets/5ed5e844-73f1-4f94-811f-e25909fc50ec" />

在ATT计算过程中，需要先得到Q\K\V三个矩阵，实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。**而Q,K,V正是通过 Self-Attention 的输入进行线性变换得到的。**



#### Q\K\V 矩阵

Q、K、V 的设计源于信息检索的 “查询 - 键 - 值” 范式，我们可以类比为：

- 你想找一本关于 “人工智能” 的书（查询 Q）；
- 图书馆里的每本书都有一个 “标签”（如 “机器学习”“计算机视觉”）（键 K）；
- 找到匹配的 “标签” 后，你需要的是这本书的 “内容”（值 V）。

流程为 查询-匹配标签-找到标签对应值

输入转换：对于序列中的每个元素（如词嵌入向量x），通过三个可学习的权重矩阵 (W_Q、W_K、W_V)进行线性变换，生成三个向量：

<img width="240" height="143" alt="image" src="https://github.com/user-attachments/assets/1eeeb8df-47e1-4ca3-8d00-2e583ef96d24" />

**(W_Q、W_K、W_V) 是模型训练过程中学习到的参数，确保 Q、K、V 能适配任务需求**
 
注意力计算：**通过 Q 与 K 的相似度计算 “注意力分数”，再用分数加权 V，得到最终的注意力输出**


**为什么这么计算：**

为了判断 “哪些元素需要被关注”，我们需要计算Q 与所有 K 的相似度（即 “注意力分数”）。常用的计算方式是点积：

<img width="312" height="54" alt="image" src="https://github.com/user-attachments/assets/d5b190f1-1391-486e-b192-30fc757714be" />

**点积的结果越大，说明 Q 和 K 的语义 / 结构越相关。**

例如，“我” 的 Q 与 “苹果” 的 K 点积分数很高，说明两者语义关联强，需要重点关注。

Softmax 归一化：让注意力成为 “概率分布”

为了让分数具有可解释性（权重之和为 1），我们对所有注意力分数做Softmax 归一化：

<img width="407" height="74" alt="image" src="https://github.com/user-attachments/assets/ecd6b27b-d1fb-4205-9940-bb6b1892a1cb" />

除以<img width="46" height="40" alt="image" src="https://github.com/user-attachments/assets/91c639c9-3d5d-4e5c-9572-0e1aa3b35187" /> (d_k 是 K 的维度）是为了避免点积结果过大，导致 Softmax 后梯度消失（数值稳定性优化）。

最后，用归一化的注意力权重对所有 V 进行加权求和，得到该位置的自注意力输出：

<img width="361" height="70" alt="image" src="https://github.com/user-attachments/assets/74ebcdbc-3b71-4987-a675-fba963dec774" />











Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵WQ,WK,WV计算得到Q,K,V。计算如下图所示，注意 X, Q, K, V 的每一行都表示一个单词。

以上的线性变化是将高维度的句子Embedding输入矩阵（由单词Embedding拼接得到的）进行线性变换，将其转换为设定维度句子Embedding矩阵

<img width="527" height="750" alt="image" src="https://github.com/user-attachments/assets/e2bcc3dc-0651-4dfd-b233-1f77963d74c7" />












































