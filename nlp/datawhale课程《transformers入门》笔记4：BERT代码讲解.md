﻿BERT代码实现
@[toc]
## 前言
&#8195;&#8195;本文是复制datawhale关于transformer教程里的一章[《如何实现一个BERT》](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A03-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AATransformer%E6%A8%A1%E5%9E%8B%EF%BC%9ABERT/3.1-%E5%A6%82%E4%BD%95%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AABERT?id=%E5%89%8D%E8%A8%80)。本文包含大量源码和讲解，通过段落和横线分割了各个模块，同时网站配备了侧边栏，帮助大家在各个小节中快速跳转，希望大家阅读完能对BERT有深刻的了解。同时建议通过pycharm、vscode等工具对bert源码进行单步调试，调试到对应的模块再对比看本章节的讲解。

&#8195;&#8195;本篇章将基于[HuggingFace/Transformers, 48.9k Star](https://github.com/huggingface/transformers)进行学习。本章节的全部代码在[huggingface bert](https://github.com/huggingface/transformers/tree/master/src/transformers/models/bert)，注意由于版本更新较快，可能存在差别，请以4.4.2版本为准。
&#8195;&#8195;HuggingFace 是一家总部位于纽约的聊天机器人初创服务商，很早就捕捉到 BERT 大潮流的信号并着手实现基于 pytorch 的 BERT 模型。这一项目最初名为 pytorch-pretrained-bert，在复现了原始效果的同时，提供了易用的方法以方便在这一强大模型的基础上进行各种玩耍和研究。

&#8195;&#8195;随着使用人数的增加，这一项目也发展成为一个较大的开源社区，合并了各种预训练语言模型以及增加了 Tensorflow 的实现，并且在 2019 年下半年改名为 Transformers。截止写文章时（2021 年 3 月 30 日）这一项目已经拥有 43k+ 的star，可以说 Transformers 已经成为事实上的 NLP 基本工具。

本文基于 Transformers 版本 4.4.2（2021 年 3 月 19 日发布）项目中，pytorch 版的 BERT 相关代码，从代码结构、具体实现与原理，以及使用的角度进行分析。

## 一、Tokenization分词-BertTokenizer
### 1.1 Tokenization代码
和BERT 有关的 Tokenizer 主要写在models/bert/tokenization_bert.py中。
```python
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
class BertTokenizer(PreTrainedTokenizer):
    """
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
```
### 1.2 Tokenization代码讲解
&#8195;&#8195;BertTokenizer 是基于BasicTokenizer和WordPieceTokenizer的分词器：

&#8195;&#8195;BasicTokenizer负责处理的第一步——按标点、空格等分割句子，并处理是否统一小写，以及清理非法字符。
&#8195;&#8195;对于中文字符，通过预处理（加空格）来按字分割；同时可以通过never_split指定对某些词不进行分割；这一步是可选的（默认执行）。
&#8195;&#8195;WordPieceTokenizer在词的基础上，进一步将词分解为子词（subword）。
&#8195;&#8195;subword 介于 char 和 word 之间，既在一定程度保留了词的含义，又能够照顾到英文中单复数、时态导致的词表爆炸和未登录词的 OOV（Out-Of-Vocabulary）问题，将词根与时态词缀等分割出来，从而减小词表，也降低了训练难度；
&#8195;&#8195;例如，tokenizer 这个词就可以拆解为“token”和“##izer”两部分，注意后面一个词的“##”表示接在前一个词后面。 BertTokenizer 有以下常用方法：

 - from_pretrained：从包含词表文件（vocab.txt）的目录中初始化一个分词器；
 - tokenize：将文本（词或者句子）分解为子词列表；
 - convert_tokens_to_ids：将子词列表转化为子词对应下标的列表；
 - convert_ids_to_tokens ：与上一个相反；
 - convert_tokens_to_string：将 subword 列表按“##”拼接回词或者句子；
 - encode：对于单个句子输入，分解词并加入特殊词形成“[CLS], x, [SEP]”的结构并转换为词表对应下标的列表；对于两个句子输入（多个句子只取前两个），分解词并加入特殊词形成“[CLS], x1, [SEP], x2, [SEP]”的结构并转换为下标列表；
 - decode：可以将 encode 方法的输出变为完整句子。 以及，类自身的方法：
```python
bt = BertTokenizer.from_pretrained('bert-base-uncased')
bt('I like natural language progressing!')
# {'input_ids': [101, 1045, 2066, 3019, 2653, 27673, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
Downloading: 100%|██████████| 232k/232k [00:00<00:00, 698kB/s]
Downloading: 100%|██████████| 28.0/28.0 [00:00<00:00, 11.1kB/s]
Downloading: 100%|██████████| 466k/466k [00:00<00:00, 863kB/s]
```
```python
{'input_ids': [101, 1045, 2066, 3019, 2653, 27673, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

## 二、Model-BertModel
&#8195;&#8195;和 BERT 模型有关的代码主要写在/models/bert/modeling_bert.py中，这一份代码有一千多行，包含 BERT 模型的基本结构和基于它的微调模型等。

&#8195;&#8195;下面从 BERT 模型本体入手分析：
```python
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """
  ```
&#8195;&#8195;BertModel 主要为 transformer encoder 结构，包含三个部分：

 - embeddings，即BertEmbeddings类的实体，根据单词符号获取对应的向量表示；
 - encoder，即BertEncoder类的实体；
 - pooler，即BertPooler类的实体，这一部分是可选的。
注意 BertModel 也可以配置为 Decoder，不过下文中不包含对这一部分的讨论。

### 2.1BertModel 前向传播过程
介绍BertModel 前向传播过程中各个参数的含义以及返回值：
```python
def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ): ...
  ```

 - input_ids：经过 tokenizer 分词后的 subword 对应的下标列表；
 - attention_mask：在 self-attention 过程中，这一块 mask 用于标记 subword 所处句子和
   padding 的区别，将 padding 部分填充为 0；
 - token_type_ids：标记 subword 当前所处句子（第一句/第二句/ padding）；
 - position_ids：标记当前词所在句子的位置下标；
 - head_mask：用于将某些层的某些注意力计算无效化；
 - inputs_embeds：如果提供了，那就不需要input_ids，跨过 embedding lookup 过程直接作为 Embedding 进入 Encoder 计算；
 - encoder_hidden_states：这一部分在 BertModel 配置为 decoder 时起作用，将执行 cross-attention 而不是 self-attention；
 - encoder_attention_mask：同上，在 cross-attention 中用于标记 encoder 端输入的 padding；
 - past_key_values：这个参数貌似是把预先计算好的 K-V 乘积传入，以降低 cross-attention 的开销（因为原本这部分是重复计算）；
 - use_cache：将保存上一个参数并传回，加速 decoding；
 - output_attentions：是否返回中间每层的 attention 输出；
 - output_hidden_states：是否返回中间每层的输出；
 - return_dict：是否按键值对的形式（ModelOutput 类，也可以当作 tuple 用）返回输出，默认为真。
注意，这里的 head_mask 对注意力计算的无效化，和下文提到的注意力头剪枝不同，而仅仅把某些注意力的计算结果给乘以这一系数。

输出部分如下：
```python

 # BertModel的前向传播返回部分
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ) 
   ```

可以看出，返回值不但包含了 encoder 和 pooler 的输出，也包含了其他指定输出的部分（hidden_states 和 attention 等，这一部分在encoder_outputs[1:]）方便取用：
```python
        # BertEncoder的前向传播返回部分，即上面的encoder_outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
```
此外，BertModel 还有以下的方法，方便 BERT 玩家进行各种操作：
 - get_input_embeddings：提取 embedding 中的 word_embeddings 即词向量部分；
 - set_input_embeddings：为 embedding 中的 word_embeddings 赋值；
 - _prune_heads：提供了将注意力头剪枝的函数，输入为{layer_num: list of heads to prune in this layer}的字典，可以将指定层的某些注意力头剪枝。

&#8195;&#8195;**剪枝是一个复杂的操作，需要将保留的注意力头部分的 Wq、Kq、Vq 和拼接后全连接部分的权重拷贝到一个新的较小的权重矩阵（注意先禁止 grad 再拷贝），并实时记录被剪掉的头以防下标出错。具体参考BertAttention部分的prune_heads方法.**

### 2.2 BertPreTrainedModel完整代码
```python
from transformers.models.bert.modeling_bert import *
class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
 ```
### 2.3 BertEmbeddings
包含三个部分求和得到： Bert-embedding 图：Bert-embedding

1. word_embeddings，上文中 subword 对应的嵌入。
2. token_type_embeddings，用于表示当前词所在的句子，辅助区别句子与 padding、句子对间的差异。
3. position_embeddings，句子中每个词的位置嵌入，用于区别词的顺序。和 transformer 论文中的设计不同，这一块是训练出来的，而不是通过 Sinusoidal 函数计算得到的固定嵌入。一般认为这种实现不利于拓展性（难以直接迁移到更长的句子中）。
三个 embedding 不带权重相加，并通过一层 LayerNorm+dropout 后输出，其大小为(batch_size, sequence_length, hidden_size)。

**这里为什么要用 LayerNorm+Dropout 呢？为什么要用 LayerNorm 而不是 BatchNorm？可以参考一个不错的回答：transformer 为什么使用 layer normalization，而不是其他的归一化方法？**
```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
 ```
## 三、 BertEncoder
&#8195;&#8195;包含多层 BertLayer，这一块本身没有特别需要说明的地方，不过有一个细节值得参考：利用 gradient checkpointing 技术以降低训练时的显存占用。

&#8195;&#8195;gradient checkpointing 即梯度检查点，通过减少保存的计算图节点压缩模型占用空间，但是在计算梯度的时候需要重新计算没有存储的值，参考论文《Training Deep Nets with Sublinear Memory Cost》，过程如下示意图 gradient-checkpointing 图：gradient-checkpointing

&#8195;&#8195;在 BertEncoder 中，gradient checkpoint 是通过 torch.utils.checkpoint.checkpoint 实现的，使用起来比较方便，可以参考文档：torch.utils.checkpoint - PyTorch 1.8.1 documentation，这一机制的具体实现比较复杂，在此不作展开。

&#8195;&#8195;再往深一层走，就进入了 Encoder 的某一层：
```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
```
### 3.2 BertAttention
&#8195;&#8195;本以为 attention 的实现就在这里，没想到还要再下一层……其中，self 成员就是多头注意力的实现，而 output 成员实现 attention 后的全连接 +dropout+residual+LayerNorm 一系列操作。
```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
首先还是回到这一层。这里出现了上文提到的剪枝操作，即 prune_heads 方法：

 def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads) 
```
这里的具体实现概括如下：

 - find_pruneable_heads_and_indices是定位需要剪掉的 head，以及需要保留的维度下标 index；
 - prune_linear_layer则负责将 Wk/Wq/Wv 权重矩阵（连同 bias）中按照 index 保留没有被剪枝的维度后转移到新的矩阵。 接下来就到重头戏——Self-Attention 的具体实现。
```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
```
### 3.3 BertSelfAttention
预警：这一块可以说是模型的核心区域，也是唯一涉及到公式的地方，所以将贴出大量代码。

初始化部分：
```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
```
&#8195;&#8195;除掉熟悉的 query、key、value 三个权重和一个 dropout，这里还有一个谜一样的position_embedding_type，以及 decoder 标记；

&#8195;&#8195;注意，hidden_size 和 all_head_size 在一开始是一样的。至于为什么要看起来多此一举地设置这一个变量——显然是因为上面那个剪枝函数，剪掉几个 attention head 以后 all_head_size 自然就小了；

&#8195;&#8195;hidden_size 必须是 num_attention_heads 的整数倍，以 bert-base 为例，每个 attention 包含 12 个 head，hidden_size 是 768，所以每个 head 大小即 attention_head_size=768/12=64；
&#8195;&#8195;position_embedding_type 是什么？继续往下看就知道了.

&#8195;&#8195;然后是重点，也就是前向传播过程。

&#8195;&#8195;首先回顾一下 multi-head self-attention 的基本公式：

$$MHA(Q, K, V) = Concat(head_1, ..., head_h)W^O$$ $$head_i = SDPA(QW_i^Q, KW_i^K, VW_i^V)$$ $$SDPA(Q, K, V) = softmax(\frac{QK^T}{\sqrt(d_k)})V$$

&#8195;&#8195;而这些注意力头，众所周知是并行计算的，所以上面的 query、key、value 三个权重是唯一的——这并不是所有 heads 共享了权重，而是“拼接”起来了。

&#8195;&#8195;原论文中多头的理由为 Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. 而另一个比较靠谱的分析有：为什么 Transformer 需要进行 Multi-head Attention？

看看 forward 方法：
```python
def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # 省略一部分cross-attention的计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # ...
```
&#8195;&#8195;这里的 transpose_for_scores 用来把 hidden_size 拆成多个头输出的形状，并且将中间两维转置以进行矩阵相乘；

&#8195;&#8195;这里 key_layer/value_layer/query_layer 的形状为：(batch_size, num_attention_heads, sequence_length, attention_head_size)； 这里 attention_scores 的形状为：(batch_size, num_attention_heads, sequence_length, sequence_length)，符合多个头单独计算获得的 attention map 形状。

&#8195;&#8195;到这里实现了 K 与 Q 相乘，获得 raw attention scores 的部分，按公式接下来应该是按 $d_k$ 进行 scaling 并做 softmax 的操作。然而先出现在眼前的是一个奇怪的positional_embedding，以及一堆爱因斯坦求和：
```python
 # ...
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        # ...
```
&#8195;&#8195;关于爱因斯坦求和约定，参考以下文档：torch.einsum - PyTorch 1.8.1 documentation
&#8195;&#8195;对于不同的positional_embedding_type，有三种操作：

 - absolute：默认值，这部分就不用处理；
 - relative_key：对 key_layer 作处理，将其与这里的positional_embedding和 key 矩阵相乘作为 key 相关的位置编码；
 - relative_key_query：对 key 和 value 都进行相乘以作为位置编码。
回到正常 attention 的流程：
```python
# ...
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask  # 这里为什么是+而不是*？

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 省略decoder返回值部分……
        return outputs
```
&#8195;&#8195;重大疑问：这里的attention_scores = attention_scores + attention_mask是在做什么？难道不应该是乘 mask 吗？

&#8195;&#8195;因为这里的 attention_mask 已经【被动过手脚】，将原本为 1 的部分变为 0，而原本为 0 的部分（即 padding）变为一个较大的负数，这样相加就得到了一个较大的负值：
>至于为什么要用【一个较大的负数】？因为这样一来经过 softmax 操作以后这一项就会变成接近 0 的小数。
```python
(Pdb) attention_mask
tensor([[[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],
        [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],
        [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],
        ...,
        [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],
        [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],
        [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]]],
       device='cuda:0')
```
&#8195;&#8195;那么，这一步是在哪里执行的呢？ 在modeling_bert.py中没有找到答案，但是在modeling_utils.py中找到了一个特别的类：class ModuleUtilsMixin，在它的get_extended_attention_mask方法中发现了端倪：
```python
 def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # 省略一部分……

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
```
&#8195;&#8195;那么，这个函数是在什么时候被调用的呢？和BertModel有什么关系呢？ OK，这里涉及到 BertModel 的继承细节了：BertModel继承自BertPreTrainedModel，后者继承自PreTrainedModel，而PreTrainedModel继承自[nn.Module, ModuleUtilsMixin, GenerationMixin]三个基类。——好复杂的封装！

&#8195;&#8195;这也就是说，BertModel必然在中间的某个步骤对原始的attention_mask调用了get_extended_attention_mask，导致attention_mask从原始的[1, 0]变为[0, -1e4]的取值。

&#8195;&#8195;最终在 BertModel 的前向传播过程中找到了这一调用（第 944 行）：
```python
  # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
```
&#8195;&#8195;问题解决了：这一方法不但实现了改变 mask 的值，还将其广播（broadcast）为可以直接与 attention map 相加的形状。 不愧是你，HuggingFace。

&#8195;&#8195;除此之外，值得注意的细节有：

&#8195;&#8195;按照每个头的维度进行缩放，对于 bert-base 就是 64 的平方根即 8；
 - attention_probs 不但做了 softmax，还用了一次 dropout，这是担心 attention 矩阵太稠密吗…… 这里也提到很不寻常，但是原始 Transformer 论文就是这么做的；
 - head_mask 就是之前提到的对多头计算的 mask，如果不设置默认是全 1，在这里就不会起作用；
 - context_layer 即 attention 矩阵与 value 矩阵的乘积，原始的大小为：(batch_size,  - -num_attention_heads, sequence_length, attention_head_size) ；
 - context_layer 进行转置和 view 操作以后，形状就恢复了(batch_size, sequence_length, hidden_size)。
```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
```
### 3.4 BertSelfOutput
```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```
&#8195;&#8195;这里又出现了 LayerNorm 和 Dropout 的组合，只不过这里是先 Dropout，进行残差连接后再进行 LayerNorm。至于为什么要做残差连接，最直接的目的就是降低网络层数过深带来的训练难度，对原始输入更加敏感～
```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```
#### 3.4.1 BertIntermediate
&#8195;&#8195;看完了 BertAttention，在 Attention 后面还有一个全连接+激活的操作：
```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```
&#8195;&#8195;这里的全连接做了一个扩展，以 bert-base 为例，扩展维度为 3072，是原始维度 768 的 4 倍之多；

&#8195;&#8195;这里的激活函数默认实现为 gelu（Gaussian Error Linerar Units(GELUS）当然，它是无法直接计算的，可以用一个包含tanh的表达式进行近似（略)。
```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```
#### 3.4.2  BertOutput
&#8195;&#8195;在这里又是一个全连接 +dropout+LayerNorm，还有一个残差连接 residual connect：
```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```
&#8195;&#8195;这里的操作和 BertSelfOutput 不能说没有关系，只能说一模一样…… 非常容易混淆的两个组件。 以下内容还包含基于 BERT 的应用模型，以及 BERT 相关的优化器和用法，将在下一篇文章作详细介绍。
```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```
#### 3.4.3  BertPooler
&#8195;&#8195;这一层只是简单地取出了句子的第一个token，即[CLS]对应的向量，然后过一个全连接层和一个激活函数后输出：（这一部分是可选的，因为pooling有很多不同的操作）
```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```
```python
from transformers.models.bert.configuration_bert import *
import torch
config = BertConfig.from_pretrained("bert-base-uncased")
bert_pooler = BertPooler(config=config)
print("input to bert pooler size: {}".format(config.hidden_size))
batch_size = 1
seq_len = 2
hidden_size = 768
x = torch.rand(batch_size, seq_len, hidden_size)
y = bert_pooler(x)
print(y.size())
input to bert pooler size: 768
torch.Size([1, 768])
```
## 四、总结
### 1.1 BertTokenizer（Tokenization分词）
- 组成结构：BasicTokenizer和WordPieceTokenizer
- BasicTokenizer主要作用：
  1. 按标点、空格分割句子，对于中文字符，通过预处理（加空格方式）进行按字分割
  2. 通过never_split指定对某些词不进行分割
  3. 处理是否统一小写
  4. 清理非法字符
- WordPieceTokenizer主要作用：
  1. 进一步将词分解为子词(subword)，例如，tokenizer 这个词就可以拆解为“token”和“##izer”两部分，注意后面一个词的“##”表示接在前一个词后面
  2. subword介于char和word之间，保留了词的含义，又能够解决英文中单复数、时态导致的词表爆炸和未登录词的OOV问题
  3. 将词根和时态词缀分割，减小词表，降低训练难度  

- BertTokenizer常用方法：
  1. from_pretrained：从包含词表文件（vocab.txt）的目录中初始化一个分词器；
  2. tokenize：将文本（词或者句子）分解为子词列表；
  3. convert_tokens_to_ids：将子词列表转化为子词对应的下标列表；
  4. convert_ids_to_tokens ：与上一个相反；
  5. convert_tokens_to_string：将subword列表按“##”拼接回词或者句子；
  6. encode：
      - 对于单个句子输入，分解词，同时加入特殊词形成“[CLS], x, [SEP]”的结构，并转换为词表对应的下标列表；
      - 对于两个句子输入（多个句子只取前两个），分解词并加入特殊词形成“[CLS], x1, [SEP], x2, [SEP]”的结构并转换为下标列表；
  7. decode：可以将encode方法的输出变为完整句子。
  
### 1.2 BertModel
BERT 模型有关的代码主要写在`/models/bert/modeling_bert.py`中，包含 BERT 模型的基本结构和基于它的微调模型等。

BertModel 主要为 transformer encoder 结构，包含三个部分：
- embeddings，即BertEmbeddings类的实体，根据单词符号获取对应的向量表示；
- encoder，即BertEncoder类的实体；
- pooler，即BertPooler类的实体，这一部分是可选的。

BertModel可以作为编码器（只有自我注意）也可以作为解码器，作为解码器的时候，只需要在自注意力层之间添加了交叉注意力（应该还要加masked机制，屏蔽未来信息）
- BertModel常用方法：
  1. get_input_embeddings：提取 embedding 中的 word_embeddings，即词向量部分；
  2. set_input_embeddings：为 embedding 中的 word_embeddings 赋值；
  3. _prune_heads：提供了将注意力头剪枝的函数，输入为{layer_num: list of heads to prune in this layer}的字典，可以将指定层的某些注意力头剪枝。
### 1.3 BertEmbeddings
- 输出结果：通过word_embeddings、token_type_embeddings、position_embeddings三个部分求和，并通过一层 LayerNorm+Dropout 后输出得到，其大小为(batch_size, sequence_length, hidden_size)
- word_embeddings：子词(subword)对应的embeddings
- token_type_embeddings：用于表示当前词所在的句子，区别句子与 padding、句子对之间的差异
- position_embeddings：表示句子中每个词的位置嵌入，用于区别词的顺序

> 使用 LayerNorm+Dropout 的必要性：  
&emsp;&emsp;通过layer normalization得到的embedding的分布，是以坐标原点为中心，1为标准差，越往外越稀疏的球体空间中

&emsp;&emsp;词嵌入在torch里基于torch.nn.Embedding实现，实例化时需要设置的参数为词表的大小和被映射的向量的维度比如embed = nn.Embedding(10,8)。向量的维度通俗来说就是向量里面有多少个数。注意，第一个参数是词表的大小，如果你目前最多有8个词，通常填写10（多一个位置留给unk和pad），你后面万一进入与这8个词不同的词就映射到unk上，序列padding的部分就映射到pad上。

&emsp;&emsp;假如我们打算映射到8维（num_features或者embed_dim），那么，整个文本的形状变为100 x 128 x 8。接下来举个小例子解释一下：假设我们词表一共有10个词(算上unk和pad)，文本里有2个句子，每个句子有4个词，我们想要把每个词映射到8维的向量。于是2，4，8对应于batch_size, seq_length, embed_dim（如果batch在第一维的话）。

&emsp;&emsp;另外，一般深度学习任务只改变num_features，所以讲维度一般是针对最后特征所在的维度
### 1.4 BertEncoder
- 技术拓展：梯度检查点（gradient checkpointing），通过减少保存的计算图节点压缩模型占用空间

#### 1.4.1 BertAttention
分为BertSelfAttention+BertSelfOutput。后一个是Add+LayerNorm。
- BertSelfAttention
  1. 初始化部分：检查隐藏层和注意力头的参数配置倍率、进行各参数的赋值
  2. 前向传播部分：
      - multi-head self-attention的基本公式：
      $$
       \text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\ 
       \text{head}_i = \text{SDPA}(\text{QW}_i^Q, \text{KW}_i^K, \text{VW}_i^V) \\
       \text{SDPA}(Q, K, V) = \text{softmax}(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$$
      - transpose_for_scores：用于将 hidden_size 拆成多个头输出的形状，并且将中间两维转置进行矩阵相乘
      - torch.einsum：根据下标表示形式，对矩阵中输入元素的乘积求和
      - positional_embedding_type：  
          - absolute：默认值，不用进行处理
          - relative_key：对key layer处理
          - relative_key_query：对 key 和 value 都进行相乘以作为位置编码
- BertSelfOutput：  
&emsp;&emsp;前向传播部分使用LayerNorm+Dropout组合，残差连接用于降低网络层数过深，带来的训练难度，对原始输入更加敏感。
#### 1.4.2 BertIntermediate
self-attention输出Z全连接3072个神经元（以small是768词向量举例），得到一个扩维4倍的结果。
- 主要结构：全连接和激活操作
- 全连接：将原始维度进行扩展，参数intermediate_size
- 激活：激活函数默认为 gelu，使用一个包含tanh的表达式进行近似求解
#### 1.4.3 BertOutput
全连接768个神经元，映射回768维向量。之后Add+LayerNorm。
主要结构：全连接、dropout+LayerNorm、残差连接（residual connect）
### 1.5BertPooler
&#8195;&#8195;主要作用：取出句子的第一个token，即\[CLS\]对应的向量，然后通过一个全连接层和一个激活函数后输出结果。
### 1.6 总结
- BERT 分词级别不是单词。相反，它关注的是 WordPiece。 tokenizer（就是tokenization.py）会将你的单词转换为适合 BERT 的 wordPiece；
- 模型是在 modeling.py（class BertModel）中定义的，和普通的 Transformer encoder 完全相同；
- run_classifier.py 是微调网络的一个示例。它还构建了监督模型分类层。如果你想构建自己的- 分类器，请查看这个文件中的 create_model() 方法；
- 可以下载一些预训练好的模型。这些模型包括 BERT Base、BERT Large，以及英语、中文和包括 102 种语言的多语言模型，这些模型都是在维基百科的数据上进行训练的。


&#8195;&#8195;另外，值得注意的是，在 HuggingFace 实现的 Bert 模型中，使用了多种节约显存的技术——gradient checkpoint，不保留前向传播节点，只在用时计算；apply_chunking_to_forward，按多个小批量和低维度计算 FFN 部

&#8195;&#8195;BertModel 包含复杂的封装和较多的组件。以 bert-base 为例，主要组件如下：
 - 总计Dropout出现了1+(1+1+1)x12=37次；
 - 总计LayerNorm出现了1+(1+1)x12=25次；
 - 总计dense全连接层出现了(1+1+1)x12+1=37次，并不是每个dense都配了激活函数…… BertModel 有极大的参数量。以 bert-base 为例，其参数量为 109M。


