import copy
import os

from myparser import myargs

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_data import PrepareData
from model.attention import MultiHeadedAttention
from model.position_wise_feedforward import PositionwiseFeedForward
from model.embedding import PositionalEncoding, Embeddings
from model.transformer import Transformer
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.generator import Generator
from lib.criterion import LabelSmoothing
from lib.optimizer import NoamOpt
from train import train
from evaluate import evaluate

def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    # 创建深拷贝函数的别名
    c = copy.deepcopy
    # 创建多头注意力机制组件
    attn = MultiHeadedAttention(h, d_model).to(myargs.device)
    # 创建前馈神经网络组件
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(myargs.device)
    # 创建位置编码组件
    position = PositionalEncoding(d_model, dropout).to(myargs.device)
    # 构建模型
    model = Transformer(
        # 创建编码层
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(myargs.device), N).to(myargs.device),
        # 创建
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout).to(myargs.device), N).to(myargs.device),
        # 创建源语言的嵌入层和位置编码
        nn.Sequential(Embeddings(d_model, src_vocab).to(myargs.device), c(position)),
        # 创建目标语言的嵌入层和位置编码
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(myargs.device), c(position)),
        # 创建生成器，用于输出最后的结果
        Generator(d_model, tgt_vocab)).to(myargs.device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(myargs.device)

def main():
    # 数据预处理
    data = PrepareData()
    myargs.src_vocab = len(data.en_word_dict)
    myargs.tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % myargs.src_vocab)
    print("tgt_vocab %d" % myargs.tgt_vocab)

    # 初始化模型
    model = make_model(
                        myargs.src_vocab,
                        myargs.tgt_vocab,
                        myargs.layers,
                        myargs.d_model,
                        myargs.d_ff,
                        myargs.h_num,
                        myargs.dropout
                    )

   
    if myargs.type == 'train':
        # 训练
        print(">>>>>>> start train")
        criterion = LabelSmoothing(myargs.tgt_vocab, padding_idx = 0, smoothing= 0.0)
        optimizer = NoamOpt(myargs.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        
        train(data, model, criterion, optimizer)
        print("<<<<<<< finished train")
    elif myargs.type == "evaluate":
        # 预测
        # 先判断模型有没有训练好(前提)
        if os.path.exists(myargs.save_file):
            # 加载模型
            model.load_state_dict(torch.load(myargs.save_file))
            # 开始预测
            print(">>>>>>> start evaluate")
            evaluate(data, model)         
            print("<<<<<<< finished evaluate")
        else:
            print("Error: pleas train before evaluate")
    else:
        print("Error: please select type within [train / evaluate]")

if __name__ == "__main__":
    main()