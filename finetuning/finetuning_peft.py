# 将多种微调方式添加进来，通过参数选择即可
from chatglm_model.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_model.tokenization_chatglm import ChatGLMTokenizer
from chatglm_model.configuration_chatglm import ChatGLMConfig
import torch
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
from chatglm_model.data_set import Seq2SeqDataSet, coll_fn
import os
from shutil import copy
import wandb
import numpy as np
from sklearn.metrics import accuracy_score
import json

wandb.init(project='chatGLM_6b_peft')

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/spo_0.json', type=str, help='')
    parser.add_argument('--model_dir', default="../hub/models--THUDM--chatglm-6b/", type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_freeze/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--input_column', type=str, default="text", help='训练数据的列')
    parser.add_argument('--output_column', type=str, default="answer", help='label')
    
    parser.add_argument('--tuning_mode', type=str, default="p-tuning v2", help='选择微调模式，有：p-tuning v1、p-tuning v2、freeze、lora')
    
    # 用于prompt微调
    parser.add_argument('--prompt_text', type=str,default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",help='')
    
    # 用于p-tuning微调
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    return parser.parse_args()


def main():
    args = set_args()
    
    if args.tuning_mode in ("p-tuning v1","p-tuning v2"):
        config = ChatGLMConfig.from_pretrained(args.model_dir)
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection
        model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir, config=config)
        
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["prefix_encoder"]):
                param.requires_grad = False
    else:
        model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir).half().cuda()
        
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
                param.requires_grad = False
            
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)
    
    
    if args.prefix_projection:
        model.gradient_checkpointing_enable()
    
    
    # 读取并修改deepspeed配置
    with open('./config/freeze_cfg.json', 'r') as f:
        data=json.load(f)
    conf =json.loads(json.dumps(data).replace('TRUE','True'))
    conf["train_micro_batch_size_per_gpu"]=args.train_batch_size
    conf["gradient_accumulation_steps"]=args.gradient_accumulation_steps
    conf["steps_per_print"]=args.log_steps
    

    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text,args.input_column,args.output_column)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    train_loss, train_acc, batch_num = 0, 0, 0
    
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
                train_loss += loss.item()
                #logits = outputs[1].detach().cpu().numpy()
                #predict = np.argmax(logits, axis=1)
                #label_ids = labels.detach().cpu().numpy()
                #train_acc += accuracy_score(label_ids, predict)
                batch_num += 1
        
        print("train loss:{}".format(train_loss/batch_num))
        # 添加wandb监测
        wandb.log({"lossN": train_loss/batch_num,"global_step":global_step})
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        
        if tuning_mode in ("p-tuning v1","p-tuning v2"):
            model.save_pretrained(save_dir)
        else:
            model_engine.save_pretrained(save_dir)
        
        copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=3 deepspeed --master_port 6666 finetuning_freeze.py
