# coding: UTF-8

import torch
import argparse
from ast import Raise
from config import Config
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from my_transformer import MyEncoderTransformer

parser = argparse.ArgumentParser(description="Chinese Text Classification By BertForSequenceClassification from Huggingface or by MyEncoderTransformer from zhangfaen!")
parser.add_argument("--data_dir", type=str, default="./data_judge", help="training data and saved model path")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", help="pretrained bert model path")
parser.add_argument("--model_type", type=str, default="MyEncoderTransformer", help="BertForSequenceClassification or MyEncoderTransformer")

args = parser.parse_args()

def main():
    config = Config.createDefaultConfig(args.data_dir, args.model_type)
    Config.getDefaultConfig().logger.info("Config is:\n" + str(config))
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
    model:BertForSequenceClassification | MyEncoderTransformer = None
    if args.model_type == "BertForSequenceClassification":
        bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
        Config.getDefaultConfig().logger.info("using BertForSequenceClassification from huggingface")
        Config.getDefaultConfig().logger.info("bert_config is:\n" + str(bert_config))
        model = BertForSequenceClassification(bert_config)
    elif args.model_type == "MyEncoderTransformer":
        Config.getDefaultConfig().logger.info("using MyEncoderTransformer from zhangfaen")
        model = MyEncoderTransformer(MyEncoderTransformer.get_default_hyparams(vocab_size=21128, num_labels=config.num_labels))
    else:
        Raise("Invalid model type argument, it must be MyEncoderTransformer or BertForSequenceClassification!")
    
    model.to(config.device)

    model_total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            Config.getDefaultConfig().logger.info(name + ": " + str(param.numel()))
            model_total_params += param.numel()
    Config.getDefaultConfig().logger.info("Total number of trainable parameters: " + str(model_total_params))

    model.load_state_dict(torch.load(config.saved_model))
    model.eval()
    while True:
        sentence = input("请输入文本或者按Ctrl+C结束:\n")
        inputs = tokenizer(
            sentence, 
            max_length=config.seq_len,
            truncation="longest_first",
            return_tensors="pt")
        inputs = inputs.to(config.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            label = torch.max(logits.data, 1)[1].tolist()
            print("分类结果:" + config.label_list[label[0]])

if __name__ == "__main__":
    main()
