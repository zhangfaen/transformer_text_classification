# coding: UTF-8

from ast import Raise
import os
import torch
import argparse
import numpy as np
import random
from config import Config
from datasets import DataProcessor
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from my_transformer import MyEncoderTransformer
from torch.optim import AdamW
from sklearn import metrics

def eval(model:BertForSequenceClassification | MyEncoderTransformer, config:Config, iterator:DataProcessor):
    model.eval()

    total_loss = 0
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    with torch.no_grad():
        for batch, labels in iterator:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels)

            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss
            true = labels.data.cpu().numpy()
            pred = torch.max(logits.data, 1)[1].cpu().numpy()
            all_labels = np.append(all_labels, true)
            all_preds = np.append(all_preds, pred)
    
    acc = metrics.accuracy_score(all_labels, all_preds)
    report = metrics.classification_report(all_labels, all_preds, target_names=config.label_list, digits=4, zero_division=0)
    confusion = metrics.confusion_matrix(all_labels, all_preds)
    return acc, total_loss / len(iterator), report, confusion
    

def test(model:BertForSequenceClassification | MyEncoderTransformer, config:Config, iterator:DataProcessor):
    Config.getDefaultConfig().logger.info(f"Testing on test data...")
    model.load_state_dict(torch.load(config.saved_model))
    acc, loss, report, confusion = eval(model, config, iterator)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    Config.getDefaultConfig().logger.info(msg.format(loss, acc))
    Config.getDefaultConfig().logger.info("Precision, Recall and F1-Score...")
    Config.getDefaultConfig().logger.info("\n" + str(report))
    Config.getDefaultConfig().logger.info("Confusion Matrix...")
    Config.getDefaultConfig().logger.info("\n" + str(confusion))


def train(model:BertForSequenceClassification | MyEncoderTransformer, config:Config, train_iterator:DataProcessor, eval_iterator:DataProcessor, test_iterator:DataProcessor):
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    last_improve = 0
    break_flag = False
    best_eval_loss = float('inf')

    for epoch in range(config.num_epochs):
        Config.getDefaultConfig().logger.info(f"Epoch [{epoch + 1}/{config.num_epochs}]. In cur epoch, data samples: {train_iterator.total_samples()}, batches: {len(train_iterator)}, batch size: {config.batch_size}")
        for _, (inputs, labels) in enumerate(train_iterator):
            optimizer.zero_grad()
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                labels=labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            loss.backward()
            optimizer.step()

            if total_batch % config.log_batch == 0:
                true = labels.data.cpu()
                pred = torch.max(logits.data, 1)[1].cpu()
                acc = metrics.accuracy_score(true, pred)
                eval_acc, eval_loss, _, _ = eval(model, config, eval_iterator)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(model.state_dict(), config.saved_model)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""

                msg = f'Batches done: {total_batch:>4}. For cur batch, TrainLoss: {loss.item():>4.2}, TrainAcc: {acc:>4.1%}, ValLoss: {eval_loss:>4.2}, ValAcc: {eval_acc:>4.1%} {improve}'
                Config.getDefaultConfig().logger.info(msg)
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                Config.getDefaultConfig().logger.info("No improvement for a long time, auto-stopping...")
                break_flag = True
                break
        if break_flag:
            break
    
    test(model, config, test_iterator)

parser = argparse.ArgumentParser(description="Chinese Text Classification By BertForSequenceClassification from Huggingface or by MyEncoderTransformer from zhangfaen!")
parser.add_argument("--data_dir", type=str, default="./data_judge", help="training data and saved model path")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", help="pretrained bert model path")
parser.add_argument("--model_type", type=str, default="MyEncoderTransformer", help="BertForSequenceClassification or MyEncoderTransformer")

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(137)
    config = Config.createDefaultConfig(args.data_dir, args.model_type)
    Config.getDefaultConfig().logger.info("Config is:\n" + str(config))

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)

    model:BertForSequenceClassification | MyEncoderTransformer = None
    if args.model_type == "BertForSequenceClassification":
        bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
        Config.getDefaultConfig().logger.info("using BertForSequenceClassification huggingface pretrained model, then finetuning it")
        Config.getDefaultConfig().logger.info("bert_config is:\n" + str(bert_config))
        # model = BertForSequenceClassification(bert_config)
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(args.pretrained_bert_dir, "pytorch_model.bin"),
            config=bert_config
        ) # type: ignore
    elif args.model_type == "MyEncoderTransformer":
        Config.getDefaultConfig().logger.info("using MyEncoderTransformer model, training from scratch")
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

    train_iterator = DataProcessor(config.train_file, config.device, tokenizer, config.batch_size, config.seq_len)
    eval_iterator = DataProcessor(config.eval_file, config.device, tokenizer, config.batch_size, config.seq_len)
    test_iterator = DataProcessor(config.test_file, config.device, tokenizer, config.batch_size, config.seq_len)
    train(model, config, train_iterator, eval_iterator, test_iterator)

if __name__ == "__main__":
    main()
