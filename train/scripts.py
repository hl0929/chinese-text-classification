import os
import sys
sys.path.append(".")
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter
from utils.utils import get_time_diff


def init_network(model, method="xavier", exclude="embedding", seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.perf_counter()
    
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # learning rate decay
    # lr = gamma * lr = 0.9 * lr for per epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0  # number of batch for loss descent
    flag = False      # stop flag
    log_dir = os.path.join(config.log_path, time.strftime("%m-%d_%H.%M", time.localtime()))
    writer = SummaryWriter(log_dir=log_dir)
    # epoch
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        
        for _, (inputs, labels) in enumerate(train_iter):
            outputs = model(inputs)
            model.zero_grad()  # Sets gradients of all model parameters to zero
            loss = F.cross_entropy(outputs, labels)
            loss.backward()    # Computes the sum of gradients of given tensors 
            optimizer.step()   # Performs a single optimization step (parameter update)
            # evaluate
            if total_batch % 100 == 0:
                ground = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(ground, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_diff = get_time_diff(start_time)
                msg = 'Step: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_diff, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        scheduler.step() # decay
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.perf_counter()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_diff = get_time_diff(start_time)
    print("Time usage:", time_diff)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    total_loss = 0
    labels_all = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in data_iter:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(data_iter), report, confusion
    return acc, total_loss / len(data_iter)
