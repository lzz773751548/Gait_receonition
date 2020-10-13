from __future__ import print_function, division
import torch
import numpy as np
from gait_siamese import siamese
import torch.nn as nn
from torch.autograd import Variable
from data_loader import RescaleT
# from data_loader import ToTensofLab
from data_loader import GaitDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import torch.nn.functional as F
import cv2


# ------- 1. define loss function --------
#
# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     # def forward(self, output1, output2, label):
#     #     euclidean_distance = F.pairwise_distance(output1, output2)
#     #     loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
#     #                                   (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#     #
#     #     return loss_contrastive
#     def forward(self, y_true, y_pred):
#         loss_contrastive = torch.mean((1 - y_true) * torch.pow(self.margin - y_pred, 2) + y_true * torch.pow(y_pred,2))
#         return loss_contrastive

def train_model(model,train_dataloader,optimizer,loss_function):

    model.train()
    running_loss = []
    for i, data in enumerate(train_dataloader):
        left, right, label = data

        # wrap them in Variable
        if torch.cuda.is_available():
            left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()

        # y zero the parameter gradients
        optimizer.zero_grad()
        predict = model(left_v, right_v)
        loss = loss_function(predict, label_v)

        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
    loss = np.mean(np.array(running_loss))
    return loss

def valid_model(model,valid_dataloader,loss_function):
    # 切换模型为预测模型
    model.eval()
    running_loss = []
    label_list = []
    predict_list = []
    # 不记录模型梯度信息
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            left, right, label = data
            if torch.cuda.is_available():
                left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()
            predict = model(left_v, right_v)
            loss = loss_function(predict, label_v)
            running_loss.append(loss.item())
            # label = np.array(label)
            for j in range(len(label)):
                label_list.append(label[j])
                predict_list.append(predict[j])
    loss = np.mean(np.array(running_loss))
    valid_acc = np.mean(np.array(label) == (np.array(predict.cpu())>=0.5))
    return loss,valid_acc



if __name__ == '__main__':

    epoch_num = 500
    batch_size = 256
    save_path = "./weight"
    # ------- define model --------
    model = siamese()
    if torch.cuda.is_available():
        model.cuda()


    # ------- define optimizer --------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-08)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    # -------  define loss function --------
    loss_function = nn.BCELoss(size_average=True)


    # -----------------------set train data-----------------------
    train_data_list = []
    train_label_list = []
    with open("./train_data.txt", "r") as txt:
        for line in txt.readlines():
            line.strip()
            train_data_list.append([line.split(",")[0],line.split(",")[1]])
            train_label_list.append(int(line.split(",")[2]))


    train_dataset = GaitDataset(
        img_name_list=train_data_list,
        label_list=train_label_list,
        transform=transforms.Compose([
            # RescaleT((126,126)),
            transforms.ToTensor()])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    # -----------------------set valid data-----------------------
    valid_data_list = []
    valid_label_list = []
    with open("./valid_data.txt", "r") as txt:
        for line in txt.readlines():
            line.strip()
            valid_data_list.append([line.split(",")[0],line.split(",")[1]])
            valid_label_list.append(int(line.split(",")[2]))

    valid_dataset = GaitDataset(
        img_name_list=valid_data_list,
        label_list=valid_label_list,
        transform=transforms.Compose([
            # RescaleT((126,126)),
            transforms.ToTensor()])
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    # -----------------------train and valid -----------------------
    best_test_acc = 0
    best_test_loss = 100
    for epoch in range(epoch_num):
        train_loss = train_model(model,train_dataloader,optimizer,loss_function)
        valid_loss ,valid_acc = valid_model(model,valid_dataloader,loss_function)
        print("[epoch: {}/{},train loss: {:.4f},valid loss: {:.4f}, valid acc:{:.2f}%".format(epoch+1,epoch_num,train_loss,valid_loss,valid_acc*100))
        if valid_acc >= best_test_acc and valid_loss <= best_test_loss:
            best_test_acc = valid_acc
            best_test_loss = valid_loss
            torch.save(model.state_dict(),save_path+"/epoch:{}-valid_loss{:.4f}-valid_acc{:.2f}%.pth".format(epoch+1,valid_loss,valid_acc*100))







