# -*- coding: utf-8 -*-
"""
Created on 2023-06-06 (Tue) 00:34:19

reference
https://venoda.hatenablog.com/entry/2020/10/11/221117

@author: I.Azuma
"""

#%%
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu' # 12.20s/it
#DEVICE = 'cpu' # 00:31<06:06, 16.62s/it

# 前処理クラス
class ImageTransform(object):
    """
    入力画像の前処理クラス
    画像のサイズをリサイズする
    
    Attributes
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize):
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor()
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor()
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)
    
# Datasetクラス
class DogDataset(data.Dataset):
    """
    犬種のDataseクラス。
    PyTorchのDatasetクラスを継承させる。
    
    Attrbutes
    ---------
    file_list: list
        画像のファイルパスを格納したリスト
    classes: list
        犬種のラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証化を設定
    """
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)
        
        # 画像ラベルをファイル名から抜き出す
        label = self.file_list[index].split('/')[-2].split('_')[1]
        
        # ラベル名を数値に変換
        label = self.classes.index(label)
        
        return img_transformed, label

# ネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),        # (batch, 64, r, r) # r=resize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                # (batch, 64, r/2, r/2) # r=resize
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),      # (batch, 128, r/2, r/2) # r=resize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                # (batch, 128, r/4, r/4) # r=resize
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),     # (batch, 256, r/4, r/4) # r=resize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                # (batch, 256, r/8, r/8) # r=resize
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),     # (batch, 128, r/8, r/8) # r=resize
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=7)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
"""
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=128 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=7)
    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        
        return x
"""
# 各種パラメータの用意
# クラス名
degree_classes = ['N', 'PB','UDH','ADH','FEA','DCIS','IC']

# リサイズ先の画像サイズ
resize = 32

# バッチサイズの指定
batch_size = 128

# エポック数
num_epochs = 30

# 2. 前処理
# 学習データ、検証データのファイルパスを格納したリストを取得する
train_path = '/workspace/datasource/BRACS/previous_versions/Version1_MedIA/Images/train/'
val_path = '/workspace/datasource/BRACS/previous_versions/Version1_MedIA/Images/val/'
# 1. get train image path
subdirs = os.listdir(train_path)
train_file_list = []
for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
    train_file_list += glob(os.path.join(train_path, subdir, '*.png'))

# 2. get val image path
subdirs = os.listdir(val_path)
valid_file_list = []
for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
    valid_file_list += glob(os.path.join(val_path, subdir, '*.png'))

# 3. Datasetの作成
train_dataset = DogDataset(
    file_list=train_file_list, classes=degree_classes,
    transform=ImageTransform(resize),
    phase='train'
)

valid_dataset = DogDataset(
    file_list=valid_file_list, classes=degree_classes,
    transform=ImageTransform(resize),
    phase='valid'
)

# 4. DataLoaderの作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=32, shuffle=False)

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader, 
    'valid': valid_dataloader
}

batch_iterator = iter(dataloaders_dict['train'])
inputs, labels = next(batch_iterator)


# 5. ネットワークの定義
net = Net()
net.to(DEVICE)
# 6. 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 7. 最適化手法の定義
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 8. 学習・検証
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-------------')
    
    for phase in ['train', 'valid']:
        if phase == 'train':
            # 学習モードに設定
            net.train()
        else:
            # 訓練モードに設定
            net.eval()
            
        # epochの損失和
        epoch_loss = 0.0
        # epochの正解数
        epoch_corrects = 0
        
        #for inputs, labels in dataloaders_dict[phase]:
        for _ in tqdm(range(len(dataloaders_dict[phase]))):
            batch_iterator = iter(dataloaders_dict['train'])
            inputs, labels = next(batch_iterator)

            inputs=inputs.to(DEVICE)
            labels=labels.to(DEVICE)
            # optimizerを初期化
            optimizer.zero_grad()
            
            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                
                outputs = net(inputs)
                # 損失を計算
                loss = criterion(outputs, labels)
                
                # ラベルを予測
                _, preds = torch.max(outputs, 1)
                
                # 訓練時は逆伝搬の計算
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    
                    # パラメータ更新
                    optimizer.step()
                    
                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)
                
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))