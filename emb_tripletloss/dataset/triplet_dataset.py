from torch.utils.data import Dataset
import torch
import random


class TripletDataset(Dataset):
    def __init__(self, feature, labels, train=True, transform=None, device='cpu'):
        self.is_train = train
        self.transform = transform
        self.device = device

        if self.is_train:
            self.feature = feature.to(self.device)

            self.labels = labels.to(self.device)
            self.index = torch.arange(0, len(labels), device=self.device)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, item):
        anchor_feature = self.feature[item]
        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.labels == anchor_label]

            positive_item = random.choice(positive_list)
            while positive_item == item:
                positive_item = random.choice(positive_list)

            positive_feature = self.feature[positive_item]

            negative_list = self.index[self.labels != anchor_label]
            negative_list = negative_list[negative_list != item]

            negative_item = random.choice(negative_list)
            negative_feature = self.feature[negative_item]

            if self.transform:
                anchor_feature = torch.tensor(anchor_feature).float()
                positive_feature = torch.tensor(positive_feature).float()
                negative_feature = torch.tensor(negative_feature).float()
        return anchor_feature, positive_feature, negative_feature, anchor_label
