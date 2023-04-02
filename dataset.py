import torch
from torchvision import transforms
import pandas as pd
import os
from PIL import Image, ImageFile
import albumentations as A
import torchvision
import bisect
import ast
import numpy as np
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FlickerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path,transform = None,perc=1.0):
        #Naively, Load all caption data in memory
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.folder_path = folder_path
        self.caption_df = pd.read_csv(os.path.join(self.folder_path,'results.csv')).dropna(axis=0).drop_duplicates(subset="image")
        #Default transform handling
        if transform == None :
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return int(len(self.caption_df)*self.perc)

    def __getitem__(self, idx):

        imgname,caption,type_ = self.caption_df.iloc[idx,:]
        caption = caption
        img = Image.open(os.path.join(self.folder_path,'flickr30k_images',imgname))
        #img = np.asarray(img)
        img = self.transform(img)

        return torch.Tensor(img), caption
    
    def filter_df(self, name):
        self.caption_df = self.caption_df[self.caption_df['type'] == name]
        return self


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,img_folder,anon_path,transform,perc =1.0):
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.ds = torchvision.datasets.CocoCaptions(root=img_folder,annFile=anon_path,transform=transform)
    
    def __len__(self):
        return int(self.perc*len(self.ds))
    
    def __getitem__(self,idx):
        img,caption = self.ds[idx]
        return img, caption[0]

transform_train = \
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


transform_test = \
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self,root,train,transform):
        self.ds = torchvision.datasets.CIFAR100(root=root,train=train,download=True,transform=transform)
        self.class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.prompt = "A photo of "

    def __len__(self):
        return int(len(self.ds))

    def __getitem__(self,idx):
        #import pdb; pdb.set_trace()
        data = self.ds[idx]
        img = data[0]
        #import pdb;pdb.set_trace()
        txt = self.prompt + self.class_names[data[1]]
        return img,txt



class NFTDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, image_folder_path, image_lookback, tweet_lookback, transform=None, verbose=False):
        print("Init dataset")
        self.verbose = verbose
        self.folder_path = folder_path
        self.transactions_df = pd.read_csv(os.path.join(self.folder_path, 'final_price_movement.csv'))
        self.tweets_data = pd.read_csv(os.path.join(self.folder_path, 'final_tweet_data.csv')).dropna(axis=0)
        self.images_path = os.path.join(image_folder_path, 'updated_images')
        self.image_lookback = image_lookback
        self.tweet_lookback = tweet_lookback
        self.project_mapping = {'CyberKongz':'KONGZ', 
                                'CrypToadz by GREMPLIN': 'TOADZ',
                                'Cool Cats NFT': 'COOL',
                                'World of Women': 'WOW',
                                'BAYC': 'BAYC',
                                'MAYC': 'MAYC',
                                'FLUF World': 'FLUF',
                                'Pudgy Penguins': 'PPG'}
        if transform == None :
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return int(len(self.transactions_df))
    
    def __getitem__(self, idx):
        
        transaction_item = self.transactions_df.iloc[idx, :]
        transaction_project = transaction_item['project']
        transaction_timestamp = transaction_item['block_timestamp']
        tweets_tmp = self.tweets_data[
            (self.tweets_data['project'] == transaction_project) & (self.tweets_data['Datetime'] < transaction_timestamp)
        ].sort_values(
            by=['Datetime', 'LikeCount', 'RetweetCount'],
            ascending=False
        ).reset_index(drop=True)
        
        txt_list = tweets_tmp[:self.tweet_lookback]['preprocessed'].to_list()
        metadata = []
        
        # likes_list = tweets_tmp[:self.tweet_lookback]['LikeCount'].to_list()
        # retweets_list = tweets_tmp[:self.tweet_lookback]
        
        if len(tweets_tmp) > self.tweet_lookback:
            tweets_txt = tweets_tmp[:self.tweet_lookback]['preprocessed'].to_list()
            for idx, row in tweets_tmp[:self.tweet_lookback].iterrows():
                if row.polarity == 0:
                    metadata.append(1e-2*(row.LikeCount + row.RetweetCount))
                else:
                    metadata.append(row.polarity*(row.LikeCount + row.RetweetCount))
                
        elif len(tweets_tmp) == 0:
            tweets_txt = ['' for i in range(self.tweet_lookback)]
            metadata = [0 for i in range(self.tweet_lookback)]
        else:
            pad_len = self.tweet_lookback - len(tweets_tmp)
            tweets_txt = tweets_tmp['preprocessed'].to_list()
            
            for idx, row in tweets_tmp[:self.tweet_lookback].iterrows():          
                if row.polarity == 0:
                    met = (1e-2*(row.LikeCount + row.RetweetCount))
                else:
                    met = (row.polarity*(row.LikeCount + row.RetweetCount))
                if(idx == 0):
                    row0_meta = met
                metadata.append(met)
                
            for i in range(pad_len):
                tweets_txt.append(txt_list[0])
                metadata.append(row0_meta)
 
        transactions_tmp = self.transactions_df[
            (self.transactions_df['project'] == transaction_project) & (self.transactions_df['block_timestamp'] < transaction_timestamp) 
        ].sort_values(
            by=['block_timestamp'],
            ascending=False
        ).reset_index(drop=True)
        
        list_token_img = transactions_tmp['valid_token_img'].to_list()
        
        if len(transactions_tmp) > self.image_lookback:
            images_trans = transactions_tmp[:self.image_lookback]['valid_token_img'].to_list()
        elif len(transactions_tmp) == 0:
            images_trans = []
        else:
            pad_len = self.image_lookback - len(transactions_tmp)
            images_trans = transactions_tmp['valid_token_img'].to_list()
            for i in range(pad_len):
                images_trans.append(list_token_img[0])
        # print(images_trans)
        images = []
        for image_name in images_trans:
            # print(f"Image name = {image_name}")
            img = Image.open(os.path.join(self.images_path, image_name))
            img = self.transform(img)
            images.append(torch.Tensor(img))
        
        if len(images) == 0:
            images = [torch.zeros(3, 224, 224) for i in range(self.image_lookback)]
        
        # images = [torch.zeros(3, 224, 224) for i in range(self.image_lookback)]
        # return images, tweets_txt, transaction_item['label']
        
        return images, tweets_txt, metadata, transaction_item['label']
        
    def filter_df(self, name):
        
        self.transactions_df = self.transactions_df[self.transactions_df['type'] == name]
        self.tweets_data = self.tweets_data[self.tweets_data['type'] == name]

        return self