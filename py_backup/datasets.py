from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from autoaugment import ImageNetPolicy
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset

class CustomDataset(Dataset):
    def __init__(self, csv, transforms, is_low = False, is_test=False, debug=False):
        if debug:
            csv = csv[::500]
            
        
        self.is_test = is_test
        if is_low:
            self.path = csv['img_path'].values
        
        else:
            self.path = csv['upscale_img_path'].values

        if not is_test:
            self.class_ = csv['label'].values

        self.transform = transforms

    def __getitem__(self, idx):
        img = np.array(Image.open(self.path[idx]).convert('RGB'))
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)

        if not self.is_test:
            y = self.class_[idx]

            return img, y
        
        else:
            return img
        

    
    def __len__(self):
        return len(self.path)


def get_transforms_AutoAug(is_low):
    size = 224 if is_low else 224

    train_transforms = transforms.Compose([
        transforms.Resize(size), 
        transforms.RandomHorizontalFlip(), 
        ImageNetPolicy(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    return train_transforms, valid_transforms

def get_loader(cfg, is_test=False):
    if cfg['use_kfold']:
        df = pd.read_csv('train.csv')
        kfold = KFold(n_splits=3, shuffle=True, random_state=1020)
        
        train_transforms, valid_transforms = get_transforms_AutoAug(cfg['is_low'])

        # K-Fold의 각 분할에 대해 DataLoader 리스트를 초기화합니다.
        train_loaders = []
        valid_loaders = []
        
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(df)):
            # 훈련 및 검증 데이터프레임을 생성합니다.
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]

            encoder = LabelEncoder()
            train_df['label'] = encoder.fit_transform(train_df['label'])
            valid_df['label'] = encoder.transform(valid_df['label'])
            
            # CustomDataset을 사용하여 훈련 및 검증 데이터셋을 생성합니다.
            train_dataset = CustomDataset(train_df, train_transforms, cfg['is_low'], False, cfg['debug'])
            valid_dataset = CustomDataset(valid_df, valid_transforms, cfg['is_low'], False, cfg['debug'])
            
            # DataLoader 인스턴스를 생성합니다.
            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            
            train_loaders.append(train_loader)
            valid_loaders.append(valid_loader)

        return train_loaders, valid_loaders
        
    else:
        if is_test:
            test_df = pd.read_csv('test.csv')
            test_dataset = CustomDataset(test_df, valid_transforms, cfg['is_low'], is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

        else:
            df = pd.read_csv('train.csv')
            train_df, valid_df = train_test_split(df, test_size=0.2, random_state=1020, stratify=df['label'])

            encoder = LabelEncoder()
            train_df['label'] = encoder.fit_transform(train_df['label'])
            valid_df['label'] = encoder.transform(valid_df['label'])

            train_transforms, valid_transforms = get_transforms_AutoAug(cfg['is_low'])
            train_dataset = CustomDataset(train_df, train_transforms, cfg['is_low'], False, cfg['debug'])
            valid_dataset = CustomDataset(valid_df, valid_transforms, cfg['is_low'], False, cfg['debug'])

            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    

    if not is_test:
        return train_loader, valid_loader
    
    else:
        return test_loader
    
    return None