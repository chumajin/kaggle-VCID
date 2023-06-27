from loadlibrary import *

def getaug(cfg):

    if cfg.in_chans != 3:
        
        aug = {
            "train": A.Compose([
                    A.Resize(cfg.imsize, cfg.imsize),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.ShiftScaleRotate(p=0.75),
                A.OneOf([
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        ], p=0.4),

                A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0.8),
                #  A.CLAHE(p=0.8),
                A.CoarseDropout(max_holes=1, max_width=int(cfg.imsize * 0.3), max_height=int(cfg.imsize * 0.3),
                                mask_fill_value=0, p=0.5),
                A.RandomGamma(p=0.8),
                # A.Cutout(max_h_size=int(size * 0.6),
                #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
                A.Normalize(
                    mean= [0] * cfg.in_chans,
                    std= [1] * cfg.in_chans

                ),

                ToTensorV2(transpose_mask=True)], p=1.),

            "valid": A.Compose([
                A.Resize(cfg.imsize, cfg.imsize),
                A.Normalize(
                    mean= [0] * cfg.in_chans,
                    std= [1] * cfg.in_chans
                ),

                ToTensorV2(transpose_mask=True)], p=1.)
        }

    else:
        aug = {
            "train": A.Compose([
                    A.Resize(cfg.imsize, cfg.imsize),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.ShiftScaleRotate(p=0.75),
                A.OneOf([
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        ], p=0.4),

                A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0.8),
                #  A.CLAHE(p=0.8),
                A.CoarseDropout(max_holes=1, max_width=int(cfg.imsize * 0.3), max_height=int(cfg.imsize * 0.3),
                                mask_fill_value=0, p=0.5),
                A.RandomGamma(p=0.8),
                # A.Cutout(max_h_size=int(size * 0.6),
                #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0
                    ),

                ToTensorV2(transpose_mask=True)], p=1.),

            "valid": A.Compose([
                A.Resize(cfg.imsize, cfg.imsize),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0
                    ),

                ToTensorV2(transpose_mask=True)], p=1.)
        }

    return aug


class PytorchDataSet(Dataset):
    
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return {"img":image, "label":label}


def make_dataloader(train_images,valid_images,valid_images_2,valid_images_3,train_masks,valid_masks,valid_masks_2,valid_masks_3,cfg):

    aug = getaug(cfg)

    train_dataset = PytorchDataSet(train_images,train_masks,aug["train"])
    valid_dataset = PytorchDataSet(valid_images,valid_masks,aug["valid"])
    train_dataloader = DataLoader(train_dataset,batch_size=cfg.train_batch,shuffle =True,num_workers=8,pin_memory=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=cfg.valid_batch,shuffle =False,num_workers=8,pin_memory=True)

    valid_dataset_2 = PytorchDataSet(valid_images_2,valid_masks_2,aug["valid"])
    valid_dataloader_2 = DataLoader(valid_dataset_2,batch_size=cfg.valid_batch,shuffle =False,num_workers=8,pin_memory=True)

    valid_dataset_3 = PytorchDataSet(valid_images_3,valid_masks_3,aug["valid"])
    valid_dataloader_3 = DataLoader(valid_dataset_3,batch_size=cfg.valid_batch,shuffle =False,num_workers=8,pin_memory=True)

    return train_dataloader,valid_dataloader,valid_dataloader_2,valid_dataloader_3

class testPytorchDataSet(Dataset):
    
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            data = self.transform(image=image)
            image = data['image']

        return image