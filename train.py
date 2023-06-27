import argparse
from pprint import pprint
import shutil

import time

from util import *
from loadlibrary import *
from dataset_loader import *
from metric import *
from model import *
from make_gt import *
from train_valid import *


# preprocess function

def read_image_mask(fragment_id,mid,irmode=False):
        
    images = []

    amari = cfg.in_chans % 2


    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2 + amari
    idxs = range(start, end)


    for i in tqdm(idxs):

        if irmode:
          image = cv2.imread(f"{cfg.inputpath}/{fragment_id}/ir.png", 0)
        else:
          image = cv2.imread(f"{cfg.inputpath}/{fragment_id}/surface_volume/{i:02}.tif", 0)


        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)


    images = np.stack(images, axis=2)


    mask = cv2.imread(f"{cfg.inputpath}/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0 # signal 1

    mask2 = cv2.imread(f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
    mask2 = np.pad(mask2, [(0, pad0), (0, pad1)], constant_values=0)

    mask2 = mask2.astype('float32')
    mask2 /= 255.0 # signal 1

    return images, mask, mask2
  
def get_train_valid_dataset(mid,irmode=False):
      
    frags = os.listdir(cfg.inputpath)
    frags.sort()
    print(frags)


    if cfg.valid_id not in frags:
          assert("your original valid id does not find. please select --originalvalidid exist validid")

    train_images = []
    train_masks = []
    train_masks2 = []


    valid_images = []
    valid_masks = []
    valid_xyxys = []

    valid_masks2 = []


    for fragment_id in frags:

        if irmode:
          image, mask,mask2 = read_image_mask(fragment_id,mid,irmode=True)
        else:
          image, mask,mask2 = read_image_mask(fragment_id,mid)

        x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.stride))
        y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size

                if fragment_id == str(cfg.valid_id):
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_masks2.append(np.sum(mask2[y1:y2, x1:x2])) 
                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
                    train_masks2.append(np.sum(mask2[y1:y2, x1:x2])) 
                    
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys,train_masks2,valid_masks2

def makedata():
  
  # 0. makedata

  print(cfg.uselayers)
  
  # make images
  train_images, train_masks, valid_images, valid_masks, valid_xyxys,train_masks2,valid_masks2 = get_train_valid_dataset(cfg.uselayers[0])
  print(len(train_images))
  train_images_2, train_masks_2, valid_images_2, valid_masks_2, valid_xyxys_2,train_masks2_2,valid_masks2_2 = get_train_valid_dataset(cfg.uselayers[1])
  train_images_3, train_masks_3, valid_images_3, valid_masks_3, valid_xyxys_3,train_masks2_3,valid_masks2_3 = get_train_valid_dataset(cfg.uselayers[2])
  if cfg.useir:
      train_images_ir, train_masks_ir, valid_images_ir, valid_masks_ir, valid_xyxys_ir,train_masks2_ir,valid_masks2_ir = get_train_valid_dataset(28,irmode=True)

  # 1. add images
  train_images.extend(train_images_2)
  train_masks.extend(train_masks_2)
  train_masks2.extend(train_masks2_2)
  del train_masks_2,train_masks2_2
  gc.collect()

  train_images.extend(train_images_3)
  train_masks.extend(train_masks_3)
  train_masks2.extend(train_masks2_3)
  del train_masks_3,train_masks2_3
  gc.collect()

  if cfg.useir:
    train_images.extend(train_images_ir)
    train_masks.extend(train_masks_ir)
    train_masks2.extend(train_masks2_ir)
    del train_masks_ir,train_masks2_ir
    gc.collect()

  # 2. filter images (np.sum(mask) is not using)

  train_images = [image for image,judge in tqdm(zip(train_images,train_masks2)) if judge != 0]
  train_masks = [mask for mask,judge in tqdm(zip(train_masks,train_masks2)) if judge != 0]
  valid_images = [image for image,judge in tqdm(zip(valid_images,valid_masks2)) if judge != 0]
  valid_masks = [mask for mask,judge in tqdm(zip(valid_masks,valid_masks2)) if judge != 0]
  valid_xyxys = [image for image,judge in tqdm(zip(valid_xyxys,valid_masks2)) if judge != 0]
  valid_images_2 = [image for image,judge in tqdm(zip(valid_images_2,valid_masks2_2)) if judge != 0]
  valid_masks_2 = [mask for mask,judge in tqdm(zip(valid_masks_2,valid_masks2_2)) if judge != 0]
  valid_xyxys_2 = [image for image,judge in tqdm(zip(valid_xyxys_2,valid_masks2_2)) if judge != 0]
  valid_images_3 = [image for image,judge in tqdm(zip(valid_images_3,valid_masks2_3)) if judge != 0]
  valid_masks_3 = [mask for mask,judge in tqdm(zip(valid_masks_3,valid_masks2_3)) if judge != 0]
  valid_xyxys_3 = [image for image,judge in tqdm(zip(valid_xyxys_3,valid_masks2_3)) if judge != 0]

  # 3. divide for 7kfold

  if cfg.valid_id==1:
    if cfg.valid_id_2 == 2:
        validjudge = [d<4500 for a,b,c,d in valid_xyxys]
    elif  cfg.valid_id_2 == 3:
        validjudge = [d>=4500 for a,b,c,d in valid_xyxys]
    else:
        validjudge = [d>=0 for a,b,c,d in valid_xyxys]

    train_images.extend([image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == False])

    if cfg.useir:

        train_images.extend([image for image,judge in tqdm(zip(valid_images_ir,validjudge)) if judge == False])
        train_masks.extend([image for image,judge in tqdm(zip(valid_masks_ir,validjudge)) if judge == False])


    valid_images = [image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == True]
    valid_masks = [image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == True]
    valid_xyxys = [image for image,judge in tqdm(zip(valid_xyxys,validjudge)) if judge == True]
    valid_images_2 = [image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == True]
    valid_masks_2 = [image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == True]
    valid_xyxys_2 = [image for image,judge in tqdm(zip(valid_xyxys_2,validjudge)) if judge == True]
    valid_images_3 = [image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == True]
    valid_masks_3 = [image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == True]
    valid_xyxys_3 = [image for image,judge in tqdm(zip(valid_xyxys_3,validjudge)) if judge == True]

  if cfg.valid_id==2:
    if cfg.valid_id_2 == 1:
        validjudge = [d<6141 for a,b,c,d in valid_xyxys]
    elif cfg.valid_id_2 == 2:
        validjudge = [d>=6141 for a,b,c,d in valid_xyxys]
        validjudge2 = [d<10739 for a,b,c,d in valid_xyxys]
        validjudge = [a*b for a,b in zip(validjudge,validjudge2)]
    else:
        validjudge = [d>=10739 for a,b,c,d in valid_xyxys]

    train_images.extend([image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == False])

    if cfg.useir:

        train_images.extend([image for image,judge in tqdm(zip(valid_images_ir,validjudge)) if judge == False])
        train_masks.extend([image for image,judge in tqdm(zip(valid_masks_ir,validjudge)) if judge == False])


    valid_images = [image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == True]
    valid_masks = [image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == True]
    valid_xyxys = [image for image,judge in tqdm(zip(valid_xyxys,validjudge)) if judge == True]
    valid_images_2 = [image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == True]
    valid_masks_2 = [image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == True]
    valid_xyxys_2 = [image for image,judge in tqdm(zip(valid_xyxys_2,validjudge)) if judge == True]
    valid_images_3 = [image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == True]
    valid_masks_3 = [image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == True]
    valid_xyxys_3 = [image for image,judge in tqdm(zip(valid_xyxys_3,validjudge)) if judge == True]

  if cfg.valid_id==3:
    if cfg.valid_id_2 == 2:
        validjudge = [d<4000 for a,b,c,d in valid_xyxys]
    elif  cfg.valid_id_2 == 3:
        validjudge = [d>=4000 for a,b,c,d in valid_xyxys]
    else:
        validjudge = [d>=0 for a,b,c,d in valid_xyxys]

    train_images.extend([image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == False])

    train_images.extend([image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == False])
    train_masks.extend([image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == False])

    if cfg.useir:

        train_images.extend([image for image,judge in tqdm(zip(valid_images_ir,validjudge)) if judge == False])
        train_masks.extend([image for image,judge in tqdm(zip(valid_masks_ir,validjudge)) if judge == False])


    valid_images = [image for image,judge in tqdm(zip(valid_images,validjudge)) if judge == True]
    valid_masks = [image for image,judge in tqdm(zip(valid_masks,validjudge)) if judge == True]
    valid_xyxys = [image for image,judge in tqdm(zip(valid_xyxys,validjudge)) if judge == True]
    valid_images_2 = [image for image,judge in tqdm(zip(valid_images_2,validjudge)) if judge == True]
    valid_masks_2 = [image for image,judge in tqdm(zip(valid_masks_2,validjudge)) if judge == True]
    valid_xyxys_2 = [image for image,judge in tqdm(zip(valid_xyxys_2,validjudge)) if judge == True]
    valid_images_3 = [image for image,judge in tqdm(zip(valid_images_3,validjudge)) if judge == True]
    valid_masks_3 = [image for image,judge in tqdm(zip(valid_masks_3,validjudge)) if judge == True]
    valid_xyxys_3 = [image for image,judge in tqdm(zip(valid_xyxys_3,validjudge)) if judge == True]

  valid_xyxys = np.stack(valid_xyxys)
  valid_xyxys_2 = np.stack(valid_xyxys_2)
  valid_xyxys_3 = np.stack(valid_xyxys_3)

  print(len(train_images),len(valid_images))

  if cfg.fulltrain:
      train_images.extend(valid_images)
      train_masks.extend(valid_masks)

      train_images.extend(valid_images_2)
      train_masks.extend(valid_masks_2)

      train_images.extend(valid_images_3)
      train_masks.extend(valid_masks_3)

  return train_images,train_masks,valid_images,valid_masks,valid_xyxys,valid_images_2,valid_masks_2,valid_xyxys_2,valid_images_3,valid_masks_3,valid_xyxys_3      


def initialization():
        
    random_seed(SEED)

    aug = getaug(cfg)

    train_dataset = PytorchDataSet(train_images,train_masks,aug["train"])
    valid_dataset = PytorchDataSet(valid_images,valid_masks,aug["valid"])

    if cfg.codetype=="A":
          train_dataloader = DataLoader(train_dataset,batch_size=cfg.train_batch,shuffle = True,num_workers=cfg.num_workers)
    else:
          train_dataloader = DataLoader(train_dataset,batch_size=cfg.train_batch,shuffle = True,num_workers=cfg.num_workers,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=cfg.valid_batch,shuffle = False,num_workers=cfg.num_workers)


    if cfg.modeltype == "huggingface":
      model = HugNet(cfg)
    elif cfg.modeltype == "segmentation_models_pytorch":
      model=Net(cfg) # model instance
    model.to(device) # if GPU is using, this must be needed. cpu is also OK in this sentence.


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr) # Algo for optimizing weight

    num_train_steps = int(len(train_dataloader) * cfg.epochs)
    cfg.num_warmup_steps = int(cfg.num_warmup_ratio * num_train_steps)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    if cfg.use_amp:
      scaler = GradScaler(enabled=cfg.use_amp)
      return train_dataloader,valid_dataloader,model,optimizer,scheduler,scaler

    return train_dataloader,valid_dataloader,model,optimizer,scheduler


def cfgupdate(cfg,opt):
  cfg.fulltrain = False

  if opt.fold == 11:
    cfg.valid_id=1
    cfg.valid_id_2 =2
  elif opt.fold == 12:
    cfg.valid_id=1
    cfg.valid_id_2 =3
  elif opt.fold == 4:
    cfg.valid_id=2
    cfg.valid_id_2 =1
  elif opt.fold == 5:
    cfg.valid_id=2
    cfg.valid_id_2 =2
  elif opt.fold == 6:
    cfg.valid_id=2
    cfg.valid_id_2 =3
  elif opt.fold == 13:
    cfg.valid_id=3
    cfg.valid_id_2 =2
  elif opt.fold == 14:
    cfg.valid_id=3
    cfg.valid_id_2 =3
  elif opt.fold == 10:
    cfg.valid_id=2
    cfg.valid_id_2 =1
    cfg.fulltrain = True
  else:
    cfg.valid_id=opt.originalvalidid
    cfg.valid_id_2 = 100
    print("you use original fragment")


  cfg.savepath = opt.savepath
  os.makedirs(cfg.savepath,exist_ok=True)
  cfg.inputpath = opt.inputpath
  cfg.num_workers = opt.num_workers

  # lr tuning
  conddf = pd.read_csv("./kaggle-VCID/condition.csv")
  cond_dict = {}
  


  for num,row in conddf.iterrows():
        
      if row["modelname"] not in cond_dict.keys():
            cond_dict[row["modelname"]] = {}
      if row["fold"] not in cond_dict[row["modelname"]].keys():
            cond_dict[row["modelname"]][row["fold"]] = {}
      
      cond_dict[row["modelname"]][row["fold"]]["lr"] = row["lr"]
      cond_dict[row["modelname"]][row["fold"]]["type"] = row["type"]
      
      cond_dict[row["modelname"]]["trainbatch"] = row["trainbatch"]
      cond_dict[row["modelname"]]["validbatch"] = row["validbatch"]



  cfg.lr = cond_dict[opt.model][fold]["lr"]
  cfg.train_batch = cond_dict[opt.model]["trainbatch"]
  cfg.valid_batch = cond_dict[opt.model]["validbatch"]

  if cond_dict[opt.model][fold]["type"] == "A":
          cfg.uselayers = [28,32,30]
          cfg.codetype="A"
  else:
        cfg.codetype="B"



  if opt.changelr:
        cfg.lr = opt.lr

  if opt.changebatch:
        cfg.train_batch = opt.trainbatch
        cfg.valid_batch = opt.validbatch


  cfg.shutdown = opt.shutdown

  cfg.epochs = opt.epoch

  return cfg

      

if __name__ == "__main__":
      

  # 0. parser setting

  parser = argparse.ArgumentParser()

  parser.add_argument("--model", type=str, default='efficientnetv2_l_in21ft1k',
                          help="model name you want to make")
  parser.add_argument("--fold", type=int, default=11,
                      help="train fold (1 fold only)")
  parser.add_argument("--savepath", type=str, default='.')
  parser.add_argument("--inputpath", type=str, default='.')
  parser.add_argument("--num_workers", type=int, default=8)

  parser.add_argument("--changelr", type=bool, default=False,help="if you want to change lr or not")
  parser.add_argument("--lr", type=float, default=1e-5,help="if you want to change lr, change this value")

  parser.add_argument("--changebatch", type=bool, default=False,help="if you want to change lr or not")
  parser.add_argument("--trainbatch", type=int, default=4,help="if you want to change trainbatch, change this value")
  parser.add_argument("--validbatch", type=int, default=8,help="if you want to change validbatch, change this value")

  parser.add_argument("--originalvalidid", type=int, default=1,help="if you want to change validbatch, change this value")
  parser.add_argument("--epoch", type=int, default=20,help="epoch")


  parser.add_argument("--shutdown", type=bool, default=False,help="shutdown google colab after making the model(only colab)")

  opt = parser.parse_args()
  pprint(opt)

  # 1. change config

  if opt.model == "efficientnetv2_l_in21ft1k":
        from config.exp177 import *
        shutil.copyfile("./kaggle-VCID/config/exp177.py","./kaggle-VCID/cfg.py")
  elif opt.model == "segformer-b3":
        from config.exp143 import *
        shutil.copyfile("./kaggle-VCID/config/exp143.py","./kaggle-VCID/cfg.py")
  elif opt.model == "efficientnet_b7_ns":
        from config.exp93 import *
        shutil.copyfile("./kaggle-VCID/config/exp93.py","./kaggle-VCID/cfg.py")
  elif opt.model == "efficientnet_b6_ns":
        from config.exp129 import *
        shutil.copyfile("./kaggle-VCID/config/exp129.py","./kaggle-VCID/cfg.py")
  elif opt.model == "tf_efficientnet_b8":
        from config.exp208 import *
        shutil.copyfile("./kaggle-VCID/config/exp208.py","./kaggle-VCID/cfg.py")
  else:
        assert("model name is not unseen. please select from segformer-b3,efficientnet_b7_ns,efficientnet_b6_ns,efficientnetv2_l_in21ft1k,tf_efficientnet_b8")


  ## 2. cfg update
  fold = opt.fold
  cfg = cfgupdate(cfg,opt)

  print("-" * 50)
  print(f"model training start : model {opt.model}, fold {fold}, lr {cfg.lr}, train_batch {cfg.train_batch}, valid_batch {cfg.valid_batch}")
  print("-" * 50)

  SEED = 508
  random_seed(SEED)

  # 3 make data

  train_images,train_masks,valid_images,valid_masks,valid_xyxys,valid_images_2,valid_masks_2,valid_xyxys_2,valid_images_3,valid_masks_3,valid_xyxys_3 = makedata()

  # 4. dataloader

  train_dataloader,valid_dataloader,valid_dataloader_2,valid_dataloader_3 = make_dataloader(train_images,valid_images,valid_images_2,valid_images_3,train_masks,valid_masks,valid_masks_2,valid_masks_3,cfg)

  print(len(train_dataloader),len(valid_dataloader))


  # 5. make grand truth
  mask2,valid_mask_gt,valid_mask_gt2,ori_h,ori_w = make_gt(cfg)

  print(mask2.shape)

  print(valid_mask_gt.shape)
  print(valid_mask_gt2.shape)

  # 6. make model and initialization
  train_dataloader,valid_dataloader,model,optimizer,scheduler,scaler = initialization()


  # 7. main

  allres = []
  allvaliddf = []

  bestscore = -1
  aucbest = -1

  best_loss = np.inf
  beststep=0


  print("")
  print(f"################  fold {fold} start ####################")
  print("")
  


  for epoch in range(cfg.epochs):

          allpreds = []

          ## training

          trainloss,model,scheduler,scaler = training(train_dataloader,model,optimizer,scheduler,scaler,cfg)

          ## validating

          validloss, mask_pred = validating(valid_dataloader,model,valid_xyxys,valid_mask_gt,cfg)
          mask_pred = np.where(mask2==0,0,mask_pred)
          mask_pred = np.where(np.isnan(mask_pred),0,mask_pred)
          allpreds.append(mask_pred)

          ## validating 2

          validloss_2, mask_pred = validating(valid_dataloader_2,model,valid_xyxys_2,valid_mask_gt,cfg)
          mask_pred = np.where(mask2==0,0,mask_pred)
          mask_pred = np.where(np.isnan(mask_pred),0,mask_pred)
          allpreds.append(mask_pred)

          ## validating 3

          validloss_3, mask_pred = validating(valid_dataloader_3,model,valid_xyxys_3,valid_mask_gt,cfg)
          mask_pred = np.where(mask2==0,0,mask_pred)
          mask_pred = np.where(np.isnan(mask_pred),0,mask_pred)
          allpreds.append(mask_pred)



          mask_pred = np.mean(np.array(allpreds),axis=0)

          ## align mask pred size (eliminate 0 padding)

          if cfg.valid_id==1:
            if cfg.valid_id_2 == 1:
              mask_pred = mask_pred[:ori_h,:ori_w]

            elif cfg.valid_id_2 == 2:
              mask_pred = mask_pred[:4500,:ori_w]

            else:
              mask_pred = mask_pred[4500:,:ori_w]


          elif cfg.valid_id==2:
            if cfg.valid_id_2 == 1:
              mask_pred = mask_pred[:6141,:ori_w]

            elif cfg.valid_id_2 == 2:
              mask_pred = mask_pred[6141:10739,:ori_w]

            else:
              mask_pred = mask_pred[10739:,:ori_w]


          elif cfg.valid_id==3:
            if cfg.valid_id_2 == 1:
              mask_pred = mask_pred[:ori_h,:ori_w]

            elif cfg.valid_id_2 == 2:
              mask_pred = mask_pred[:4000,:ori_w]

            else:
              mask_pred = mask_pred[4000:,:ori_w]



          mask_pred = get_ranking(mask_pred)
          avg_val_loss = validloss * 1/3 + validloss_2 * 1/3 + validloss_3 * 1/3

          del validloss,validloss_2,validloss_3


          ## metric and th calc
          best_dice, best_th = calc_cv(valid_mask_gt2, mask_pred)
          score = best_dice
          aucscore = get_score(valid_mask_gt2, mask_pred)
          print(f"auc score is {aucscore}")

          # visualize

          fig, axes = plt.subplots(1, 3, figsize=(15, 8))
          axes[0].imshow(valid_mask_gt)
          axes[1].imshow(mask_pred, vmin=0, vmax=1)
          axes[2].imshow((mask_pred>=best_th).astype(int))
          plt.show()

          # earlystop

          update_best = score > bestscore
          update_best_auc = aucscore > aucbest

          if update_best:

              print(f"Best score is {bestscore} → {score}. Saving model")
              bestscore = score

              state = {
                          'state_dict': model.state_dict(),
                          'preds': mask_pred,
                          "bestscore":bestscore
                      }


              torch.save(state, os.path.join(cfg.savepath,f"model{fold}.pth"))

              del state
              torch.cuda.empty_cache()
              gc.collect()

              beststep=0

          else:
              beststep +=1


          print("")
          print(f"fold {fold} : epoch {epoch} train_loss {trainloss} valid_loss {avg_val_loss} valid_score {score} bestscore {bestscore}")
          print("")



          if beststep == cfg.earlystop:
              print("Early stop end process")
              break

          allres.append([fold,epoch,trainloss,avg_val_loss,score,bestscore])

          del mask_pred,trainloss
          torch.cuda.empty_cache()
          gc.collect()


  allresdf = pd.DataFrame(allres)
  allresdf.columns = ["fold","epoch","train_loss","valid_loss","valid_score","bestscore"]
  allresdf.to_csv(f"{cfg.savepath}/allres{fold}.csv",index=False)

  if cfg.sessionend:
    runtime.unassign()
