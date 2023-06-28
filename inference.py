import argparse
from pprint import pprint

from util import *
from loadlibrary import *
from dataset_loader import *
from metric import *
from model import *
from make_gt import *
from train_valid import *


# preprocess function

def read_image(fragment_id,mid):
    images = []

    amari = cfg.in_chans % 2
    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2 + amari
    idxs = range(start, end)

    for i in tqdm(idxs):
        
        image = cv2.imread(f"{cfg.inputpath}/{fragment_id}/surface_volume/{i:02}.tif", 0)

        if opt.rotation:
              image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    
    ## mask make
    mask2 = cv2.imread(f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
    if opt.rotation:
        mask2 = cv2.rotate(mask2, cv2.ROTATE_90_CLOCKWISE)
    
    mask2 = np.pad(mask2, [(0, pad0), (0, pad1)], constant_values=0)

    mask2 = mask2.astype('float32')
    mask2 /= 255.0 # 正規化 ?
    
    return images,mask2

def make_test_dataset(fragment_id,mid):
    test_images,mask2 = read_image(fragment_id,mid)
    
    
    x1_list = list(range(0, test_images.shape[1]-cfg.tile_size+1, cfg.stride))
    y1_list = list(range(0, test_images.shape[0]-cfg.tile_size+1, cfg.stride))
    
    test_images_list = []
    xyxys = []
    valid_masks2 = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + cfg.tile_size
            x2 = x1 + cfg.tile_size
            
            test_images_list.append(test_images[y1:y2, x1:x2])
            xyxys.append((x1, y1, x2, y2))
            
            valid_masks2.append(np.sum(mask2[y1:y2, x1:x2]))
            
            
    test_images_list = [image for image,judge in tqdm(zip(test_images_list,valid_masks2)) if judge != 0]
    xyxys = [xyxy for xyxy,judge in tqdm(zip(xyxys,valid_masks2)) if judge != 0]
            
    xyxys = np.stack(xyxys)
    
    aug = getaug(cfg)   
            
    test_dataset = testPytorchDataSet(test_images_list, transform=aug["valid"])
    
    test_loader = DataLoader(test_dataset,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    
    return test_loader, xyxys

def cfgupdate(cfg,opt):
      
  cfg.usemodels = opt.usemodels
  cfg.tta1 = True
  cfg.tta2 = False
  cfg.tta3 = False


  cfg.savepath = opt.savepath
  cfg.inputpath = opt.inputpath
  cfg.num_workers = opt.num_workers

  if opt.changebatch:
        cfg.batch_size = opt.validbatch
  else:
        cfg.batch_size = 4


  cfg.sessionend = opt.shutdown
  cfg.target_size = 1
  cfg.modelpath = opt.modelpath

  return cfg

def make_maskpred(fragment_id,test_loader,xyxys):
    
        binary_mask = cv2.imread(f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
    
        if opt.rotation:
          binary_mask = cv2.rotate(binary_mask, cv2.ROTATE_90_CLOCKWISE)
        binary_mask = (binary_mask / 255).astype(int)

        ori_h = binary_mask.shape[0]
        ori_w = binary_mask.shape[1]
        # mask = mask / 255

        pad0 = (cfg.tile_size - binary_mask.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - binary_mask.shape[1] % cfg.tile_size)

        binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask_pred = np.zeros(binary_mask.shape)
        mask_count = np.zeros(binary_mask.shape)

        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):

            allpreds = []

            images = images.to(device)
            batch_size = images.size(0)

            with torch.no_grad():
                               
                y_preds = inference(images,allmodels)
                allpreds.append(y_preds)

            if cfg.tta1:

                images2 =  torch.flip(images,[2])
                with torch.no_grad():
                    y_preds = inference(images2,allmodels)
                    y_preds = y_preds[:,:,::-1,:]

                # make whole mask
                allpreds.append(y_preds)


            y_preds = np.mean(np.array(allpreds),axis=0)

            del images
            
            if cfg.tta1 >0:
                del images2
            
            gc.collect()
            torch.cuda.empty_cache()

            start_idx = step*cfg.batch_size
            end_idx = start_idx + batch_size
            for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
                mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
                mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

      #  plt.imshow(mask_count)
      #  plt.show()

        print(f'mask_count_min: {mask_count.min()}')
        mask_pred /= mask_count

        mask_pred = mask_pred[:ori_h, :ori_w]
        binary_mask = binary_mask[:ori_h, :ori_w]

        
        del mask_count
        return mask_pred      

if __name__ == "__main__":
      

  # 0. parser setting

  parser = argparse.ArgumentParser()

  parser.add_argument("--model", type=str, default='efficientnetv2_l_in21ft1k',
                          help="model name you want to make")
  parser.add_argument("--usemodels", nargs='+',type=int, default=11,help="inference models")
  parser.add_argument("--savepath", type=str, default='.')
  parser.add_argument("--inputpath", type=str, default='.')
  parser.add_argument("--modelpath", type=str, default='.')

  parser.add_argument("--num_workers", type=int, default=8)

  parser.add_argument("--changebatch", type=bool, default=False,help="if you want to change lr or not")
  parser.add_argument("--validbatch", type=int, default=4,help="if you want to change validbatch, change this value")

  parser.add_argument("--th", type=float, default=0.96,help="threshold")
  parser.add_argument("--rotation", type=bool, default=False,help="if you want to rotate image and infer, change this value")


  parser.add_argument("--shutdown", type=bool, default=False,help="shutdown google colab after making the model(only colab)")

  opt = parser.parse_args()
  pprint(opt)

  # 1. change config

  if opt.model == "efficientnetv2_l_in21ft1k":
        from config.exp177 import *
  elif opt.model == "segformer-b3":
        from config.exp143 import *
  elif opt.model == "efficientnet_b7_ns":
        from config.exp93 import *
  elif opt.model == "efficientnet_b6_ns":
        from config.exp129 import *
  elif opt.model == "tf_efficientnet_b8":
        from config.exp208 import *
  else:
        assert("model name is not unseen. please select from segformer-b3,efficientnet_b7_ns,efficientnet_b6_ns,efficientnetv2_l_in21ft1k,tf_efficientnet_b8")

  cfgupdate(cfg,opt)


  print("-" * 50)
  print(f"model inference start : model {opt.model}, batchsize {cfg.batch_size}")
  print("-" * 50)

  SEED = 508
  random_seed(SEED)

  # 2 model load

  allmodels = getallmodels(cfg)

  # 3 main

  fragment_ids = sorted(os.listdir(cfg.inputpath))


  results = []
  for fragment_id in fragment_ids:
        
        print(f"frag {fragment_id} start")
        
        binary_mask = cv2.imread(f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
        
        if opt.rotation:
              binary_mask = cv2.rotate(binary_mask, cv2.ROTATE_90_CLOCKWISE)

        test_loader, xyxys = make_test_dataset(fragment_id,cfg.uselayers[0])        
        mask_pred = make_maskpred(fragment_id,test_loader,xyxys) * 1/3
        del test_loader
        gc.collect()
        torch.cuda.empty_cache()
                
        test_loader2, xyxys = make_test_dataset(fragment_id,cfg.uselayers[1])
        mask_pred += make_maskpred(fragment_id,test_loader2,xyxys) * 1/3
        del test_loader2
        gc.collect()
        torch.cuda.empty_cache()
        
        test_loader3, xyxys = make_test_dataset(fragment_id,cfg.uselayers[2])
        mask_pred += make_maskpred(fragment_id,test_loader3,xyxys) * 1/3
        del test_loader3
        gc.collect()
        torch.cuda.empty_cache()
        
        
        mask_pred = np.where(binary_mask==0,0,mask_pred)
        mask_pred = np.where(np.isnan(mask_pred),0,mask_pred)

        if opt.rotation:
              mask_pred = cv2.rotate(mask_pred, cv2.ROTATE_90_COUNTERCLOCKWISE) # もとに戻して保存
        np.save(f"{cfg.savepath}/maskpred_{fragment_id}",mask_pred)

        mask_pred = get_ranking(mask_pred)
        mask_pred = mask_pred >= opt.th
        np.save(f"{cfg.savepath}/maskpred_bin_{fragment_id}",mask_pred)

        inklabels_rle = rle(mask_pred)
        results.append((fragment_id, inklabels_rle))
        

        del mask_pred,binary_mask
        gc.collect()

  sub = pd.DataFrame(results, columns=['Id', 'Predicted'])
  sub.to_csv(f"{cfg.savepath}/rle_result.csv", index=False)


  if cfg.sessionend:
    runtime.unassign()
