from loadlibrary import *


SEED = 508

def random_seed(SEED):

      random.seed(SEED)
      os.environ['PYTHONHASHSEED'] = str(SEED)
      np.random.seed(SEED)
      torch.manual_seed(SEED)
      torch.cuda.manual_seed(SEED)
      torch.cuda.manual_seed_all(SEED)
      torch.backends.cudnn.deterministic = True





def read_image_mask(fragment_id,mid,cfg,irmode=False):
    
    images = []

    amari = cfg.in_chans % 2


  #  mid = 28
    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2 + amari
    #idxs = range(65)
    idxs = range(start, end)


    for i in tqdm(idxs):

        if irmode:
          image = cv2.imread(f"{cfg.inputpath}/{fragment_id}/ir.png", 0)
        else:
          image = cv2.imread(f"{cfg.inputpath}/{fragment_id}/surface_volume/{i:02}.tif", 0)

        #image = cv2.medianBlur(image, ksize=5)

        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)


    images = np.stack(images, axis=2)

    ## labelmake

    mask = cv2.imread(f"{cfg.inputpath}/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0 # signalを1に

    ## mask make : 領域をmaskのみにするため
    mask2 = cv2.imread(f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
    mask2 = np.pad(mask2, [(0, pad0), (0, pad1)], constant_values=0)

    mask2 = mask2.astype('float32')
    mask2 /= 255.0 # signalを1に



    return images, mask, mask2

def get_train_valid_dataset(mid,cfg,irmode=False):
    train_images = []
    train_masks = []
    train_masks2 = []


    valid_images = []
    valid_masks = []
    valid_xyxys = []

    valid_masks2 = []


    for fragment_id in range(1, 4):

        if irmode:
          image, mask,mask2 = read_image_mask(fragment_id,mid,cfg,irmode=True)
        else:
          image, mask,mask2 = read_image_mask(fragment_id,mid,cfg)

        x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.stride))
        y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size
                # xyxys.append((x1, y1, x2, y2))

                if fragment_id == cfg.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_masks2.append(np.sum(mask2[y1:y2, x1:x2])) # maskの合計が0のところはtrain/validしないようにするため


                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
                    train_masks2.append(np.sum(mask2[y1:y2, x1:x2])) # maskの合計が0のところはtrain/validしないようにするため

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys,train_masks2,valid_masks2

# all
def makedata_main(cfg):

    random_seed(SEED)

    train_images, train_masks, valid_images, valid_masks, valid_xyxys,train_masks2,valid_masks2 = get_train_valid_dataset(cfg.uselayers[0],cfg)
    train_images_2, train_masks_2, valid_images_2, valid_masks_2, valid_xyxys_2,train_masks2_2,valid_masks2_2 = get_train_valid_dataset(cfg.uselayers[1],cfg)
    train_images_3, train_masks_3, valid_images_3, valid_masks_3, valid_xyxys_3,train_masks2_3,valid_masks2_3 = get_train_valid_dataset(cfg.uselayers[2],cfg)

    if cfg.useir:
        train_images_ir, train_masks_ir, valid_images_ir, valid_masks_ir, valid_xyxys_ir,train_masks2_ir,valid_masks2_ir = get_train_valid_dataset(28,cfg,irmode=True)
        train_images.extend(train_images_2)


    ## 1. train_mask2,3 extension

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


    # 2. use only where mask is 1
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


    # 3. devide validation frag to train/valid

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

        # 3.

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

    if cfg.fulltrain:
        train_images.extend(valid_images)
        train_masks.extend(valid_masks)

        train_images.extend(valid_images_2)
        train_masks.extend(valid_masks_2)

        train_images.extend(valid_images_3)
        train_masks.extend(valid_masks_3)

    return train_images,valid_images,valid_images_2,valid_images_3,valid_xyxys,valid_xyxys_2,valid_xyxys_3,train_masks,valid_masks,valid_masks_2,valid_masks_3