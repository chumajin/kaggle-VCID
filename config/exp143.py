
EXP=143

import os

# ====================================================
# Configurations
# ====================================================
class CFG:


    # ============== 0. basic =============

    
    compname = 'vesuvius'
    savepath = f"./VCID-exp{EXP}"
    seed = 42

    sessionend = True
    upload = True

    traintype = "binary" # binary / regression / classification
    makebinary =False # binaryを1にする必要がある時
    
    vervose = 250 # noneの場合、1epochに１回

    debug = False
    trainmode=True
  

    # ============== 1. Data =============
    
    
    train_batch = 4                                                                                                                                                                                  
    valid_batch = train_batch * 2
   
    valid_id = 3
    valid_id_2 = 3

    train_fold = [0]

    
    # ============== 2. Model & Training =============

    nclasses = 1

    modeltype = "huggingface" # huggingface, timm-unet,segmentation_models_pytorch

    modelname = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024" #  segmentation_models_pytorchの場合tuをつける 'tu-tf_efficientnetv2_xl_in21ft1k'

    weight = "imagenet" # urlを見る
    in_chans = 3 # 65
    imsize = 1024 # 224 or 224
    
    tile_size = 1024
    stride = tile_size // 4

    epochs=20

    scheduler='cosine' # ['linear', 'cosine']
    num_cycles=0.5
    num_warmup_ratio=0.1
    num_warmup_steps=0

    lr = 2e-4
    max_grad_norm = 10

    earlystop = 4

    use_amp = True


    metric_direction = 'maximize'

    validst = 0
    tta = True
    useir = True

    if modeltype == "huggingface":
     uselayers = [26,29,32]
    else:
     uselayers = [28,30,32]


os.makedirs(CFG.savepath,exist_ok=True)
cfg=CFG()
