
EXP=177

import os

# ====================================================
# Configurations
# ====================================================
class CFG:


    # ============== 0. basic =============

    
    compname = 'vesuvius'
    savepath = f"./VCID-exp{EXP}"

    sessionend = True
   
    vervose = 250 # noneの場合、1epochに１回

    # ============== 1. Data =============
    
    train_batch = 4                                                                                                                                                                                    
    valid_batch = train_batch * 2
   
    valid_id = 3
    valid_id_2 = 2

    train_fold = [0]
    
    # ============== 2. Model & Training =============

    nclasses = 1

    modeltype = "segmentation_models_pytorch" # huggingface, timm-unet,segmentation_models_pytorch

    modelname = "tu-tf_efficientnetv2_l_in21ft1k" #  segmentation_models_pytorchの場合tuをつける 'tu-tf_efficientnetv2_xl_in21ft1k'

    weight = "imagenet" # urlを見る
    in_chans = 6 # 65
    imsize = 480 # 224 or 224
    
    tile_size = 480
    stride = tile_size // 4

    epochs=20

    scheduler='cosine' # ['linear', 'cosine']
    num_cycles=0.5
    num_warmup_ratio=0.1
    num_warmup_steps=0

    lr = 7.5e-4
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
