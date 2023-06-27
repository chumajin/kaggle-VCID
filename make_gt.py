from loadlibrary import *

def make_gt(cfg):
  

    fragment_id = cfg.valid_id

    valid_mask_gt = cv2.imread( f"{cfg.inputpath}/{fragment_id}/inklabels.png", 0)

    ori_h = valid_mask_gt.shape[0]
    ori_w = valid_mask_gt.shape[1]


    valid_mask_gt = valid_mask_gt / 255
    pad0 = (cfg.tile_size - valid_mask_gt.shape[0] % cfg.tile_size)
    pad1 = (cfg.tile_size - valid_mask_gt.shape[1] % cfg.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    mask2 = cv2.imread( f"{cfg.inputpath}/{fragment_id}/mask.png", 0)
    mask2 = mask2 / 255
    mask2 = np.pad(mask2, [(0, pad0), (0, pad1)], constant_values=0)

    if cfg.valid_id==1:

        if cfg.valid_id_2 == 2:
            valid_mask_gt2 = valid_mask_gt[:4500,:ori_w]
        elif cfg.valid_id_2 == 3:
            valid_mask_gt2 = valid_mask_gt[4500:,:ori_w]
        else:
            valid_mask_gt2 = valid_mask_gt[:ori_h,:ori_w]


    elif cfg.valid_id==2:

        if cfg.valid_id_2 == 1:
            valid_mask_gt2 = valid_mask_gt[:6141,:ori_w]
        elif cfg.valid_id_2 == 2:
            valid_mask_gt2 = valid_mask_gt[6141:10739,:ori_w]
        else:
            valid_mask_gt2 = valid_mask_gt[10739:,:ori_w]

    elif cfg.valid_id==3:

        if cfg.valid_id_2 == 2:
            valid_mask_gt2 = valid_mask_gt[:4000,:ori_w]
        elif cfg.valid_id_2 == 3:
            valid_mask_gt2 = valid_mask_gt[4000:,:ori_w]
        else:
            valid_mask_gt2 = valid_mask_gt[:ori_h,:ori_w]

    else:
        valid_mask_gt2 = valid_mask_gt[:ori_h,:ori_w]


    return mask2,valid_mask_gt,valid_mask_gt2,ori_h,ori_w
