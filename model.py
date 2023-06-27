from loadlibrary import *

from util import *

class HugNet(nn.Module):
    
    def __init__(self,cfg):
        super(HugNet,self).__init__()
        self.cfg = cfg

        self.model = SegformerForSemanticSegmentation.from_pretrained(cfg.modelname,num_labels=1,ignore_mismatched_sizes=True)
        self.model.save_pretrained(cfg.savepath)


    def forward(self,img,targets=None,mode=None):

        output = self.model(img)
        output = output["logits"]
        output = nn.functional.interpolate(output, size=img.shape[-2:], mode="bilinear", align_corners=False) # 4倍にする


        loss = 0
        metrics = 0

        if targets is not None:
          loss = self.loss(output, targets)
        else:
          return sigmoid(output.detach().cpu().numpy())

        return output, loss, metrics


    #### loss function #####################

    def loss(self, outputs, targets):


        loss_fct = smp.losses.SoftBCEWithLogitsLoss()
        loss = loss_fct(outputs, targets)

        return loss


# %%
class Net(nn.Module):


    def __init__(self,cfg):
        super(Net,self).__init__()
        self.cfg = cfg
        self.encoder = smp.Unet(
            encoder_name=cfg.modelname,
            encoder_weights=cfg.weight,
            in_channels=cfg.in_chans,
            classes=cfg.nclasses,
            activation=None,
        )


    def forward(self,img,targets=None,mode=None):

        output = self.encoder(img)

        loss = 0
        metrics = 0

        if targets is not None:
          loss = self.loss(output, targets)
        else:
          return sigmoid(output.detach().cpu().numpy())

        return output, loss, metrics


    #### loss function #####################

    def loss(self, outputs, targets):


        loss_fct = smp.losses.SoftBCEWithLogitsLoss()
        loss = loss_fct(outputs, targets)

        return loss

    #### monitor metrics ######


def get_scheduler(cfg, optimizer, num_train_steps):
        scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

class HugNetinf(nn.Module):
    

    def __init__(self,cfg):
        super(HugNetinf,self).__init__() 
        self.cfg = cfg

        self.model = SegformerForSemanticSegmentation.from_pretrained(cfg.modelpath)

        
   
    
    def forward(self,img,targets=None,mode=None): 

        output = self.model(img)
        output = output["logits"]
        output = nn.functional.interpolate(output, size=img.shape[-2:], mode="bilinear", align_corners=False) # 4倍にする

        return sigmoid(output.detach().cpu().numpy())

class Netinf(nn.Module):
    

    def __init__(self,cfg):
        super(Netinf,self).__init__() 
        self.encoder = smp.Unet(
            encoder_name=cfg.modelname, 
            encoder_weights=None,
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
        )

    
    def forward(self,img,targets=None,mode=None): 

        output = self.encoder(img)
        return sigmoid(output.detach().cpu().numpy())


def getallmodels(cfg):

  allmodels = []

  for fold in cfg.usemodels:
      
      print(fold)
      
      if cfg.modeltype == "huggingface":
          model = HugNetinf(cfg)
      else:
          model = Netinf(cfg)
      model.to(device)
      
      model_path = f"{cfg.modelpath}/model{fold}.pth"
      state = torch.load(model_path)['state_dict']
      model.load_state_dict(state)
      model.eval()
          
      allmodels.append(model)
      
      del state
      del model
      
      gc.collect()
      torch.cuda.empty_cache()
  return allmodels
