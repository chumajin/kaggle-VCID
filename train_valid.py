from loadlibrary import *

def training(
    train_dataloader,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg):

  total_loss = 0 # Initializing total loss

  model.train()

  allpreds = []
  alltargets = []

## 表示用
  steps_per_epoch = int(cfg.train_batch / len(train_dataloader)) + 1
  t=time.time()

  for step,a in enumerate(tqdm(train_dataloader)):

        train_x= a["img"].to(device)
        train_y = a["label"].to(device)

        with autocast(cfg.use_amp):
          output, loss, metrics = model(train_x,train_y) # prediction

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()

        total_loss += loss.item() # integration of loss


        ##### training中の表示 ######

        if (step+1) % cfg.vervose ==0:

#          score = metrics["score"]

#          print (f"Step [{step+1}/{steps_per_epoch}] Loss: {loss.item():.3f} Time: {time.time()-t:.1f} Metric: {score:.3f} lr : {scheduler.get_lr()[0]:.3f},gradnorm : {grad_norm:.3f}")
          print (f"Step [{step+1}/{steps_per_epoch}] Loss: {loss.item():.3f} Time: {time.time()-t:.1f} lr : {scheduler.get_lr()[0]},gradnorm : {grad_norm:.3f}")

          torch.cuda.empty_cache()
          gc.collect()

  total_loss = total_loss/len(train_dataloader)

  return total_loss,model,scheduler,scaler


# %% [markdown]
# # 12.validation func

# %%
def validating(
    valid_dataloader,
    model,
    valid_xyxys,
    valid_mask_gt,
    cfg
    ):


  mask_pred = np.zeros(valid_mask_gt.shape)
  mask_count = np.zeros(valid_mask_gt.shape)

  


  total_loss = 0 # Initializing total loss
  model.eval()


  finpreds = []

  for step,a in enumerate(tqdm(valid_dataloader)):

    train_x= a["img"].to(device)
    train_y = a["label"].to(device)
    batch_size = train_y.size(0)

    allpreds = []


    with torch.no_grad():
          output,loss,metric = model(train_x,train_y) # prediction
    output = torch.sigmoid(output).to("cpu").numpy()
    total_loss += loss.item() # integration of loss
    allpreds.append(output)

    if cfg.tta:

        ## tta1
        train_x2 =  torch.flip(train_x,[2])
        with torch.no_grad():
            output2 = model(train_x2)
            output2 = output2[:,:,::-1,:]

        # make whole mask
        allpreds.append(output2)

    if cfg.tta:
      allpreds = np.mean(np.array(allpreds),axis=0)
    else:
      allpreds = output

    start_idx = step*cfg.valid_batch
    end_idx = start_idx + batch_size
    for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
          
        try:
           
          mask_pred[y1:y2, x1:x2] += allpreds[i].squeeze(0)
          mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))
        except:
          print(x1,x2,y1,y2)

          print(allpreds[i].squeeze(0).shape)



  total_loss = total_loss/len(valid_dataloader)

  print(f'mask_count_min: {mask_count.min()}')
  mask_pred /= mask_count
  return total_loss, mask_pred

def inference(images,allmodels):
    preds = np.mean([model(images) for model in allmodels],axis=0)
    return preds      

