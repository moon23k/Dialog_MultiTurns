import time
import math
import torch
import torch.nn as nn



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def create_src_mask(src, pad_idx=1):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask.to(src.device)
    return src_mask



def create_trg_mask(trg, pad_idx=1):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg.size(-1), trg.size(-1)))).bool()
    trg_mask = trg_pad_mask & trg_sub_mask.to(trg.device)
    
    return trg_mask.to(trg.device)




def train_epoch(model, dataloader, criterion, optimizer, clip, device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(device), batch[1].to(device)

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:].contiguous().view(-1)

        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg_input)

        pred = model(src, trg_input, src_mask, trg_mask)
        
        pred_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, pred_dim)

        loss = criterion(pred, trg_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()


        epoch_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'  Batch {i + 1} / {len(dataloader)}  Eval_Loss: {loss}')

    return epoch_loss / len(dataloader)




def valid_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].to(device), batch[1].to(device)

            trg_input = trg[:, :-1]
            trg_y = trg[:, 1:].contiguous().view(-1)
            
            src_mask = create_src_mask(src)
            trg_mask = create_trg_mask(trg_input)

            pred = model(src, trg_input, src_mask, trg_mask)
            
            pred_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, pred_dim)

            loss = criterion(pred, trg_y)

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'  Batch {i + 1} / {len(dataloader)}  Train_Loss: {loss}')

    return epoch_loss / len(dataloader)