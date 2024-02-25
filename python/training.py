import torch
from torch import nn
from tqdm.auto import tqdm
from params import n_epochs, batch_size
from data_processing import display_encoded_image, save_weights

def train(model,optimizer,dataloader):
    print("Training loop running...\n")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    lmbda = lambda epoch: 0.93
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
    for epoch in range(n_epochs):
        for step, (batch, lengths) in enumerate(tqdm(dataloader)):
            batch = batch.squeeze(2).transpose(0, 1)

            optimizer.zero_grad()

            output, loss, l_kl, l_s, l_p = model(batch, lengths, anneal_loss=True, step=epoch * len(dataloader) + step)

            loss.backward()

            grad_threshold = 1.0 # tunable parameter, prevents exploding gradient
            nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_threshold)
            nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_threshold)

            # update encoder and decoder parameters using adam algorithm
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {loss.item()}")
                print(f"l_kl: {l_kl.item():.4f} l_s: {l_s.item():.4f} l_p: {l_p.item():.4f}") 
                print("---------------------------------------------------------\n")
                

            if step % len(dataloader)/(10*batch_size) == 0:
                  save_weights("model/temp/",max_count=3)
                  torch.cuda.empty_cache()

            if step % 50 == 0:
                # draw image
                display_encoded_image(output[:, 0, :])
                display_encoded_image(batch[:, 0, :])
        scheduler.step()
            
            
                
def run_tests(dataloader,model,count = 2**31 - 1):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    for step, (batch, lengths) in enumerate(tqdm(dataloader)):
            
            batch = batch.squeeze(2).transpose(0, 1)
            output, loss, l_kl, l_s, l_p = model(batch, lengths, anneal_loss=False)
            
            print(f"Step: {step + 1}, Loss: {loss.item()}")
            print(f"l_kl: {l_kl.item():.4f} l_s: {l_s.item():.4f} l_p: {l_p.item():.4f}") 
            print("---------------------------------------------------------\n")
            # draw image
            display_encoded_image(output[:, 0, :])
            display_encoded_image(batch[:, 0, :])
            
            if (step > count):
                return

    
    
