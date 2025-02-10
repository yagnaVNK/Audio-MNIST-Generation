import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from src.TransformerMonai import MonaiDecoderOnlyModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from custom_transforms import TrimSilence, FixLength
from src.Dataset import AudioMNIST
import torch.nn.functional as F
import torch.optim as optim


# Training loop
def train_monai(model, x_train_loader, y_train_loader,x_test_loader,y_test_loader, vocab_size=256, num_epochs=5, learning_rate=1e-4, device=''):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loss=[]
    validation_loss = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(zip(x_train_loader, y_train_loader)):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # xb = I1, I2 , I3,.... I352
            # yb = I2, I3, I4,... I353

            output = model(xb, yb[:, :-1])  # Feed input to predict next steps
            
            # Reshape output and target to match the input requirements of CrossEntropyLoss
            B, T, vocab_size = output.shape
            output = output.view(B * T, vocab_size)  # Flatten output to shape (B*T, vocab_size)

            # Add one hot encoding to the output as the target is indices and output is logits.
            # Don't do this - it's unnecessary with CrossEntropyLoss

            target = yb[:, 1:].contiguous().view(-1)  # Flatten target to shape (B*T) 

            # Ensure output and target match in size
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size]
                target = target[:min_size]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(x_train_loader))
        print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(x_train_loader)}')

        model.eval()
        total_loss = 0
        for xb, yb in zip(x_test_loader, y_test_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            output = model(xb, yb[:, :-1])  # Feed input to predict next steps
            
            # Reshape output and target to match the input requirements of CrossEntropyLoss
            B, T, vocab_size = output.shape
            output = output.view(B * T, vocab_size)  # Flatten output to shape (B*T, vocab_size)
            target = yb[:, 1:].contiguous().view(-1)  # Flatten target to shape (B*T)

            # Ensure output and target match in size
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size]
                target = target[:min_size]

            loss = criterion(output, target)
            total_loss += loss.item()
        validation_loss.append(total_loss / len(x_test_loader))
        print(f'Epoch {epoch + 1}, Validation Loss: {total_loss / len(x_test_loader)}')



    plt.figure(figsize=(10, 5)) 
    plt.plot(range(1,num_epochs+1), train_loss, 'bo-', label='Training Loss') 
    plt.plot(range(1,num_epochs+1), validation_loss, 'ro-', label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss over Epochs') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig(f"Monai_Cond_TransformerLossCurves_{num_epochs}.png")
    return model

if __name__ == '__main__':

    
    epochs = 250
    MONAI_TRANSFORMER_MODEL_PATH = f'saved_models/MONAI_Cond_Transformer_epochs_{epochs}.pt'
    device = 'cuda:0'
    VQVAE_PATH = 'saved_models/vqvae_monai.pth'
    classes = range(10)
    batch_size = 32
    num_workers = 4
    data_dir = '../Data'
    eval_folder = 'Evaluation_results'
    vqvae_model = torch.load(VQVAE_PATH).to(device)
    block_size = 352
    vocab_size = 256
    m = MonaiDecoderOnlyModel(d_model=256 , nhead = 8, num_layers = 6, vocab_size = vocab_size, max_len = 352, block_size = 352).to(device)
    
    # if os.path.exists(MONAI_TRANSFORMER_MODEL_PATH):
    #     m = torch.load(MONAI_TRANSFORMER_MODEL_PATH).to(device)

    m = torch.load('saved_models/MONAI_Cond_Transformer_epochs_200.pt').to(device)

    transforms = [TrimSilence(5), FixLength(16000)]

    dataset = AudioMNIST(
        data_dir,
        transform=T.Compose(transforms)
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    ds_train, ds_val, ds_test = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )


    data_label = torch.cat([torch.cat((label.unsqueeze(1).to(device), vqvae_model.model.index_quantize(x.to(device)).flatten(1)), dim=1) for x, label in dl_train],dim=0)

    
    datasetlen = len(data_label)
    print(data_label[0].shape)
    print(data_label.shape)
    x_train = torch.stack([data_label[i,:block_size] for i in range(len(data_label))])
    y_train = torch.stack([data_label[i,1:block_size+1] for i in range(len(data_label))])

    optimizer = torch.optim.Adam(m.parameters(), lr = 1e-3)
    x_train_loader = DataLoader(x_train,batch_size=batch_size)
    y_train_loader = DataLoader(y_train,batch_size=batch_size)

    learning_rate = 1e-4


    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    test_data = torch.cat([torch.cat((label.unsqueeze(1).to(device), vqvae_model.model.index_quantize(x.to(device)).flatten(1)), dim=1) for x, label in dl_test],dim=0)
    x_test = torch.stack([test_data[i,:block_size] for i in range(len(test_data))])
    y_test = torch.stack([test_data[i,1:block_size+1] for i in range(len(test_data))])
    x_test_loader = DataLoader(x_test,batch_size=batch_size)
    y_test_loader = DataLoader(y_test,batch_size=batch_size)

    print(next(iter(x_test_loader)).shape)

    m = train_monai(m,x_train_loader,y_train_loader,x_test_loader,y_test_loader, vocab_size=vocab_size, num_epochs=epochs,learning_rate = learning_rate,device = device)
    torch.save(m,MONAI_TRANSFORMER_MODEL_PATH)
    context = torch.tensor([[3],[42],[62]]).to(device=device)
    print(m.generate(context, max_new_tokens=352))