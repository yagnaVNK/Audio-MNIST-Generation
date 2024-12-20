import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from src.Transformer import TransformerModel
import matplotlib.pyplot as plt
from src.utils import *
from tqdm import tqdm
import os
from src.Dataset import AudioDataset

def train_model(m, x_train_loader, y_train_loader, x_test_loader, y_test_loader, epochs=10):
    train_loss = []
    validation_loss = []
    for iter in range(epochs):
        print("Epoch 1")
        m.train()
        total_loss = 0
        for xb, yb in tqdm(zip(x_train_loader, y_train_loader)):
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(x_train_loader))
        print(f'Epoch {iter + 1}, Train Loss: {total_loss / len(x_train_loader)}')
        
        m.eval()
        total_loss = 0
        for xb, yb in tqdm(zip(x_test_loader, y_test_loader)):
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            total_loss += loss.item()
        validation_loss.append(total_loss / len(x_test_loader))
        print(f'Epoch {iter + 1}, Validation Loss: {total_loss / len(x_test_loader)}')
    
    plt.figure(figsize=(10, 5)) 
    plt.plot(range(1,epochs+1), train_loss, 'bo-', label='Training Loss') 
    plt.plot(range(1,epochs+1), validation_loss, 'ro-', label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss over Epochs') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig("TransformerLossCurves.png")
    return m

def prepare_transformer_data(vqvae_model, dataloader, block_size=737):
    """
    Prepare data for transformer by encoding spectrograms and flattening indices
    
    Args:
        vqvae_model (VQVAE): Trained VQVAE model
        dataloader (DataLoader): DataLoader containing spectrograms and labels
        block_size (int): Total sequence length including label
    
    Returns:
        tuple: Prepared x and y data for transformer training
    """
    # Prepare data
    data_label = []
    for x, label in dataloader:
        # Encode spectrogram to indices
        x = x.to(device).unsqueeze(1)
        #print(x.shape)
        # Get indices from encoder
        ze = vqvae_model.Encoder(x)
        
        # Quantize indices 
        indices = vqvae_model.codebook.quantize_indices(ze)
        
        # Flatten indices
        flattened_indices = indices.flatten(start_dim=1)
        
        # Combine label with flattened indices
        combined = torch.cat([label.unsqueeze(1).to(device), flattened_indices], dim=1)
        data_label.append(combined)
        #print(combined.shape)
    # Concatenate all data
    data_label = torch.cat(data_label, dim=0)
    
    # Prepare sequences
    x = torch.stack([data_label[i,:block_size-1] for i in tqdm(range(len(data_label)))])
    y = torch.stack([data_label[i,1:block_size+1] for i in tqdm(range(len(data_label)))])
    
    return x, y

if __name__ == '__main__':
    # Update block_size to accommodate label + flattened indices
    block_size = 737  # 1 label + 736 flattened indices
    n_embd = 512  # Adjust embedding dimension if needed
    vocab_size = 512  # Number of unique indices in codebook

    # Load VQVAE model
    vqvae_model = torch.load(
        os.path.join("results", 'vqvae_run_20241216_143233', 'final_model.pt'), 
        weights_only=False
    ).to(device)
    
    # Initialize Transformer Model 
    m = TransformerModel().to(device)
    
    # Load existing transformer model if available
    try:
        m = torch.load(TRANSFORMER_MODEL_PATH, weights_only=False).to(device)
        print("Loaded existing transformer model")
    except:
        print("Creating new transformer model")

    # Load dataset
    dataset = AudioDataset()
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # Prepare training and test data
    x_train, y_train = prepare_transformer_data(vqvae_model, train_loader)
    x_test, y_test = prepare_transformer_data(vqvae_model, test_loader)
    
    # Prepare data loaders
    x_train_loader = DataLoader(x_train, batch_size=8)
    y_train_loader = DataLoader(y_train, batch_size=8)
    x_test_loader = DataLoader(x_test, batch_size=8)
    y_test_loader = DataLoader(y_test, batch_size=8)
    
    print("started training")
    # Training parameters
    epochs = 50
    learning_rate = 1e-4
    
    # Optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    # Train the model
    m = train_model(m, x_train_loader, y_train_loader, x_test_loader, y_test_loader, epochs)
    
    # Save the trained model
    torch.save(m, os.path.join("results", 'vqvae_run_20241216_143233', 'Transformer.pt'))
    
    # Example generation
    context = torch.tensor([[3],[42],[62]]).to(device=device)
    print(m.generate(context, max_new_tokens=736))