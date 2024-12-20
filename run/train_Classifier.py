import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

from src.Dataset import AudioDataset
from src.Classifier import LightningAudioClassifier

def save_model(model, save_path):
    """Save the model state dict and hyperparameters."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model state dict
    torch.save({
        'state_dict': model.state_dict(),
        'hparams': model.hparams,
    }, save_path)
    print(f"Model saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Create and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cm, 
        index=[f'True {i}' for i in range(10)],
        columns=[f'Pred {i}' for i in range(10)]
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main(args):
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize dataset
    full_dataset = AudioDataset(
        root_dir=args.data_dir,
        target_length=args.target_length
    )
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Initialize model
    model = LightningAudioClassifier(learning_rate=args.learning_rate)

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # Setup logger
    logger = TensorBoardLogger(
        os.path.join(results_dir, 'lightning_logs'), 
        name='audio_classifier'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[early_stopping],
        logger=logger,
        gradient_clip_val=args.grad_clip
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Save the final model
    model_save_path = os.path.join(results_dir, 'final_model.pt')
    save_model(model, model_save_path)

    # Test model
    trainer.test(dataloaders=test_loader)
    
    # Get device
    device = next(model.parameters()).device
    
    # Evaluate and create confusion matrix
    predictions, true_labels = evaluate_model(model, test_loader, device)
    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels, predictions, confusion_matrix_path)
    
    # Save test accuracy
    test_accuracy = (predictions == true_labels).mean()
    results = {
        'test_accuracy': test_accuracy,
        'model_path': model_save_path,
        'confusion_matrix_path': confusion_matrix_path
    }
    
    # Save results
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    
    print(f"\nResults saved in {results_dir}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved at: {model_save_path}")
    print(f"Confusion matrix saved at: {confusion_matrix_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Program arguments
    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Directory containing the dataset')
    parser.add_argument('--target_length', type=int, default=47998,
                        help='Target length for audio waveforms')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
    main(args)