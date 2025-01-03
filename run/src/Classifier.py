import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy

class LightningAudioClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        # CNN Architecture
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(2, 32, kernel_size=3, padding=1), # (2, 128, 173) -> (32, 128, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (32, 128, 40) -> (64, 64, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (64, 64, 20) -> (128, 32, 10)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Fourth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (128, 32, 10) -> (256, 16, 5)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(256*16*5 , 512), # (1, 256*16*5) -> (512)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) # (512) -> (10)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x=x.float()
        # Pass through CNN
        x = self.features(x) # (2, 128, 352) -> (256, 16, 44)
        x = x.view(x.size(0), -1)  # (256, 16, 44) -> (1, 256*16*44)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.train_accuracy(logits.softmax(dim=-1), y)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy(logits.softmax(dim=-1), y)
        self.log('val_acc', self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.test_accuracy(logits.softmax(dim=-1), y)
        self.log('test_acc', self.test_accuracy, prog_bar=True)