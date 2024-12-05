import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import warnings
warnings.filterwarnings('ignore')

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 224,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 2,
    "NUM_CLASSES": 2,
    "CLASS_NAMES": ["Fake", "Real"]
}

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectionModel, self).__init__()
        
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        num_features = self.efficientnet.classifier[1].in_features
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.efficientnet(x)

def load_datasets(train_dir, val_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIGURATION["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIGURATION["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return model

def main():
    train_directory = "deepfake-and-real-images/Dataset/Train"
    val_directory = "deepfake-and-real-images/Dataset/Validation"
    
    train_loader, val_loader = load_datasets(train_directory, val_directory)
    
    model = DeepfakeDetectionModel(num_classes=CONFIGURATION["NUM_CLASSES"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIGURATION["LEARNING_RATE"])
    
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=CONFIGURATION["N_EPOCHS"]
    )
    
    torch.save(trained_model.state_dict(), 'deepfake_detection_model.pth')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
