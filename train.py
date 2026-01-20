import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms, models
from torch.utils.data import DataLoader

SEED = 43
IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def make_transform_fn(transform):
    def transform_fn(examples):
        images = [transform(img.convert('RGB')) for img in examples['image']]
        labels = examples['label']
        return {'pixel_values': images, 'labels': labels}
    return transform_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'torch device: {device}')

def train_epoch(model, loader, criterion, optimizer, device):
    
    """Train the model for one epoch."""
    model.train() 
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

         # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predicts = outputs.argmax(1)
        correct += (predicts == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def fit(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    
    """Train the model for multiple epochs."""
    print("=" * 60)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    print("=" * 60)
    print("Training complete!")
    print()
    return model

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicts = outputs.argmax(1)
            correct += (predicts == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def build_model(model_name: str, num_classes: int) -> nn.Module:
    
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def export_model_to_onnx(model, onnx_path):
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        external_data=False
    )

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":

    print("Loading dataset 'tanganke/gtsrb'...")
    dataset = load_dataset('tanganke/gtsrb')

    data_train = dataset['train']

    num_rows = data_train.num_rows
    num_classes = data_train.features['label'].num_classes

    print(f'num_rows: {num_rows}')
    print(f'num_classes: {num_classes}')

    print("Split data")
    split = data_train.train_test_split(test_size=0.2, stratify_by_column='label', seed=SEED)

    train_tds = split['train'].with_transform(make_transform_fn(train_transform))
    test_tds = split['test'].with_transform(make_transform_fn(test_transform))

    train_loader = DataLoader(train_tds, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_tds, batch_size=64, shuffle=False, num_workers=4)
    
    model = build_model('resnet18', num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Train for 5 epochs
    num_epochs = 5

    model = fit(model, train_loader, test_loader, criterion, optimizer, device, epochs=num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("\nModel saved to 'model_weights.pth'")

    # Export model to ONNX
    export_model_to_onnx(model, "model_gtsr.onnx")



