from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import modele_from_scratch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from statistics import mean

def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        running_correct = 0
        running_total = 0

        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, preds = outputs.max(1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            running_acc = 100 * running_correct / running_total

            t.set_description(f"Epoch {epoch+1}/{epochs}: Loss: {mean(running_loss):.4f}    Acc: {running_acc:.2f}%")

def test(model, loader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


transform = transforms.Compose([
    transforms.Resize((278, 185)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((278, 185)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop((278, 185), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root="./content/sorted_movie_posters_paligema")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = transform

classes = dataset.classes
num_classes = len(classes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = modele_from_scratch.VGG16(num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 3
train(model, optimizer, train_loader, epochs)
test_acc = test(model, val_loader)
print(f'Test accuracy:{test_acc}')
torch.save(model.state_dict(), "genre_model.pth")
print("Saved weights to genre_model.pth")
