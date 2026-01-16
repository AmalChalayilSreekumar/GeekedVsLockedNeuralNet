import os
import time
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn


class CsvImageDataset(Dataset):
    def __init__(self, csvPath, imagesDir, transform=None):
        self.dataFrame = pd.read_csv(csvPath, header=None, names=["filename", "label"])
        self.imagesDir = imagesDir
        self.transform = transform

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):
        row = self.dataFrame.iloc[index]

        imagePath = os.path.join(self.imagesDir, row["filename"])
        label = int(row["label"])

        image = Image.open(imagePath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def calculateAccuracy(predictedLogits, trueLabels):
    predictedClasses = predictedLogits.argmax(dim=1)
    return (predictedClasses == trueLabels).float().mean().item()


if __name__ == '__main__':
    # Data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CsvImageDataset(
        csvPath="DataSet.csv",
        imagesDir="Images",
        transform=transform
    )

    print(f"Total images in dataset: {len(dataset)}")

    # Split train/val (80/20)
    trainSize = int(0.8 * len(dataset))
    valSize = len(dataset) - trainSize
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    print(f"Training images: {trainSize}")
    print(f"Validation images: {valSize}")

    trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True, num_workers=2)
    valLoader = DataLoader(valDataset, batch_size=16, shuffle=False, num_workers=2)

    # Model setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: locked vs geeked

    device = torch.device("cpu")
    model = model.to(device)

    lossFunction = nn.CrossEntropyLoss() #Cross-entropy loss good for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #Adam(Adaptive Moment Estimation) good for optimizing data with a lot of noise

    # Training loop
    bestValAccuracy = 0.0
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    for epochIndex in range(5):
        epochStartTime = time.time()
        model.train()

        totalTrainLoss = 0.0
        totalTrainAccuracy = 0.0

        for imageBatch, labelBatch in trainLoader:
            imageBatch = imageBatch.to(device)
            labelBatch = labelBatch.to(device)

            optimizer.zero_grad()
            outputLogits = model(imageBatch)
            loss = lossFunction(outputLogits, labelBatch)

            loss.backward() #Back propogation
            optimizer.step()

            totalTrainLoss += loss.item()
            totalTrainAccuracy += calculateAccuracy(outputLogits, labelBatch)

        # Validation
        model.eval()

        totalValLoss = 0.0
        totalValAccuracy = 0.0

        with torch.no_grad():
            for imageBatch, labelBatch in valLoader:
                imageBatch = imageBatch.to(device)
                labelBatch = labelBatch.to(device)

                outputLogits = model(imageBatch)
                loss = lossFunction(outputLogits, labelBatch)

                totalValLoss += loss.item()
                totalValAccuracy += calculateAccuracy(outputLogits, labelBatch)

        avgTrainLoss = totalTrainLoss / len(trainLoader)
        avgTrainAccuracy = totalTrainAccuracy / len(trainLoader)
        avgValLoss = totalValLoss / len(valLoader)
        avgValAccuracy = totalValAccuracy / len(valLoader)
        epochTime = time.time() - epochStartTime

        print(
            f"Epoch {epochIndex + 1:2d}/{15}: "
            f"trainLoss={avgTrainLoss:.4f}, trainAcc={avgTrainAccuracy:.4f} | "
            f"valLoss={avgValLoss:.4f}, valAcc={avgValAccuracy:.4f} "
            f"[{epochTime:.1f}s]"
        )

        # Save best model
        if avgValAccuracy > bestValAccuracy:
            bestValAccuracy = avgValAccuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  New best model saved! (valAcc: {bestValAccuracy:.4f})")

    # Save final model
    torch.save(model.state_dict(), 'model.pth')
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best pass accuracy: {bestValAccuracy:.4f}")
    
    if os.path.exists('model.pth'):
        modelSize = os.path.getsize('model.pth') / (1024*1024)
    
    if os.path.exists('best_model.pth'):
        bestModelSize = os.path.getsize('best_model.pth') / (1024*1024)
