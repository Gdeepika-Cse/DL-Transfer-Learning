# NAME: DEEPIKA G
# REG.NO: 212224040060
# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.
VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

## Neural Network Model

<img width="1043" height="802" alt="592412633-c67d64fe-6d90-46b2-84b2-84a9bf1d77e0" src="https://github.com/user-attachments/assets/c23233bd-3a30-45fd-b6ea-a7f1c8a9f79b" />

## DESIGN STEPS
# STEP 1:
Import required libraries and define image transforms.

# STEP 2:
Load training and testing datasets using ImageFolder.

# STEP 3:
Visualize sample images from the dataset.

# STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

# STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

# STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name: DEEPIKA G
### Register Number: 212224040060
```python
# Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
# write your code here
from torchvision.models import VGG19_Weights
model =models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
# Write your code here
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, test_loader,num_epochs=100):
    train_losses = []

    val_losses = []

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion (outputs, labels.unsqueeze(1).float())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        #Compute validation loss

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images, labels = images.to(device), labels.to (device)

                outputs = model(images)

                loss = criterion(outputs, labels.unsqueeze (1).float())

                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))

        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: DEEPIKA G")
    print("Register Number: 212224040060")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="604" height="734" alt="image" src="https://github.com/user-attachments/assets/c19cb4ad-a443-4252-bc8c-e54d47c510d6" />

<img width="507" height="721" alt="image" src="https://github.com/user-attachments/assets/61fef846-b23c-4fa7-9d9b-6c657377f59f" />

<img width="697" height="754" alt="image" src="https://github.com/user-attachments/assets/8f79ded9-fde6-46a0-b344-fbd0397ca1fd" />

## Confusion Matrix

<img width="719" height="577" alt="image" src="https://github.com/user-attachments/assets/4c9d8273-ee3d-41fd-ba6d-7b8241f7ce93" />

## Classification Report

<img width="403" height="201" alt="image" src="https://github.com/user-attachments/assets/fe654f5b-9997-492f-b7d0-d2acb6da3800" />

### New Sample Data Prediction

<img width="579" height="892" alt="image" src="https://github.com/user-attachments/assets/b51351ef-811c-4912-ac5c-81d445511894" />

## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
