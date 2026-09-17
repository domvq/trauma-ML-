
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch.optim as optim

# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "wound_dataset/train"
VAL_DIR = "wound_dataset/valid"

BATCH_SIZE = 16
EPOCHS = 8
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("AI EMS INJURY CLASSIFIER")
print("=" * 50)
print(f"Device: {DEVICE}")

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

# ============================================================
# MODEL
# ============================================================

model = timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=len(train_dataset.classes)
)

model = model.to(DEVICE)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)

# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    training_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = correct / total

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {training_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Acc: {val_accuracy:.2%}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": train_dataset.classes
    },
    "injury_model.pth"
)

print()
print("=" * 50)
print("MODEL TRAINING COMPLETE")
print("=" * 50)
print("Saved: injury_model.pth")
print(f"Classes: {train_dataset.classes}")

