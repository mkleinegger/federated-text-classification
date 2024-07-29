import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    device: torch.device = DEVICE,
    verbose=False,
):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    verbose=False,
):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total, correct = 0, 0
    total_loss = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_loss /= len(test_loader)
    accuracy = correct / total

    if verbose:
        print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    return total_loss, accuracy


def run_centralised(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float = 0.9,
    device: torch.device = DEVICE,
    verbose=False,
):
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_model(model, train_loader, epochs, optim, device)
    loss, accuracy = evaluate_model(model, test_loader, device)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    return accuracy, loss
