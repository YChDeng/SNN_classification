# src/training.py

import torch
import numpy as np
import time

def training(model, train_loader, val_loader, optimizer, criterion, device, epochs, tolerance=3, min_delta=0.01, snn_mode=False, num_steps=None, path_load_model="model.pt"):
    time_delta = 0
    total_val_history = []
    accuracy_history = []
    best_accuracy = 0
    count = 0
    early_stop_val = np.inf
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, valid_loss = [], []
        total_train, correct_train = 0, 0
        total_val, correct_val = 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            if snn_mode:
                data = data.view(data.shape[0], -1)
                spk_rec, _ = model(data)
                loss_val = sum(criterion(mem, targets) for mem in spk_rec) / num_steps
                _, predicted = spk_rec.sum(dim=0).max(1)
            else:
                output = model(data)
                loss_val = criterion(output, targets)
                predicted = output.argmax(dim=1)

            loss_val.backward()
            optimizer.step()
            train_loss.append(loss_val.item())
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                if snn_mode:
                    data = data.view(data.shape[0], -1)
                    spk_rec, _ = model(data)
                    loss_val = sum(criterion(mem, targets) for mem in spk_rec) / num_steps
                    _, predicted = spk_rec.sum(dim=0).max(1)
                else:
                    output = model(data)
                    loss_val = criterion(output, targets)
                    predicted = output.argmax(dim=1)
                valid_loss.append(loss_val.item())
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        total_val_history.append(np.mean(valid_loss))
        accuracy_history.append(correct_val / total_val)

        if best_accuracy <= accuracy_history[-1]:
            torch.save(model.state_dict(), path_load_model)
            best_accuracy = accuracy_history[-1]

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {np.mean(train_loss):.4f}, Validation Loss: {np.mean(valid_loss):.4f}, Validation Accuracy: {accuracy_history[-1]*100:.2f}%")

        if abs(early_stop_val - np.mean(valid_loss)) < min_delta:
            count += 1
            if count >= tolerance:
                print("Early stopping")
                break
        else:
            count = 0
        early_stop_val = np.mean(valid_loss)

    time_delta = (time.time() - start) / (epoch + 1)
    return total_val_history, accuracy_history, time_delta

def testing(model, test_loader, device, snn_mode=False, num_steps=None):
    total, correct = 0, 0
    predictions, true_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            if snn_mode:
                data = data.view(data.shape[0], -1)
                spk_rec, _ = model(data)
                _, predicted = spk_rec.sum(dim=0).max(1)
            else:
                output = model(data)
                predicted = output.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return true_labels, predictions