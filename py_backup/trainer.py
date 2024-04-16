import torch.optim as optim
import torch.nn as nn
from collections import deque
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import csv
import torch
from tqdm.cli import tqdm
from models import *

def run_model(model, loader, loss_fn=None, optimizer=None, is_training=False, epoch=None, is_test = False):
    targets = []
    preds = []
    smooth_loss_queue = deque(maxlen=100)

    if is_training:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Valid/Test'

    running_loss = 0.0
    bar = tqdm(loader, ascii=True)

    if not is_test:
        for cnt, (data, target) in enumerate(bar):
            data = data.to('cuda:1')
            target = target.to('cuda:1')
            if is_training:
                optimizer.zero_grad()

            outputs = model(data)

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            if target.dim() == 0:
                target = target.unsqueeze(0)

            if not is_test:
                total_loss = loss_fn(outputs, target.long())
                running_loss += total_loss.item()
                smooth_loss_queue.append(total_loss.item())
                smooth_loss = sum(smooth_loss_queue) / len(smooth_loss_queue)

            predicted = torch.argmax(outputs, dim=-1)

            if not is_test:
                preds.extend(predicted.detach().cpu().tolist())
                targets.extend(target.detach().cpu().tolist())

            else:
                preds.extend(predicted.detach().cpu().tolist())

            if is_training:
                total_loss.backward()
                optimizer.step()

            
            bar.set_description(f'Loss: {total_loss:.6f} | Smooth Loss: {smooth_loss:.6f}')
        f1_score_ = f1_score(np.array(targets), np.array(preds), average='macro')
        acc_score = accuracy_score(np.array(targets), np.array(preds))

        return running_loss / len(loader), acc_score, f1_score_, np.array(targets), np.array(preds)

    else:
        preds = []
        smooth_loss_queue = deque(maxlen=50)

        model.eval()
        mode = 'Valid/Test'

        bar = tqdm(loader, ascii=True)

        for cnt, (data) in enumerate(bar):
            data = data.to('cuda:1')

            outputs = model(data)

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            predicted = torch.argmax(outputs, dim=-1)
            preds.extend(predicted.detach().cpu().tolist())
            
        return preds
    


def train_kfold(model, cfg, train_loaders, valid_loaders):
    for idx, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
        model = get_swin_large()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        best_f1 = 0

        # Initialization list for train visualization
        train_per_loss = []
        valid_per_loss = []

        train_per_acc = []
        valid_per_acc = []

        train_per_f1 = []
        valid_per_f1 = []

        # Print table header

        with open(f"logs/{cfg['attempt_name']}_{idx+1}fold.csv", "w", newline='') as csvfile:
            fieldnames = ['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Valid Loss', 'Valid Acc', 'Valid F1', 'Learning Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()
            for e in range(cfg['epochs']):
                train_loss, train_acc, train_f1, train_targets, train_preds = run_model(model, train_loader, criterion, optimizer, is_training=True, epoch=cfg['epochs'])
                valid_loss, valid_acc, valid_f1, valid_targets, valid_preds = run_model(model, valid_loader, criterion, optimizer, is_training=False, epoch=cfg['epochs'])
                
                train_per_loss.append(train_loss)
                valid_per_loss.append(valid_loss)
                
                train_per_acc.append(train_acc)
                valid_per_acc.append(valid_acc)
                
                train_per_f1.append(train_f1)
                valid_per_f1.append(valid_f1)
                
                # Print epoch results in table format
                print(f'{"-"*75}')
                print_output = f'Epoch: {e} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Train F1: {train_f1:.6f} | Valid Loss: {valid_loss:.6f} | Valid Acc: {valid_acc:.6f} | Valid F1: {valid_f1:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}'
                print(print_output)
                print(f'{"-"*75}')
                writer.writerow({
                    'Epoch': e,
                    'Train Loss': train_loss,
                    'Train Acc': train_acc,
                    'Train F1': train_f1,
                    'Valid Loss': valid_loss,
                    'Valid Acc': valid_acc,
                    'Valid F1': valid_f1,
                    'Learning Rate': optimizer.param_groups[0]['lr']
                })
                
                scheduler.step()

                if valid_f1 < best_f1:
                    print(f'{"*"*75}\nModel saved! Improved from {best_f1:.6f} to {valid_f1:.6f}\n{"*"*75}')
                    best_f1 = valid_loss
                    torch.save(model.state_dict(), f'models/{cfg["attempt_name"]}_{idx+1}fold.pt')
                
def train(model, cfg, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    best_loss = float('inf')

    # Initialization list for train visualization
    train_per_loss = []
    valid_per_loss = []

    train_per_acc = []
    valid_per_acc = []

    train_per_f1 = []
    valid_per_f1 = []

    # Print table header

    with open(f"logs/{cfg['attempt_name']}.csv", "w", newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Valid Loss', 'Valid Acc', 'Valid F1', 'Learning Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()
        for e in range(cfg['epochs']):
            train_loss, train_acc, train_f1, train_targets, train_preds = run_model(model, train_loader, criterion, optimizer, is_training=True, epoch=cfg['epochs'])
            valid_loss, valid_acc, valid_f1, valid_targets, valid_preds = run_model(model, valid_loader, criterion, optimizer, is_training=False, epoch=cfg['epochs'])
            
            train_per_loss.append(train_loss)
            valid_per_loss.append(valid_loss)
            
            train_per_acc.append(train_acc)
            valid_per_acc.append(valid_acc)
            
            train_per_f1.append(train_f1)
            valid_per_f1.append(valid_f1)
            
            # Print epoch results in table format
            print(f'{"-"*75}')
            print_output = f'Epoch: {e} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Train F1: {train_f1:.6f} | Valid Loss: {valid_loss:.6f} | Valid Acc: {valid_acc:.6f} | Valid F1: {valid_f1:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}'
            print(print_output)
            print(f'{"-"*75}')
            writer.writerow({
                'Epoch': e,
                'Train Loss': train_loss,
                'Train Acc': train_acc,
                'Train F1': train_f1,
                'Valid Loss': valid_loss,
                'Valid Acc': valid_acc,
                'Valid F1': valid_f1,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })
            
            scheduler.step()

            if valid_loss < best_loss:
                print(f'{"*"*75}\nModel saved! Improved from {best_loss:.6f} to {valid_loss:.6f}\n{"*"*75}')
                best_loss = valid_loss
                torch.save(model.state_dict(), f'models/{cfg["attempt_name"]}.pt')