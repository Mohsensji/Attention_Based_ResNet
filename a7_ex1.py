
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
random.seed(1234)
from torch.utils.data import DataLoader,Subset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import SubsetAugmentation
from dataset import ImagesDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# stacking function
def stacking(batch_as_list: list):
    images, class_ids, class_names, image_filepaths = zip(*batch_as_list)
    stacked_images_tensor = torch.stack(images)
    class_ids_conversion= [torch.tensor(i,dtype=torch.int64).reshape(-1) for i in class_ids]
    stacked_class_ids_tensor = torch.stack(class_ids_conversion)    
    class_names_list = list(class_names)
    image_filepaths_list = list(image_filepaths)
    return (
        stacked_images_tensor,
        stacked_class_ids_tensor,
        class_names_list,
        image_filepaths_list,
    )

# Initializing Dataset

dataset = ImagesDataset("./training_data", 100, 100)


# Spliting Data

train_portion = 0.8
valid_portion = 0.17
test_portion= 1 - (train_portion + valid_portion)
train_data, val_data, test_data = random_split(
    dataset, [train_portion, valid_portion, test_portion]
)

# Subset Slicing (for testing functionality ) 

# subset_indices = torch.randperm(len(dataset))[:1000]
# subset_dataset = Subset(dataset, subset_indices)
# train_portion = 0.77
# valid_portion = 0.2
# test_portion= 1 - (train_portion + valid_portion)
# train_data, val_data, test_data = random_split(
#     subset_dataset, [train_portion, valid_portion, test_portion]
# )



# Training Data Augmentaion
print("Total Before Augmentation: ", len(train_data)+ len(val_data) + len(test_data) )
print("Train Data Before Augmentation: ",len(train_data))
train_data = SubsetAugmentation(train_data)

total_len = len(train_data)+ len(val_data) + len(test_data)  
print("Total : ",total_len)
print("Train Data: ",len(train_data))
print("Validation Data: ",len(val_data))
print("Test Data: ",len(test_data))

# Training Loop
def training_loop(network: nn.Module, train_data, eval_data, num_epochs, show_progress=False):
    torch.cuda.empty_cache()
    train_loader = DataLoader(train_data, batch_size=25, shuffle=True, collate_fn=stacking)
    val_loader = DataLoader(eval_data, batch_size=25, shuffle=False, collate_fn=stacking)
    epochs = tqdm(range(num_epochs)) if show_progress else range(num_epochs)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001,weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer,'min')
    criterion = nn.CrossEntropyLoss()
    epoch_train_loss = []
    patience = 7
    epoch_val_loss = []
    best_loss = float("inf")
    for epoch in epochs:
        torch.cuda.empty_cache()
        network.train()
        train_loss = []
        for input, targets, _1, _2 in train_loader:
            # print(input.shape)
            input = input.to(device)
            targets = targets.to(device)
            targets = targets.squeeze()
            output = network(input)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
           
        epoch_train_loss.append(np.mean(train_loss))
        network.eval()
        val_loss = []
        total =0 
        correct = 0
        with torch.no_grad():
            for input, targets, _1, _2 in val_loader:
                input = input.to(device)
                targets = targets.to(device)
                targets = targets.squeeze()
                output = network(input)
                val_loss_fn = criterion(output, targets)
                val_loss.append(val_loss_fn.item())
                _, predicted = torch.max(output, 1)  
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        avg_val_loss = np.mean(val_loss)
        scheduler.step(avg_val_loss)
        epoch_val_loss.append(avg_val_loss)
        
        print("\n","train epoch loss list",epoch_train_loss)
        print("\n","eval epoch loss list",epoch_val_loss)
        print(avg_val_loss,best_loss,"best loss should be updated:", True if avg_val_loss < best_loss else False,)
        print("Accuracy: ",  correct / total)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = 7
            print("patience reset")
            torch.save(network.state_dict(), "model.pth")
        else:
            patience -= 1

        if patience == 0:
            print("no loss improvement breaking the loop")
            break

    return (epoch_train_loss, epoch_val_loss)

# Ploting Function
def plot_losses(train_losses: list, eval_losses: list):
    plt.plot(train_losses, c="blue")
    plt.plot(eval_losses, c="orange")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(list(range(0, len(train_losses), 5)))
    plt.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.savefig("epoch_loss.pdf")


if __name__ == "__main__":
    from architecture import model
    
    class_names,class_ids  = dataset.filenames_classnames, dataset.classnames_to_ids
    num_classes = len(class_ids)
    print("Number of Classes: ", num_classes)
    network = model
    network.to(device)
    
    train_loss, eval_loss = training_loop(network, train_data, val_data, 100, show_progress=True)
    plot_losses(train_loss, eval_loss)

    test_loader = DataLoader(test_data, batch_size=25, shuffle=False,collate_fn=stacking)
    def predict(network = network , test_loader=test_loader,num_classes = num_classes ,class_ids=class_ids, class_names=class_names):
        total = 0
        TP = 0
        FN = 0
        model =  network
        model.to(device)
        state_dict = torch.load("model.pth")
        model.load_state_dict(state_dict=state_dict)
        model.eval()

        with torch.no_grad():
            for stacked_images_tensor, stacked_class_ids_tensor, _, _ in test_loader:
                stacked_images_tensor = stacked_images_tensor.to(device)
                stacked_class_ids_tensor = stacked_class_ids_tensor.to(device)
                outputs = model(stacked_images_tensor)
                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1) 
                print("Prediction:", pred, "True Value:", stacked_class_ids_tensor.squeeze())
                TP += (pred == stacked_class_ids_tensor.squeeze()).sum().item()
                FN += (pred != stacked_class_ids_tensor.squeeze()).sum().item()
                total += stacked_class_ids_tensor.size(0)
        accuracy = TP / total
        print(f"Total: {total}, True: {TP}, False: {FN}")
        print("Accuracy:", accuracy)

    predict()