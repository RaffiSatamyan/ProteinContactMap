import time

from imports import *
from models import ContactMapPredictionTransformer, ContactMapPredictionLSTM
from pdb_files_utils import model_ems_2, alphabet
from utils import (CostomDataset, EmbeddingCollate, count_parameters, estimate_vram_usage, FocalLoss,
                   calculate_roc_auc_with_inverted_mask, plot_roc_curve)

pdb_files_path = "toy_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = ContactMapPredictionTransformer(mini_emb_dim=320, hidden_dim=32, num_layers=8, num_heads=4,
                                         dropout=0.1)

model = ContactMapPredictionLSTM(embedding_dim=320)
# Loading the saved model
try:
    checkpoint = torch.load("last_model.pt", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    pass

# Transfer the model to cuda
model = model.to(device)

# Defining losses
bce = nn.BCELoss(reduction="none")
fl = FocalLoss(alpha=0.9, gamma=2)

# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Defining treshold for metrics and prediction making
treshold = 0.5
num_epochs = 1

print(f"Estimated [parameters: {count_parameters(model)}, video_ram: {estimate_vram_usage(model):.4f} GB (if float32)]")

# Create a data loader
dataset = CostomDataset(pdb_files_path)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
train_loader.collate_fn = EmbeddingCollate(model_ems_2, alphabet)
time.sleep(0.25)

# Training by epochs
for epoch in range(num_epochs):
    true, total, tp, fp, fn = 0, 0, 0, 0, 0
    pbar = tqdm(train_loader)
    for i, [batch_x, batch_x_mask, batch_y, batch_y_mask] in enumerate(pbar):
        # Transfer the Tensors to cuda
        batch_x = batch_x.to(device)
        batch_x_mask = batch_x_mask.to(device)
        batch_y = batch_y.to(device)
        batch_y_mask = batch_y_mask.to(device)

        # Get outputs of model
        optimizer.zero_grad()
        outputs = model(batch_x)

        # Parameters to understand how training is going
        actual_ones = torch.sum(batch_y)  # how many positive classes are there
        predicted_ones = torch.sum(outputs * (batch_y_mask == 0) > treshold)  # how many do we predict as positive
        total_size = torch.sum(batch_y_mask == 0)  # total size of contact map, without pad
        k = (torch.sum(batch_y == 0) - torch.sum(batch_y_mask)) / torch.sum(batch_y)  # a ratio, of our pred/actual pred
        padded_size = torch.sum(batch_x_mask)  # the padded size of output

        # Loss function, considering the mask
        loss = fl(outputs, batch_y)
        loss = (loss * (~batch_y_mask)).sum(dim=(-1, -2)) / (~batch_y_mask).sum(dim=(-1, -2))
        loss = loss.mean()
        loss.backward()
        optimizer.step()  # Gradient descent
        with torch.no_grad():
            outputs = outputs.detach().cpu()
            batch_y_mask = batch_y_mask.detach().cpu()
            batch_y = batch_y.detach().cpu()

        if i % 10 == 0:
            auc = calculate_roc_auc_with_inverted_mask(batch_y, outputs.detach(), batch_y_mask)  # counting auc
            plot_roc_curve(batch_y, outputs.detach(), batch_y_mask)  # plotting ROC_AUC curve

            # For beautiful printing the progress
            pbar.set_description(f"loss: {loss.item():.4f} | actual_ones: {actual_ones:.4f} |"
                                 f" predicted_ones: {predicted_ones} | total_size: {total_size:.4f}| AUC: {auc:.4f}")

            # Here is the part of metrics which are counted using masks
            tp += torch.sum(((outputs >= treshold) * (~batch_y_mask)) * batch_y)
            fp += torch.sum(((outputs >= treshold) * (~batch_y_mask)) * ~batch_y.to(torch.bool))
            fn += torch.sum(((outputs <= treshold) * (~batch_y_mask)) * batch_y)
            true += torch.sum((outputs >= treshold) * (~batch_y_mask) == batch_y * (~batch_y_mask)) - torch.sum(
                batch_y_mask)
            total += torch.sum(~batch_y_mask)

    # Metrics
    acc = true / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(tp)
    print(f"accuracy: {acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}")

    # Saving the model
    torch.save({'model_state_dict': model.state_dict()}, "last_model.pt")
