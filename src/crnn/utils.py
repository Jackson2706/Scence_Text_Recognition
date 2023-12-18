import os

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm


def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    labels = []

    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        img = Image.open(img_path)
        for label, bb in zip(img_label, bbs):
            # Crop image
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
            # filter out if 90% of the crop image is black or white
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue
            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue
            if len(img_label) < 3:
                continue

            # save image
            file_name = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, file_name))

            new_img_path = os.path.join(save_dir, file_name)

            label = new_img_path + "\t" + label
            labels.append(label)

            count = count + 1
    print(f"Created {count} images")

    # Write  labels to  a text file
    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def encode(label, char_to_idx, max_label_len):
    encoded_label = torch.tensor(
        [char_to_idx[char] for char in label], dtype=torch.long
    )
    label_len = len(encoded_label)
    lengths = torch.tensor(label_len, dtype=torch.long)
    padded_labels = F.pad(
        encoded_label, (0, max_label_len - label_len), value=0
    )
    return padded_labels, lengths


def decode(encoded_sequence, idx_to_char, blank_char="-"):
    decoded_sequence = []

    for seq in encoded_sequence:
        decode_label = []
        for idx, token in enumerate(seq):
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decode_label.append(char)
        decode_label = "".join(decode_label)
        decoded_sequence.append(decode_label)
    return decoded_sequence


def eval(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels, labels_len in tqdm(dataloader):
            inputs, labels, labels_len = (
                inputs.to(device),
                labels.to(device),
                labels_len.to(device),
            )

            output = model(inputs)
            logits_len = torch.full(
                size=(len(inputs),), fill_value=output.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(output, labels, logits_len, labels_len)
            losses.append(loss.item())
    losses = sum(losses) / len(losses)
    return losses


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    model_save_path,
):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        batch_train_losses = []
        model.train()
        for inputs, labels, labels_len in tqdm(train_loader):
            inputs, labels, labels_len = (
                inputs.to(device),
                labels.to(device),
                labels_len.to(device),
            )
            optimizer.zero_grad()
            output = model(inputs)
            logits_len = torch.full(
                size=(len(inputs),), fill_value=output.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(output, labels, logits_len, labels_len)
            loss.backward()
            # Gradient clipping: avoid case NaN in loss function
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = eval(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(
            f"EPOCH {epoch+1}/{epochs}:\t Train loss: {train_loss:.4f}\t Val Loss: {val_loss:.4f}"
        )

        scheduler.step()
    torch.save(model.state_dict(), os.path.join(model_save_path, "crnn.pt"))
    return train_losses, val_losses
    return train_losses, val_losses
