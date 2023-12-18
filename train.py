"""
import the built-in libraries
    :from fire import Fire:  an useful alternative for args.pars, help dev code
    faster
"""
import os

import torch
import ultralytics
import yaml
from fire import Fire
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

"""
import the modified libraries
    :from src import prepare_data: import library whose function
    is to download data from the fixed url and save data in a fix direction
"""
from torch.utils.data import DataLoader

from src import (
    CRNN,
    STRDataset,
    convert_to_YOLO_format,
    decode,
    encode,
    eval,
    fit,
    prepare_data,
    prepare_dataset,
    save_data,
    split_bounding_boxes,
)


def main(
    url,
    data_dir,
    ocr_dir,
    val_size=0.2,
    test_size=0.125,
    epochs=200,
    imgsz=512,
    train_batch_size=2,
    test_batch_size=1,
    hidden_size=512,
    n_layers=1,
    dropout_prob=0.2,
    unfreeze_layers=3,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    crnn_model_path="./weights",
):
    print("*" * 100)
    print("Download data from url: {}".format(url))
    print("Saving the direction: {}".format(data_dir))
    prepare_data(url, data_dir)
    print("Status: Done\n")

    print("*" * 100)
    print("Create the dataset")
    word_xml_path = ""
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, "words.xml")):
                word_xml_path = os.path.join(folder_path, "words.xml")
    image_paths, image_sizes, bounding_boxes, image_labels = prepare_dataset(
        word_xml_path
    )
    print("Status: Done\n")
    print("*" * 100)
    print("Check ultralytics libraries")
    ultralytics.checks()
    print("Status: Done\n")

    print("*" * 100)
    print("Convert raw data to YOLO format dataset for text detection")
    yolov8_data = convert_to_YOLO_format(
        image_paths=image_paths,
        bounding_boxes=bounding_boxes,
        image_sizes=image_sizes,
    )

    print("Status: Done\n")

    print("*" * 100)
    print("Train, Val, Test dataset splitting")
    is_shuffle = True
    seed = 123
    # Split train and test dataset from yolov8_data which was created above
    train_data, test_data = train_test_split(
        yolov8_data, test_size=val_size, random_state=seed, shuffle=is_shuffle
    )
    test_data, val_data = train_test_split(
        test_data, test_size=test_size, random_state=seed, shuffle=is_shuffle
    )
    save_data(
        data=train_data,
        save_dir="./dataset/train",
    )
    save_data(
        data=val_data,
        save_dir="./dataset/val",
    )
    save_data(
        data=test_data,
        save_dir="./dataset/test",
    )
    print("Status: Done\n")

    print("*" * 100)
    print("Create data.yaml file")
    train_path = input("Train path: ")
    test_path = input("Test path: ")
    val_path = input("Val path: ")
    yaml_path = input("Yaml path: ")
    data_yaml = {
        "path": "dataset",
        "train": train_path,
        "test": test_path,
        "val": val_path,
        "nc": 1,
        "name": "text",
    }

    yolo_yaml_path = os.path.join(
        yaml_path,
        "data.yaml",
    )
    with open(yolo_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print("Status: Done\n")

    print("*" * 100)
    print("Training YOLO model")
    model_name = input("Yolo model name: ")
    # Load model
    model = YOLO(f"{model_name}.yaml").load(f"{model_name}.pt")
    model.to("cuda")

    # Train model

    # model.train(
    #     data=yolo_yaml_path,
    #     epochs=epochs,
    #     imgsz=imgsz,
    #     project="models",
    #     name=f"{model_name}/detect/train",
    # )
    print("Status: Done\n")
    print("*" * 100)
    print("Create OCR Dataset")
    split_bounding_boxes(image_paths, image_labels, bounding_boxes, ocr_dir)
    root_dir = ocr_dir

    image_paths = []
    labels = []

    # Read labels from text file
    with open(os.path.join(root_dir, "labels.txt"), "r", encoding="UTF-8") as f:
        for label in f:
            labels.append(label.strip().split("\t")[1])
            image_paths.append(label.strip().split("\t")[0])
    print(f"Total images: {len(image_paths)}")

    letters = [char.split(".")[0].lower() for char in labels]
    letters = "".join(letters)
    letters = sorted(list(set(list(letters))))

    chars = "".join(letters)

    # for "blank" character
    blank_character = "-"
    chars += blank_character
    vocab_size = len(chars)

    print(f"Vocab: {chars}")
    print(f"Vocab size: {vocab_size}")
    print("Status: Done")

    print("*" * 100)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    max_label_len = max([len(label) for label in labels])

    idx_to_char = {char: idx for idx, char in char_to_idx.items()}

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5
                ),
                transforms.Grayscale(num_output_channels=1),
                transforms.GaussianBlur(3),
                transforms.RandomAffine(degrees=1, shear=1),
                transforms.RandomPerspective(
                    distortion_scale=0.3, p=0.5, interpolation=3
                ),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    }
    # Split train and test dataset from yolov8_data which was created above
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths,
        labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle,
    )

    train_dataset = STRDataset(
        X=X_train,
        y=y_train,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["train"],
    )
    test_dataset = STRDataset(
        X=X_test,
        y=y_test,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )
    val_dataset = STRDataset(
        X=X_val,
        y=y_val,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )
    print("Status: Done\n")

    print("*" * 100)
    print("Defining the CRNN model")
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout_prob,
        unfreeze_layers=unfreeze_layers,
    ).to(device)
    print("Status: Done\n")
    print("*" * 100)
    print("Training CRNN model")
    lr = 0.001
    weight_decay = 1e-5
    scheduler_step_size = epochs * 0.4
    blanl_char = "-"
    criterion = nn.CTCLoss(blank=char_to_idx[blanl_char], zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.1
    )
    os.makedirs(crnn_model_path, exist_ok=True)
    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        model_save_path=crnn_model_path,
    )


if __name__ == "__main__":
    Fire(main)
