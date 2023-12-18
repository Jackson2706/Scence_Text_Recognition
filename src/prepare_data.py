"""
import built-in libraries
    :os: to communicate with folders as well as files
    :urllib.request as request: to download data from the web
    :xml.etree.ElementTree as ET: to read data from .xml format file
    :zipfile: to read and extract data from .zip format data
"""
import os
import shutil
import urllib.request as request
import xml.etree.ElementTree as ET
import zipfile


def prepare_data(url, data_dir):
    """
    This function is to download ICDAR2003 from url
    :url: the url of file needed downloading
    :data_dir: the path which save raw data
    """
    os.makedirs(data_dir, exist_ok=True)

    target_path = os.path.join(data_dir, "scene.zip")
    if not os.path.exists(target_path):
        request.urlretrieve(url, target_path)
    zip_file = zipfile.ZipFile(target_path)
    zip_file.extractall(data_dir)
    zip_file.close()


def prepare_dataset(word_xml_path):
    """
    This function is to prepare dataset from raw data which is download
    by above function
    :word_xml_path: the path which contains files saving information of
    this data, such as: location of bounding boxes, OCR word
    """
    current_folder_path = os.path.dirname(word_xml_path)
    image_paths = []
    image_sizes = []
    bounding_boxes = []
    image_labels = []

    tree = ET.parse(word_xml_path)
    root = tree.getroot()
    for image_objects in root:
        image_path = os.path.join(current_folder_path, image_objects[0].text)
        image_paths.append(image_path)

        width = int(image_objects[1].attrib["x"])
        height = int(image_objects[1].attrib["y"])
        image_sizes.append((width, height))
        bb_of_image = []
        labels_of_image = []
        for bbs in image_objects.findall("taggedRectangles"):
            for bb in bbs:
                # remove all case with non-number or non-alphabet
                if not bb[0].text.isalnum():
                    continue

                # remove all case with unicode
                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue

                bbox = [
                    int(float(bb.attrib["x"])),
                    int(float(bb.attrib["y"])),
                    int(float(bb.attrib["width"])),
                    int(float(bb.attrib["height"])),
                ]
                label = bb[0].text

                bb_of_image.append(bbox)
                labels_of_image.append(label)

        bounding_boxes.append(bb_of_image)
        image_labels.append(labels_of_image)

    assert len(image_paths) == len(image_sizes)
    assert len(image_sizes) == len(bounding_boxes)
    assert len(bounding_boxes) == len(image_labels)

    return image_paths, image_sizes, bounding_boxes, image_labels


def convert_to_YOLO_format(image_paths, bounding_boxes, image_sizes):
    """
    This function is to convert raw data to YOLO format data
        :image_paths: is a list containing the path of images
        :bounding_boxes: is a list containing all bounding boxes per image
        :image_sizes: is a lust containing resolution of each image
    """
    yolov8_data = []
    for image_path, bboxes, (image_width, image_height) in zip(
        image_paths, bounding_boxes, image_sizes
    ):
        yolov8_labels = []
        for bbox in bboxes:
            x, y, w, h = bbox

            # Normalized the config
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Because there is only one class, `text`
            class_id = 0

            # Convert to Yolov8 format
            label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolov8_labels.append(label)
        yolov8_data.append((image_path, yolov8_labels))
    return yolov8_data


def save_data(data, save_dir):
    # Create folder if it's not created
    os.makedirs(save_dir, exist_ok=True)

    # Make image and label folder
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, yolov8_labels in data:
        # Save image to image folder
        shutil.copy(image_path, os.path.join(save_dir, "images"))
        # Save label to label folder
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(
            os.path.join(save_dir, "labels", f"{image_name}.txt"), "w"
        ) as f:
            for label in yolov8_labels:
                f.write(f"{label}\n")
                f.write(f"{label}\n")
