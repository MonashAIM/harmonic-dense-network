import sys
import os
import glob
import json

if __name__ == "__main__":
    total_instance = int(sys.argv[1])

    dataset = {}
    dataset["name"] = "Covid 19 Lesion Segmentation"
    dataset["modality"] = "CT"

    dataset["numTraining"] = total_instance
    dataset["training"] = []

    img_path = os.path.join(
        "data",
        "covid",
        "images",
        "*",
    )

    fold_list = [0,1,2,3]

    img_files = glob.glob(img_path)
    index = 0
    for image_path in img_files:
        if len(dataset["training"]) < dataset["numTraining"]:
            label_path = image_path.replace("images", "labels")
            dataset["training"].append(
                {"id": index, "fold" : fold_list[index % len(fold_list)],"image": image_path, "label": label_path}
            )
            index += 1
        else:
            break

    with open("covid_dataset.json", "w") as f:
        json.dump(dataset, f)
