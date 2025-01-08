import os
import glob
import json
import yaml

if __name__ == "__main__":
    params = yaml.safe_load(open("./src/params.yml"))
    dataset_name = params["prepare-data"]["dataset"]
    dataset = {}
    dataset["name"] = dataset_name
    dataset["modality"] = "CT"

    dataset["trainNum"] = int(params["prepare-data"]["train_size"])
    dataset["training"] = []

    img_path = os.path.join(
        "src",
        "data",
        dataset_name,
        "images",
        "*",
    )

    fold_list = [0, 1, 2, 3]

    img_files = glob.glob(img_path)
    index = 0
    for image_path in img_files:
        if len(dataset["training"]) < dataset["trainNum"]:
            label_path = image_path.replace("images", "labels")
            dataset["training"].append(
                {
                    "id": index,
                    "fold": fold_list[index % len(fold_list)],
                    "image": image_path,
                    "label": label_path,
                }
            )
            index += 1
        else:
            break

    with open(f".\src\data\{dataset_name}_dataset.json", "w") as f:
        json.dump(dataset, f)
