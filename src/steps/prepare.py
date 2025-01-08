import os
import glob
import json
import dvc.api

if __name__ == "__main__":
    params = dvc.api.params_show()
    dataset_name = params["prepare"]["dataset"]
    dataset = {}
    dataset["name"] = dataset_name
    dataset["modality"] = "CT"

    dataset["trainNum"] = int(params["prepare"]["trainNum"])
    dataset["training"] = []

    img_path = os.path.join(
        "src",
        "data",
        "covid",
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

    with open(fr".\src\data\{dataset_name}_dataset.json", "w") as f:
        json.dump(dataset, f)