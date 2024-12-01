from data.torchio_dataset import get_isles_22


if __name__ == "__main__":  # pragma: no cover
    dataloader = get_isles_22()
    print(len(dataloader))
