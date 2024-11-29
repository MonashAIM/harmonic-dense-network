from models.HarDNet import HarDNet


if __name__ == "__main__": # pragma: no cover
    model = HarDNet(arch="39DS")
    print(len(model.layers))
    model2 = HarDNet(arch="68")
    print(len(model2.layers))
