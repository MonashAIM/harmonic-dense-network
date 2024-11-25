from torch.utils.data import DataLoader
from data.dataset_torchio import get_isles_22_dwi_subjects
import torchio as tio
if __name__ == "__main__":

    sj = get_isles_22_dwi_subjects()
    torch_io_ds = tio.SubjectsDataset(sj)
    print(len(torch_io_ds))
    torch_io_dl = DataLoader(torch_io_ds, batch_size=8, shuffle=True)
    print(len(torch_io_dl))
