import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import glob
import os


class LGEDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=224, pat_id=0, mode='fewshot'):
        self.list_path = list_path
        self.crop_size = crop_size
        search_path = os.path.join(list_path, "trainB/*_{}_lge*.png".format(pat_id))
        self.img_ids = glob.glob(search_path)
        self.label_ids = glob.glob(os.path.join(list_path, "trainBmask/*_{}_lge*.png".format(pat_id)))
        if mode == 'fulldata':
            self.img_ids = glob.glob(os.path.join(list_path, "trainB/pat*lge*.png"))
            self.label_ids = glob.glob(os.path.join(list_path, "trainBmask/pat*lge*.png"))
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        self.id_to_trainid = {0: 0, 85: 1, 212: 2, 255: 3}
        for img_file, label_file in zip(self.img_ids, self.label_ids):
            name = os.path.basename(img_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')  # H,W,C
        img_w, img_h = image.size
        if img_w != self.crop_size:
            border_size = int((img_w - self.crop_size) // 2)
            image = image.crop((border_size, border_size, img_w-border_size, img_h-border_size))
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        return image.copy(), name


if __name__ == '__main__':
    dst = LGEDataSet("../../data/mscmrseg/", pat_id=10, crop_size=224)
    trainloader = data.DataLoader(dst, batch_size=1, shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, _ = data
        print(imgs.shape)
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        plt.axis('off')
        plt.imshow(img)
        plt.show()
