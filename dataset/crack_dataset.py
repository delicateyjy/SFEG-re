import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    """用于裂缝数据集的数据集类。"""

    def __init__(self, args):
        """初始化此数据集类。

        参数：
            args（选项类）——存储所有实验标志；需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, args)
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()

        self.phase = args.phase

    def __getitem__(self, index):
        """
        返回一个数据点及其元数据。

        参数：
            index - - 用于数据索引的随机整数

        返回包含 A、B、A_paths 和 B_paths 的字典
            image (tensor) - - 一张图片
            label (tensor) - - 其对应的分割
            A_paths (str) - - 图片路径
            B_paths (str) - - 图片路径（与 A_paths 相同）
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """返回数据集中的图片总数。"""
        return len(self.img_paths)


