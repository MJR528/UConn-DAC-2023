from decimal import Decimal
from typing import Optional

import albumentations
import cv2
from albumentations import BboxParams
from albumentations.pytorch import ToTensorV2
from pylabel.dataset import Dataset as pylabel_dataset
from torch.utils.data import Dataset
import pandas as pd


class DACDataset(Dataset):
    def __init__(
        self,
        pylabel_ds: pylabel_dataset,
        split: str = "train",
        transforms: Optional[list] = None,
    ):
        super().__init__()
        # self.root_dir = root_dir
        self.ds = pylabel_ds
        # self.ds.df.convert_dtypes(infer_objects=True)
        self.split = split
        self.df: pd.DataFrame = self.ds.df[self.ds.df["split"] == self.split]
        self.df.reset_index(inplace=True, drop=True)
        if self.split == "train" and transforms is not None:
            self.transforms = [
                albumentations.Resize(512, 512),
                *transforms,
                ToTensorV2(),
            ]
        else:
            self.transforms = [albumentations.Resize(512, 512), ToTensorV2()]
        self.T = albumentations.Compose(
            self.transforms, BboxParams(format="pascal_voc")
        )

        self.unique_images = self.df.img_filename.unique()
        self._category_to_int = {
            "Motor Vehicle": 0,
            "Non-motorized Vehicle": 1,
            "Pedestrian": 2,
            "Traffic Light-Red Light": 3,
            "Traffic Light-Yellow Light": 4,
            "Traffic Light-Green Light": 5,
            "Traffic Light-Off": 6,
            "Solid lane line": 7,
            "Dotted lane line": 8,
            "Crosswalk": 9,
        }

    def _pascal_voc_to_yolo(self, pascal_voc_lst: list, image_w, image_h):
        x1, x2, y1, y2, cat = pascal_voc_lst
        cat_id = self._category_to_int[cat]
        return [
            cat_id,
            ((x2 + x1) / (2 * image_w)),
            ((y2 + y1) / (2 * image_h)),
            (x2 - x1) / image_w,
            (y2 - y1) / image_h,
        ]

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):

        df = self.df[self.df.img_filename == self.unique_images[idx]]
        df.reset_index(inplace=True, drop=True)
        # print(df)
        img_path = df.img_folder[0] + "/" + df.img_filename.iloc[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []
        for index, row in df.iterrows():
            # arranges labels in this format [x_min, y_min, x_max, y_max, label]
            # Get bounding box coordinates and make sure that they don't extend beyond image boundary
            xmin = float(Decimal(row.ann_bbox_xmin).max(0))
            ymin = float(Decimal(row.ann_bbox_ymin).max(0))
            xmax = float(Decimal(row.ann_bbox_xmax).min(row.img_width))
            ymax = float(Decimal(row.ann_bbox_ymax).min(row.img_height))
            label = row.cat_name
            bboxes.append([xmin, ymin, xmax, ymax, label])

            transformed = self.T(image=image, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            # bboxes = albumentations.core.bbox_utils.convert_bboxes_to_albumentations(bboxes, "pascal_voc",
            #                                                                          img.shape[:2], check_validity=True)
            # bboxes = albumentations.core.bbox_utils.convert_bboxes_from_albumentations(bboxes, "yolo", img.shape[:2],
            #                                                                            check_validity=True)
            # bboxes = np.array(bboxes)
        return img, [
            self._pascal_voc_to_yolo(bbox, img.shape[1], img.shape[2])
            for bbox in bboxes
        ]


if __name__ == "__main__":
    from pylabel.importer import ImportVOC

    ds = ImportVOC(
        "/home/cc/Dev/IdeaProjects/UConn/Ding/dac2023-gpu/data/train/Annotations",
        "/home/cc/Dev/IdeaProjects/UConn/Ding/dac2023-gpu/data/train/JPEGImages",
    )
    ds.splitter.GroupShuffleSplit(train_pct=0.7, val_pct=0.1, test_pct=0.2)
    print(ds.df["split"].unique())
    # train_ds = DACDataset(ds, "train")
    # print(train_ds[0])
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    # for img, bboxes in train_dl:
    #     print(img.shape)
    #     print(bboxes)
    #     break
