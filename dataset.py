from typing import Optional

import albumentations
import cv2
import pandas as pd
import torch
from albumentations import BboxParams
from albumentations.pytorch import ToTensorV2
from pylabel.dataset import Dataset as pylabel_dataset
from torch.utils.data import Dataset
import numpy as npa

class DACDataset(Dataset):
    def __init__(
        self,
        pylabel_ds: pylabel_dataset,
        resize_to: tuple[int, int] = (512,512),
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
        self.df = self.df.convert_dtypes(infer_objects=True)
        if self.split == "train" and transforms is not None:
            self.transforms = [
                albumentations.Resize(*resize_to),
                *transforms,
                ToTensorV2(),
            ]
        else:
            self.transforms = [albumentations.Resize(*resize_to), ToTensorV2()]
        self.T = albumentations.Compose(
            self.transforms, BboxParams(format="pascal_voc")
        )

        self.unique_images = self.df['img_id'].unique()
        self.category_to_int = {
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
        self.bboxesss = {}
        self.img_pths = {}
        for _, row in self.df.iterrows():
            img_id = row['img_id']
            
            if img_id not in self.bboxesss:
                xmin = float(max(row['ann_bbox_xmin'], 0))
                ymin = float(max(row['ann_bbox_ymin'], 0))
                xmax = float(min(row['ann_bbox_xmax'], row['img_width']))
                ymax = float(min(row['ann_bbox_ymax'], row['img_height']))
                label_id= int(row['cat_id'])
                # bboxes = albumentations.core.bbox_utils.convert_bboxes_to_albumentations([xmin, ymin, xmax, ymax, label_id], "pascal_voc",
            #                                                                          img.shape[:2], check_validity=True)
                # bboxes = albumentations.core.bbox_utils.convert_bboxes_from_albumentations(b, "yolo", img.shape[:2],
            #                                                                            check_validity=True)
                self.bboxesss[img_id] = np.array([[xmin,ymin,xmax,ymax,label_id]])
                self.img_pths[img_id] = row['img_folder'] + "/" + row['img_filename']
                continue
            xmin = float(max(row['ann_bbox_xmin'], 0))
            ymin = float(max(row['ann_bbox_ymin'], 0))
            xmax = float(min(row['ann_bbox_xmax'], row['img_width']))
            ymax = float(min(row['ann_bbox_ymax'], row['img_height']))
            label_id= int(row['cat_id'])
            self.bboxesss[img_id] = np.append(self.bboxesss[img_id],[[xmin, ymin, xmax, ymax, label_id]],axis=0)
        

    def _pascal_voc_to_yolo(self, box: list, size):
        cat = box[-1]
        # # cat_id = self.category_to_int[cat]
        # return [
        #     0,
        #     cat,
        #     ((x2 + x1) / (2 * image_w)),
        #     ((y2 + y1) / (2 * image_h)),
        #     (x2 - x1) / image_w,
        #     (y2 - y1) / image_h,
        # ]
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        
        return (0,cat,x,y,w,h)

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):

        # print(df)
        img_path = self.img_pths[self.unique_images[idx]]
        bboxes = self.bboxesss[self.unique_images[idx]]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.T(image=image, bboxes=bboxes)
        img = transformed["image"]
   
        return img, torch.Tensor(
            [
                self._pascal_voc_to_yolo(bbox, (img.shape[1], img.shape[2]))
                for bbox in bboxes
            ]
        )


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
