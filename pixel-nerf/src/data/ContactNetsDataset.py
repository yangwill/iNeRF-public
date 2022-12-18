import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor, construct_contact_nets_transformation
import yaml


class ContactNetsDataset(torch.utils.data.Dataset):
    """
    Dataset from ContactNets
    """

    def __init__(
        self, path, stage="train", image_size=(480, 640), world_scale=1.0, split_seed=1234, val_frac=0.2, test_frac=0.2
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        # self.base_path = path + "_" + stage
        self.base_path = path
        self.dataset_name = os.path.basename(path)

        print("Loading ContactNets dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        self.intrins = sorted(
            # glob.glob(os.path.join(self.base_path, "*", "transforms.json"))
            glob.glob(os.path.join(self.base_path, "*", "realsense_pose.yaml"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = 0.8
        self.z_far = 3.0
        # self.z_near = 4.0
        # self.z_far = 6.0
        self.lindisp = False
        self._load_split(val_frac, test_frac, split_seed)

    def __len__(self):
        return len(self.intrins)

    def _load_split(self, val_frac, test_frac, seed):
        permute_file = os.path.join(self.base_path, "split_{}.pth".format(seed))
        num_objs = len(self)
        if os.path.isfile(permute_file):
            print("Loading dataset split from {}".format(permute_file))
            permute = torch.load(permute_file)
        else:
            if val_frac == 0 and test_frac == 0:
                warn("creating empty validation and test sets")
            state = np.random.get_state()
            np.random.seed(seed)
            print("Created dataset split in {}".format(permute_file))

            permute = np.random.permutation(num_objs)
            torch.save(permute, permute_file)
            np.random.set_state(state)

        val_size = int(val_frac * num_objs)
        test_size = int(test_frac * num_objs)
        train_end = num_objs - (val_size + test_size)
        val_end = num_objs - test_size
        # TODO revert back to uncommented when there is more data
        # if self.stage == 'train':
        #     subset = permute[:train_end]
        # elif self.stage == 'val':
        #     subset = permute[train_end:val_end]
        # elif self.stage == 'test':
        #     subset = permute[val_end:]

        subset = permute[:train_end]
        self.intrins = [self.intrins[i] for i in subset]
        print(len(self))
        assert len(self) == len(subset)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "masked_rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "gt_poses", "*")))
        assert len(rgb_paths) == len(pose_paths)
        camera_pose = 0
        camera_rotation = 0

        with open(self.intrins[0], "r") as intrinfile:
            camera_intrins = yaml.safe_load(intrinfile)
            camera_pose_x = camera_intrins['cam0']['pose']['position']['x']
            camera_pose_y = camera_intrins['cam0']['pose']['position']['y']
            camera_pose_z = camera_intrins['cam0']['pose']['position']['z']
            camera_rotation_x = camera_intrins['cam0']['pose']['rotation']['x']
            camera_rotation_y = camera_intrins['cam0']['pose']['rotation']['y']
            camera_rotation_z = camera_intrins['cam0']['pose']['rotation']['z']
            camera_pose = np.array([camera_pose_x, camera_pose_y, camera_pose_z])
            camera_rotation = np.array([camera_rotation_x, camera_rotation_y, camera_rotation_z])

        camera_transform = construct_contact_nets_transformation(camera_pose, camera_rotation)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        # print(rgb_paths)
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            # print(pose)
            # print(pose)
            pose = camera_transform @ pose

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        permute = np.random.permutation(all_imgs.shape[0])[:30]
        # print(permute)
        all_imgs = all_imgs[permute]
        all_poses = all_poses[permute]
        all_masks = all_masks[permute]
        all_bboxes = all_bboxes[permute]

        # print(all_imgs.shape)
        # print(all_bboxes.shape)
        # print(all_masks.shape)
        # TODO get focal distance and maybe cx and cy

        focal = 1

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            # cx *= scale
            # cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            # "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
