import sys
import os

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import json
import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import torchvision.transforms as T
import tqdm
import imageio
import cv2
import mediapy as media
import matplotlib.pyplot as plt
from PIL import Image

def extra_args(parser):
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        help="Input image to condition on.",
    )
    parser.add_argument(
        "--target",
        "-T",
        type=str,
        help="Target image to estimate the pose.",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default=os.path.join(ROOT_DIR, "pose_estimation"),
        help="Output directory",
    )
    parser.add_argument("--size", type=int, default=128, help="Input image maxdim")
    parser.add_argument(
        "--out_size",
        type=str,
        default="128",
        help="Output image size, either 1 or 2 number (w h)",
    )

    parser.add_argument("--focal", type=float, default=131.25, help="Focal length")
    parser.add_argument("--radius", type=float, default=1.3, help="Camera distance")
    parser.add_argument("--z_near", type=float, default=0.8)
    parser.add_argument("--z_far", type=float, default=1.8)
    parser.add_argument(
        "--elevation",
        "-e",
        type=float,
        default=0.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument("--fps", type=int, default=15, help="FPS of video")
    parser.add_argument("--gif", action="store_true", help="Store gif instead of mp4")
    parser.add_argument(
        "--no_vid",
        action="store_true",
        help="Do not store video (only image frames will be written)",
    )
    parser.add_argument("--lrate", type=float, default=1e-2)
    parser.add_argument("--n_steps", type=int, default=500, help="Number of steps for pose optimization.")
    return parser


def main():
    config = {
        'input': './input/1.png',
        'target': './input/2.png',
        'output': './pose_estimation'
    }
    input_image_np = np.array(Image.open(config['input']).convert("RGB"))
    target_image_np = np.array(Image.open(config['target']).convert("RGB"))

    media.show_images({
        'Source': input_image_np,
        'Target': target_image_np
    })
    args, conf = util.args.parse_args(
        extra_args, default_expname="srn_car", default_data_format="srn", jupyter=True
    )
    args.resume = True
    os.makedirs(args.output, exist_ok=True)

    device = util.get_cuda(args.gpu_id[0])

    z_near, z_far = args.z_near, args.z_far
    focal = torch.tensor(args.focal, dtype=torch.float32, device=device)

    in_sz = args.size
    sz = list(map(int, args.out_size.split()))
    if len(sz) == 1:
        H = W = sz[0]
    else:
        assert len(sz) == 2
        W, H = sz

    net = make_model(conf["model"]).to(device=device).load_weights(args)

    # Create the renderer.
    renderer = NeRFRenderer.from_conf(
        conf["renderer"], eval_batch_size=args.ray_batch_size
    ).to(device=device)
    render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True)
    image_to_tensor = util.get_image_to_tensor_balanced()

    # Encoding the input image.
    print(f"Input image: {config['input']}")
    input_image = Image.fromarray(input_image_np)
    input_image = T.Resize(in_sz)(input_image)
    input_image = image_to_tensor(input_image).to(device=device)
    input_pose = torch.eye(4)
    input_pose[2, -1] = args.radius

    print(f"Target image: {config['target']}")
    target_image = Image.fromarray(target_image_np)
    target_image = T.Resize(in_sz)(target_image)
    target_image_flatten = np.reshape(target_image, [-1, 3]) / 255.0
    target_image_flatten = torch.from_numpy(target_image_flatten).float().to(device=device)

    cam_pose = torch.clone(input_pose.detach()).unsqueeze(0)
    cam_pose.requires_grad = True

    print("Input pose:")
    print(f"{input_pose}")
    print("Init pose:")
    print(f"{cam_pose[0]}")

    # Create optimizer.
    optimizer = torch.optim.Adam(params=[cam_pose], lr=args.lrate)
    n_steps = 100 + 1

    # Loss.
    mse_loss = torch.nn.MSELoss()

    # Sampling.
    n_rays = 1024
    sampling = 'center'

    # Pose optimization.
    predicted_poses = []
    fine_patches = []
    gt_patches = []

    for i_step in range(n_steps):
        # Encode.
        net.encode(
            input_image.unsqueeze(0), input_pose.unsqueeze(0).to(device=device), focal,
        )

        render_rays = util.gen_rays(cam_pose, W, H, focal, z_near, z_far)
        render_rays_flatten = render_rays.view(-1, 8)
        assert render_rays_flatten.shape[0] == H * W
        if sampling == 'random':
            idxs_sampled = torch.randint(0, H * W, (n_rays,))
        elif sampling == 'center':
            frac = 0.5
            mask = torch.zeros((H, W))
            h_low = int(0.5 * (1 - frac) * H)
            h_high = int(0.5 * (1 + frac) * H)
            w_low = int(0.5 * (1 - frac) * W)
            w_high = int(0.5 * (1 + frac) * W)
            mask[h_low:h_high, w_low:w_high] = 1
            mask = mask.reshape(H * W)

            idxs_masked = torch.where(mask > 0)[0]
            idxs_sampled = idxs_masked[torch.randint(0, idxs_masked.shape[0], (n_rays,))]
        elif sampling == 'patch':
            frac = 0.25
            mask = torch.zeros((H, W))
            h_low = int(0.5 * (1 - frac) * H)
            h_high = int(0.5 * (1 + frac) * H)
            w_low = int(0.5 * (1 - frac) * W)
            w_high = int(0.5 * (1 + frac) * W)
            mask[h_low:h_high, w_low:w_high] = 1
            mask = mask.reshape(H * W)

            idxs_sampled = torch.where(mask > 0)[0]

        render_rays_sampled = render_rays_flatten[idxs_sampled].to(device=device)

        rgb, _ = render_par(render_rays_sampled[None])
        loss = mse_loss(rgb, target_image_flatten[idxs_sampled][None])

        optimizer.zero_grad()
        loss.backward()

        if i_step % 10 == 0:
            predicted_poses.append(torch.clone(cam_pose[0]).detach().numpy())
            fine_patches.append(torch.clone(rgb[0]).detach().cpu().numpy().reshape(32, 32, 3))
            gt_patches.append(torch.clone(target_image_flatten[idxs_sampled]).detach().cpu().numpy().reshape(32, 32, 3))

            #         pose_pred = predicted_poses[-1].copy()
            #         pose_pred[2, -1] -= args.radius
            #         pose_pred = pose_input @ pose_pred
            #         error_R, error_t = compute_pose_error(pose_pred, pose_target)
            print(f"Step {i_step}, loss: {loss}")

        optimizer.step()

if __name__ == '__main__':
    main()
