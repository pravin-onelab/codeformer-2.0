import os
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import normalize

from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils import img2tensor, imwrite, tensor2img
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.utils.misc import is_gray

pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}

# download weights
if not os.path.exists("CodeFormer/weights/CodeFormer/codeformer.pth"):
    load_file_from_url(
        url=pretrain_model_url["codeformer"], model_dir="CodeFormer/weights/CodeFormer", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/facelib/detection_Resnet50_Final.pth"):
    load_file_from_url(
        url=pretrain_model_url["detection"], model_dir="CodeFormer/weights/facelib", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/facelib/parsing_parsenet.pth"):
    load_file_from_url(
        url=pretrain_model_url["parsing"], model_dir="CodeFormer/weights/facelib", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth"):
    load_file_from_url(
        url=pretrain_model_url["realesrgan"], model_dir="CodeFormer/weights/realesrgan", progress=True, file_name=None
    )


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


upsampler = set_realesrgan()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

os.makedirs("output", exist_ok=True)


def inference_app_with_bboxes(image, bounding_boxes, background_enhance, face_upsample, upscale, codeformer_fidelity):
    # Ensure input image is properly loaded (image is already a numpy array)
    img = image
    print("\timage size:", img.shape)

    upscale = int(upscale)  # convert type to int
    if upscale > 4:  # avoid memory issues for too large upscale
        upscale = 4
    if upscale > 2 and max(img.shape[:2]) > 1000:  # avoid memory issues for large image resolution
        upscale = 2
    if max(img.shape[:2]) > 1500:  # further avoid memory issues for very large image resolution
        upscale = 1
        background_enhance = False
        face_upsample = False

    # Initialize FaceRestoreHelper to handle face upscaling and restoration
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=None,  # No need for detection model
        save_ext="png",
        use_parse=True,
        device=device,
    )
    bg_upsampler = upsampler if background_enhance else None
    face_upsampler = upsampler if face_upsample else None

    # For each face bounding box, crop and enhance
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        face_img = img[y1:y2, x1:x2]  # Crop the face region

        # Resize the cropped face to the required size (512x512) if needed
        face_img_resized = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(face_img_resized, threshold=5)
        if face_helper.is_gray:
            print("\tgrayscale input: True")
        face_helper.cropped_faces = [face_img_resized]

        # Process the face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # Prepare data
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            restored_face_resized = cv2.resize(restored_face, (x2 - x1, y2 - y1))  # Resize back to the original bounding box size

            # Paste the enhanced face back into the original image
            img[y1:y2, x1:x2] = restored_face_resized

    # Optionally upscale the background
    if bg_upsampler is not None:
        img = bg_upsampler.enhance(img, outscale=upscale)[0]

    return img
