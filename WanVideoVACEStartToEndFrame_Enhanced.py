import torch
import torch.nn.functional as F
import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# 导入ComfyUI的工具函数
from comfy.utils import common_upscale


class WanVideoVACEStartToEndFrame_Enhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": "Number of frames to generate",
                    },
                ),
                "empty_frame_level": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Level of empty frames (0.0 = black, 1.0 = white)",
                    },
                ),
            },
            "optional": {
                "start_image": (
                    "IMAGE",
                    {"tooltip": "Optional start image for the sequence"},
                ),
                "end_image": (
                    "IMAGE",
                    {"tooltip": "Optional end image for the sequence"},
                ),
                "control_images": (
                    "IMAGE",
                    {"tooltip": "Optional control images for the sequence"},
                ),
                "inpaint_mask": (
                    "MASK",
                    {"tooltip": "Optional inpaint mask for the sequence"},
                ),
                "insert_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for inserted image",
                    },
                ),
                "insert_mask_to_insert_image": (
                    "MASK",
                    {
                        "tooltip": "Optional mask to use for the inserted image (overrides insert_image alpha channel)"
                    },
                ),
                "insert_image_num1": (
                    "IMAGE",
                    {
                        "tooltip": "Optional additional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_num1_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for additional inserted image 1",
                    },
                ),
                "insert_mask_to_insert_image_num1": (
                    "MASK",
                    {"tooltip": "Optional mask for additional inserted image 1"},
                ),
                "insert_image_num2": (
                    "IMAGE",
                    {
                        "tooltip": "Optional additional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_num2_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for additional inserted image 2",
                    },
                ),
                "insert_mask_to_insert_image_num2": (
                    "MASK",
                    {"tooltip": "Optional mask for additional inserted image 2"},
                ),
                "insert_image_num3": (
                    "IMAGE",
                    {
                        "tooltip": "Optional additional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_num3_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for additional inserted image 3",
                    },
                ),
                "insert_mask_to_insert_image_num3": (
                    "MASK",
                    {"tooltip": "Optional mask for additional inserted image 3"},
                ),
                "insert_image_num4": (
                    "IMAGE",
                    {
                        "tooltip": "Optional additional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_num4_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for additional inserted image 4",
                    },
                ),
                "insert_mask_to_insert_image_num4": (
                    "MASK",
                    {"tooltip": "Optional mask for additional inserted image 4"},
                ),
                "insert_image_num5": (
                    "IMAGE",
                    {
                        "tooltip": "Optional additional image to insert (alpha channel is used as mask)"
                    },
                ),
                "insert_frame_num5_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index for additional inserted image 5",
                    },
                ),
                "insert_mask_to_insert_image_num5": (
                    "MASK",
                    {"tooltip": "Optional mask for additional inserted image 5"},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/Enhanced"
    DESCRIPTION = "Enhanced helper node to create start/end frame batch and masks for VACE with support for multiple insert images"

    def process(
        self,
        num_frames,
        empty_frame_level,
        start_image=None,
        end_image=None,
        control_images=None,
        inpaint_mask=None,
        insert_image=None,
        insert_frame_index=0,
        insert_mask_to_insert_image=None,
        insert_image_num1=None,
        insert_frame_num1_index=0,
        insert_mask_to_insert_image_num1=None,
        insert_image_num2=None,
        insert_frame_num2_index=0,
        insert_mask_to_insert_image_num2=None,
        insert_image_num3=None,
        insert_frame_num3_index=0,
        insert_mask_to_insert_image_num3=None,
        insert_image_num4=None,
        insert_frame_num4_index=0,
        insert_mask_to_insert_image_num4=None,
        insert_image_num5=None,
        insert_frame_num5_index=0,
        insert_mask_to_insert_image_num5=None,
    ):

        # Determine image dimensions and device from any available input image
        H, W, device = None, None, None

        # Get dimensions from any available image
        for img in [
            start_image,
            end_image,
            control_images,
            insert_image,
            insert_image_num1,
            insert_image_num2,
            insert_image_num3,
            insert_image_num4,
            insert_image_num5,
        ]:
            if img is not None:
                H, W = img.shape[1], img.shape[2]
                device = img.device
                break

        # If no images provided, use default dimensions
        if H is None or W is None:
            H, W = 480, 832
            device = torch.device("cpu")

        log.info(
            f"Creating frame sequence with dimensions: {H}x{W}, frames: {num_frames}"
        )

        # Initialize output batch with empty frames
        out_batch = torch.full(
            (num_frames, H, W, 3), empty_frame_level, dtype=torch.float32, device=device
        )

        # Initialize mask batch (all zeros initially)
        mask_batch = torch.zeros((num_frames, H, W), dtype=torch.float32, device=device)

        # Process start_image
        if start_image is not None:
            log.info("Processing start_image")
            start_rgb = start_image[0, :, :, :3]  # Take first image, RGB channels

            # Resize if needed
            if start_rgb.shape[0] != H or start_rgb.shape[1] != W:
                log.warning(
                    f"Resizing start image from {start_rgb.shape} to ({H}, {W})"
                )
                resized_start = (
                    common_upscale(
                        start_rgb.movedim(-1, 0).unsqueeze(
                            0
                        ),  # Add batch dim and move channels
                        W,
                        H,
                        "lanczos",
                        "disabled",
                    )
                    .squeeze(0)
                    .movedim(0, -1)
                )  # Remove batch dim and move channels back
            else:
                resized_start = start_rgb

            out_batch[0] = resized_start
            mask_batch[0] = 1.0

        # Process end_image
        if end_image is not None:
            log.info("Processing end_image")
            end_rgb = end_image[0, :, :, :3]  # Take first image, RGB channels

            # Resize if needed
            if end_rgb.shape[0] != H or end_rgb.shape[1] != W:
                log.warning(f"Resizing end image from {end_rgb.shape} to ({H}, {W})")
                resized_end = (
                    common_upscale(
                        end_rgb.movedim(-1, 0).unsqueeze(
                            0
                        ),  # Add batch dim and move channels
                        W,
                        H,
                        "lanczos",
                        "disabled",
                    )
                    .squeeze(0)
                    .movedim(0, -1)
                )  # Remove batch dim and move channels back
            else:
                resized_end = end_rgb

            out_batch[-1] = resized_end
            mask_batch[-1] = 1.0

        # Process control_images
        if control_images is not None:
            log.info(f"Processing control_images with {control_images.shape[0]} frames")
            for i in range(min(control_images.shape[0], num_frames)):
                control_rgb = control_images[i, :, :, :3]

                # Resize if needed
                if control_rgb.shape[0] != H or control_rgb.shape[1] != W:
                    log.warning(
                        f"Resizing control image {i} from {control_rgb.shape} to ({H}, {W})"
                    )
                    resized_control = (
                        common_upscale(
                            control_rgb.movedim(-1, 0).unsqueeze(
                                0
                            ),  # Add batch dim and move channels
                            W,
                            H,
                            "lanczos",
                            "disabled",
                        )
                        .squeeze(0)
                        .movedim(0, -1)
                    )  # Remove batch dim and move channels back
                else:
                    resized_control = control_rgb

                out_batch[i] = resized_control
                mask_batch[i] = 1.0

        # Helper function to process insert images
        def process_insert_image(current_insert_image, frame_index, insert_mask):
            if current_insert_image is None:
                return

            # Clamp frame index to valid range
            frame_index = max(0, min(frame_index, num_frames - 1))

            log.info(f"Processing insert image at frame {frame_index}")

            # Get RGB channels
            if current_insert_image.shape[-1] == 4:
                insert_image_rgb = current_insert_image[0, :, :, :3]
                alpha_mask = current_insert_image[0, :, :, 3]
            else:
                insert_image_rgb = current_insert_image[0, :, :, :3]
                alpha_mask = None

            # Resize insert_image_rgb to match target H, W
            if insert_image_rgb.shape[0] != H or insert_image_rgb.shape[1] != W:
                log.warning(
                    f"Resizing insert image RGB from {insert_image_rgb.shape} to ({H}, {W})"
                )
                resized_insert_rgb = (
                    common_upscale(
                        insert_image_rgb.movedim(-1, 0).unsqueeze(
                            0
                        ),  # Add batch dim and move channels
                        W,
                        H,
                        "lanczos",
                        "disabled",
                    )
                    .squeeze(0)
                    .movedim(0, -1)
                )  # Remove batch dim and move channels back
            else:
                resized_insert_rgb = insert_image_rgb

            # Determine which mask to use
            if insert_mask is not None:
                # Use provided mask
                blend_mask = insert_mask[0] if insert_mask.dim() > 2 else insert_mask
                log.info("Using provided insert mask")
            elif alpha_mask is not None:
                # Use alpha channel as mask
                blend_mask = alpha_mask
                log.info("Using alpha channel as mask")
            else:
                # No mask, use full image
                blend_mask = torch.ones((H, W), dtype=torch.float32, device=device)
                log.info("No mask provided, using full image")

            # Resize mask if needed
            if blend_mask.shape[0] != H or blend_mask.shape[1] != W:
                log.warning(
                    f"Resizing blend mask from {blend_mask.shape} to ({H}, {W})"
                )
                blend_mask = (
                    torch.nn.functional.interpolate(
                        blend_mask.unsqueeze(0).unsqueeze(
                            0
                        ),  # Add batch and channel dims
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )  # Remove batch and channel dims

            # Expand mask to match RGB channels
            blend_mask_expanded = blend_mask.unsqueeze(-1).expand(-1, -1, 3)

            # Get existing frame
            existing_frame = out_batch[frame_index]

            # Blend images: white mask areas show insert image, black areas show existing
            # Flipped logic: black mask areas show existing, white mask areas show insert
            blended_frame = (
                blend_mask_expanded * resized_insert_rgb
                + (1.0 - blend_mask_expanded) * existing_frame
            )

            # Update output
            out_batch[frame_index] = blended_frame
            mask_batch[frame_index] = torch.maximum(mask_batch[frame_index], blend_mask)

        # Process all insert images
        insert_images_data = [
            (insert_image, insert_frame_index, insert_mask_to_insert_image),
            (
                insert_image_num1,
                insert_frame_num1_index,
                insert_mask_to_insert_image_num1,
            ),
            (
                insert_image_num2,
                insert_frame_num2_index,
                insert_mask_to_insert_image_num2,
            ),
            (
                insert_image_num3,
                insert_frame_num3_index,
                insert_mask_to_insert_image_num3,
            ),
            (
                insert_image_num4,
                insert_frame_num4_index,
                insert_mask_to_insert_image_num4,
            ),
            (
                insert_image_num5,
                insert_frame_num5_index,
                insert_mask_to_insert_image_num5,
            ),
        ]

        for img, idx, mask in insert_images_data:
            process_insert_image(img, idx, mask)

        # Process inpaint_mask if provided
        if inpaint_mask is not None:
            log.info("Processing inpaint mask")
            # Resize inpaint mask if needed
            inpaint_resized = inpaint_mask
            if inpaint_mask.shape[-2] != H or inpaint_mask.shape[-1] != W:
                log.warning(
                    f"Resizing inpaint mask from {inpaint_mask.shape} to match ({H}, {W})"
                )
                inpaint_resized = torch.nn.functional.interpolate(
                    (
                        inpaint_mask.unsqueeze(0)
                        if inpaint_mask.dim() == 2
                        else inpaint_mask.unsqueeze(0)
                    ),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Apply inpaint mask to all frames
            if inpaint_resized.dim() == 2:
                # Single mask for all frames
                for i in range(num_frames):
                    mask_batch[i] = torch.maximum(mask_batch[i], inpaint_resized)
            else:
                # Multiple masks
                for i in range(min(inpaint_resized.shape[0], num_frames)):
                    mask_batch[i] = torch.maximum(mask_batch[i], inpaint_resized[i])

        log.info(
            f"Generated batch shape: {out_batch.shape}, mask shape: {mask_batch.shape}"
        )

        return (out_batch, mask_batch)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoVACEStartToEndFrame_Enhanced": WanVideoVACEStartToEndFrame_Enhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoVACEStartToEndFrame_Enhanced": "WanVideo VACE Start To End Frame (Enhanced)",
}
