import torch
import torch.nn.functional as F

from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import os
import numpy as np
from PIL import Image
import argparse

from diffusers.models.attention_processor import Attention

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
from transformers import AutoProcessor, pipeline, AutoModelForMaskGeneration


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    object_detector,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    segmentator,
    processor,
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    detect_pipeline,
    segmentator,
    segment_processor,
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(detect_pipeline, image, labels, threshold, detector_id)
    detections = segment(segmentator, segment_processor, image, detections, polygon_refinement)

    return np.array(image), detections


class CustomFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, height=44, width=88, attn_enforce=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.height = height
        self.width = width
        self.num_pixels = height * width
        self.step = 0
        self.attn_enforce = attn_enforce

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        self.step += 1
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)


        if self.attn_enforce != 1.0:
            attn_probs = (torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale).softmax(dim=-1)
            img_attn_probs = attn_probs[:, :, -self.num_pixels:, -self.num_pixels:]
            img_attn_probs = img_attn_probs.reshape((batch_size, attn.heads, self.height, self.width, self.height, self.width))
            img_attn_probs[:, :, :, self.width//2:, :, :self.width//2] *= self.attn_enforce
            img_attn_probs = img_attn_probs.reshape((batch_size, attn.heads, self.num_pixels, self.num_pixels))
            attn_probs[:, :, -self.num_pixels:, -self.num_pixels:] = img_attn_probs
            hidden_states = torch.einsum('bhqk,bhkd->bhqd', attn_probs, value)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

        
def _apply_hard_inpainting(original_img: Image.Image,
                           edited_img: Image.Image,
                           mask_img_L: Image.Image) -> Image.Image:
    """
    (1 - M) * original + M * edited  (M은 0~1의 grayscale)
    original_img, edited_img, mask_img_L 의 크기는 동일해야 함.
    """
    ori = np.asarray(original_img, dtype=np.float32)
    edt = np.asarray(edited_img, dtype=np.float32)
    m = np.asarray(mask_img_L, dtype=np.float32) / 255.0   # HxW [0,1]
    m = m[..., None]                                       # HxWx1

    out = (1.0 - m) * ori + m * edt
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def create_editing_mask(width, height, mask_type='rectangle', **kwargs):
    """
    Create different types of masks for editing specific regions.
    
    Args:
        width: Width of the mask
        height: Height of the mask
        mask_type: Type of mask ('rectangle', 'circle', 'custom')
        **kwargs: Additional parameters for each mask type
            - For 'rectangle': x, y, w, h
            - For 'circle': cx, cy, radius
            - For 'custom': mask_path (path to a mask image)
    """
    mask = np.zeros((height, width, 3), dtype=np.uint8)
   
    if mask_type == 'rectangle':
        x = kwargs.get('x', width//4)
        y = kwargs.get('y', height//4)
        w = kwargs.get('w', width//2)
        h = kwargs.get('h', height//2)
        mask[y:y+h, x:x+w] = 255
        
    elif mask_type == 'circle':
        cx = kwargs.get('cx', width//2)
        cy = kwargs.get('cy', height//2)
        radius = kwargs.get('radius', min(width, height)//4)
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask[dist_from_center <= radius] = 255
        
    elif mask_type == 'custom':
        mask_path = kwargs.get('mask_path')
        if mask_path and os.path.exists(mask_path):
            custom_mask = Image.open(mask_path).convert('L').resize((width, height))
            mask_array = np.array(custom_mask)
            mask[mask_array > 127] = 255
        else:
            raise ValueError(f"Custom mask path not found: {mask_path}")
    
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subject-Driven Image Editing with Diptych Prompting')
    
    # Basic parameters
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    parser.add_argument('--ctrl_scale', type=float, default=0.95)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--pixel_offset', type=int, default=8)
    
    # Input images
    parser.add_argument('--reference_image_path', type=str, required=True,
                        help='Path to the reference image containing the subject')
    parser.add_argument('--target_image_path', type=str, required=True,
                        help='Path to the target image to edit')
    parser.add_argument('--subject_name', type=str, required=True,
                        help='Name of the subject to extract from reference')
    
    # Mask parameters
    parser.add_argument('--mask_type', type=str, default='rectangle',
                        choices=['rectangle', 'circle', 'custom'],
                        help='Type of mask to use for editing region')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Path to custom mask image (for mask_type=custom)')
    
    # Rectangle mask parameters
    parser.add_argument('--mask_x', type=int, default=None,
                        help='X coordinate for rectangle mask')
    parser.add_argument('--mask_y', type=int, default=None,
                        help='Y coordinate for rectangle mask')
    parser.add_argument('--mask_w', type=int, default=None,
                        help='Width for rectangle mask')
    parser.add_argument('--mask_h', type=int, default=None,
                        help='Height for rectangle mask')
    
    # Circle mask parameters
    parser.add_argument('--mask_cx', type=int, default=None,
                        help='Center X for circle mask')
    parser.add_argument('--mask_cy', type=int, default=None,
                        help='Center Y for circle mask')
    parser.add_argument('--mask_radius', type=int, default=None,
                        help='Radius for circle mask')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt describing the desired edit')
    parser.add_argument('--output_path', type=str, default='edited_result.png',
                        help='Path to save the edited image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for generation')
    parser.add_argument('--num_inference_steps', type=int, default=30,
                        help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                        help='Guidance scale for generation')

    args = parser.parse_args()

    # Build pipeline
    print("Loading models...")
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", 
        torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    base_attn_procs = pipe.transformer.attn_processors.copy()

    # Initialize detection and segmentation models
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).cuda()
    segment_processor = AutoProcessor.from_pretrained(segmenter_id)
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=torch.device("cuda"))

    def segment_subject(image, object_name):
        """Extract and segment the subject from the reference image"""
        image_array, detections = grounded_segmentation(
            object_detector,
            segmentator,
            segment_processor,
            image=image,
            labels=[object_name],
            threshold=0.3,
            polygon_refinement=True,
        )
        if not detections:
            raise ValueError(f"Could not detect '{object_name}' in the reference image")
        
        # Extract subject with white background
        segment_result = image_array * np.expand_dims(detections[0].mask / 255, axis=-1) + \
                        np.ones_like(image_array) * (1 - np.expand_dims(detections[0].mask / 255, axis=-1)) * 255
        segmented_image = Image.fromarray(segment_result.astype(np.uint8))
        return segmented_image

    # Process images
    width = args.width + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2
    size = (width*2, height)

    # Load and process reference image
    print(f"Processing reference image: {args.reference_image_path}")
    reference_image = load_image(args.reference_image_path).resize((width, height)).convert("RGB")
    segmented_reference = segment_subject(reference_image, args.subject_name)

    # Load target image
    print(f"Loading target image: {args.target_image_path}")
    target_image = load_image(args.target_image_path).resize((width, height)).convert("RGB")

    # Create diptych with reference on left, target on right
    diptych_image = np.concatenate([np.array(segmented_reference), np.array(target_image)], axis=1)
    diptych_image = Image.fromarray(diptych_image)

    # Create editing mask (only edit specified region in right panel)
    print(f"Creating {args.mask_type} mask for editing...")
    mask_kwargs = {}
    
    if args.mask_type == 'rectangle':
        mask_kwargs = {
            'x': args.mask_x if args.mask_x is not None else width//4,
            'y': args.mask_y if args.mask_y is not None else height//4,
            'w': args.mask_w if args.mask_w is not None else width//2,
            'h': args.mask_h if args.mask_h is not None else height//2
        }
    elif args.mask_type == 'circle':
        mask_kwargs = {
            'cx': args.mask_cx if args.mask_cx is not None else width//2,
            'cy': args.mask_cy if args.mask_cy is not None else height//2,
            'radius': args.mask_radius if args.mask_radius is not None else min(width, height)//4
        }
    elif args.mask_type == 'custom':
        mask_kwargs = {'mask_path': args.mask_path}

#     # Create mask for right panel only
#     right_panel_mask = create_editing_mask(width, height, args.mask_type, **mask_kwargs)
    
#     # Combine: left panel (no mask) + right panel (with mask)
#     full_mask = np.concatenate([np.zeros((height, width, 3)), right_panel_mask], axis=1)
#     mask_image = Image.fromarray(full_mask.astype(np.uint8)).convert("L")

#     # Save mask for visualization
#     mask_image.save(args.output_path.replace('.png', '_mask.png'))
#     print(f"Mask saved to: {args.output_path.replace('.png', '_mask.png')}")

    # --- 마스크 생성은 그대로 ---
    right_panel_mask_rgb = create_editing_mask(width, height, args.mask_type, **mask_kwargs)
    right_panel_mask_L = Image.fromarray(
        right_panel_mask_rgb[..., 0] if right_panel_mask_rgb.ndim == 3 else right_panel_mask_rgb
    ).convert("L")

    full_mask_L = Image.new("L", (width * 2, height))
    full_mask_L.paste(right_panel_mask_L, (width, 0))
    full_mask_L.save(args.output_path.replace('.png', '_mask.png'))
    print(f"Mask saved to: {args.output_path.replace('.png', '_mask.png')}")

    # --- 새로 추가: 우패널을 실제로 '가린' 타깃 만들기 ---
    def paint_mask_on_image(img: Image.Image, mask_L: Image.Image, fill=128):
        arr = np.asarray(img).copy()
        m = np.asarray(mask_L)
        arr[m > 0] = fill  # 회색으로 덮기(255로 해도 동작)
        return Image.fromarray(arr)

    
    masked_target = paint_mask_on_image(target_image, right_panel_mask_L, fill=128)

    # 컨트롤용 디프틱: [좌 = segmented_reference, 우 = masked_target]
    diptych_control_image = Image.fromarray(
        np.concatenate([np.array(segmented_reference), np.array(masked_target)], axis=1)
    )
    
    # Create prompt for diptych editing
    diptych_prompt = f"A diptych with two side-by-side images. On the left, a photo of {args.subject_name}. On the right, {args.prompt}"

    # Set up attention processor for cross-panel attention
    new_attn_procs = base_attn_procs.copy()
    for i, (k, v) in enumerate(new_attn_procs.items()):
        new_attn_procs[k] = CustomFluxAttnProcessor2_0(
            height=height // 16, 
            width=width // 16 * 2, 
            attn_enforce=args.attn_enforce
        )
    pipe.transformer.set_attn_processor(new_attn_procs)

    
    
    # --- 생성 호출 ---
    print("Generating edited image...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result = pipe(
        prompt=diptych_prompt,
        height=size[1],
        width=size[0],
        control_image=diptych_control_image,  # ← 가린 타깃이 들어간 디프틱
        control_mask=full_mask_L,             # ← 우패널 마스크가 붙은 풀 마스크(L)
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=args.ctrl_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt="",
        true_guidance_scale=args.guidance_scale
    ).images[0]


    # --- 오른쪽 패널만 꺼내기 ---
    result_right = result.crop((width, 0, width*2, height))
    result_right = result_right.crop(
        (args.pixel_offset, args.pixel_offset, width - args.pixel_offset, height - args.pixel_offset)
    )

    # 타깃/마스크도 동일 영역으로 크롭
    target_cropped = target_image.crop(
        (args.pixel_offset, args.pixel_offset, width - args.pixel_offset, height - args.pixel_offset)
    )
    mask_cropped_L = right_panel_mask_L.crop(
        (args.pixel_offset, args.pixel_offset, width - args.pixel_offset, height - args.pixel_offset)
    )

    # 하드 인페인팅: (1-M)*target + M*edited
    final_edited = _apply_hard_inpainting(target_cropped, result_right, mask_cropped_L)

    from datetime import datetime
    target_base = os.path.splitext(os.path.basename(args.target_image_path))[0]
    subject_base = args.subject_name.replace(" ", "_")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"{target_base}_{subject_base}_{timestamp}"

    # 경로 생성
    edit_output_dir = os.path.join("output", "edit", experiment_name)
    os.makedirs(edit_output_dir, exist_ok=True)

    # 저장 경로
    output_path = os.path.join(edit_output_dir, "edited.png")
    mask_output_path = os.path.join(edit_output_dir, "mask.png")
    diptych_output_path = os.path.join(edit_output_dir, "full_diptych.png")
    
    # ▼ 디버깅 프린트 & 저장 (파이프 호출 전에!)
    print('control_image size:', diptych_control_image.size)         # (2W, H)
    print('control_mask size, mode:', full_mask_L.size, full_mask_L.mode)  # (2W, H), 'L'
    print('mask unique:', np.unique(np.array(full_mask_L)))          # 기대값: [  0 255]

    dip_path = os.path.join(edit_output_dir, "debug_diptych_control.png")
    full_mask_L.save(os.path.join(edit_output_dir, "debug_full_mask_L.png"))
    diptych_control_image.save(dip_path)
    
    
    # ✅ 최종 결과/마스크 저장
    final_edited.save(output_path)
    # 사람이 보기 쉬운 그레이스케일 마스크 저장(우패널 크롭 버전 권장)
    mask_cropped_L.save(mask_output_path)

    print(f"✅ Edited image saved to: {output_path}")
    print(f"✅ Mask saved to: {mask_output_path}")
    
    # ✅ full diptych도 저장 (파이프 결과 전체). 마스크는 full_mask_L 사용
    full_result = pipe(
        prompt=diptych_prompt,
        height=size[1],
        width=size[0],
        control_image=diptych_control_image, 
        control_mask=full_mask_L,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=args.ctrl_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt="",
        true_guidance_scale=args.guidance_scale
    ).images[0]
    full_result.save(diptych_output_path)

    print(f"✅ Full diptych saved to: {diptych_output_path}")