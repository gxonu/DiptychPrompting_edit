# Subject-Driven Image Editing with Diptych Prompting

## Overview
This implementation allows you to edit specific regions of a target image by inserting a subject from a reference image using in-context learning. Unlike the original implementation that replaces the entire right panel, this version supports partial masking for more precise editing.

## Required Inputs

### 1. Reference Image (`--reference_image_path`)
- Image containing the subject you want to use for editing
- The subject will be automatically segmented and extracted

### 2. Target Image (`--target_image_path`)
- The image you want to edit
- Only the masked region will be modified

### 3. Subject Name (`--subject_name`)
- Name of the subject to extract from the reference image
- Used for automatic segmentation (e.g., "dog", "chair", "person")

### 4. Editing Mask
You can specify the editing region using three methods:

#### Rectangle Mask (`--mask_type rectangle`)
- `--mask_x`: X coordinate of top-left corner
- `--mask_y`: Y coordinate of top-left corner
- `--mask_w`: Width of rectangle
- `--mask_h`: Height of rectangle

#### Circle Mask (`--mask_type circle`)
- `--mask_cx`: Center X coordinate
- `--mask_cy`: Center Y coordinate
- `--mask_radius`: Radius of the circle

#### Custom Mask (`--mask_type custom`)
- `--mask_path`: Path to a grayscale mask image
  - White pixels (255) = areas to edit
  - Black pixels (0) = areas to preserve

### 5. Text Prompt (`--prompt`)
- Describes how the subject should appear in the edited region
- Example: "the same dog sitting on a couch"

## Usage Examples

### Example 1: Rectangle Mask
Replace a specific rectangular area with a subject from reference:

```bash
python subject_driven_editing.py \
  --reference_image_path ./images/dog.jpg \
  --target_image_path ./images/living_room.jpg \
  --subject_name "dog" \
  --mask_type rectangle \
  --mask_x 200 --mask_y 300 --mask_w 300 --mask_h 400 \
  --prompt "the same dog sitting on a couch" \
  --output_path ./outputs/edited_room.png
```

### Example 2: Circle Mask
Add a subject to a circular region:

```bash
python subject_driven_editing.py \
  --reference_image_path ./images/flower.jpg \
  --target_image_path ./images/garden.jpg \
  --subject_name "flower" \
  --mask_type circle \
  --mask_cx 400 --mask_cy 300 --mask_radius 150 \
  --prompt "the same flower in the garden" \
  --output_path ./outputs/edited_garden.png
```

### Example 3: Custom Mask
Use a pre-defined mask for complex shapes:

```bash
python subject_driven_editing.py \
  --reference_image_path ./images/cat.jpg \
  --target_image_path ./images/sofa.jpg \
  --subject_name "cat" \
  --mask_type custom \
  --mask_path ./masks/sofa_mask.png \
  --prompt "the same cat lying on the sofa" \
  --output_path ./outputs/edited_sofa.png
```

## Advanced Parameters

### Generation Quality
- `--num_inference_steps`: Number of denoising steps (default: 30, higher = better quality but slower)
- `--guidance_scale`: Text guidance strength (default: 3.5)
- `--ctrl_scale`: ControlNet conditioning scale (default: 0.95)

### Attention Control
- `--attn_enforce`: Cross-panel attention strength (default: 1.3, higher = stronger subject transfer)

### Image Size
- `--width`: Output width (default: 768)
- `--height`: Output height (default: 768)

### Other
- `--seed`: Random seed for reproducibility (default: 42)
- `--pixel_offset`: Padding for edge artifacts removal (default: 8)

## Output Files

The script generates three files:
1. `[output_path]`: Final edited target image
2. `[output_path]_mask.png`: Visualization of the mask used
3. `[output_path]_full_diptych.png`: Full diptych showing reference and edited result side-by-side

## Tips for Best Results

1. **Clear Subject Segmentation**: Ensure the subject in the reference image is clearly distinguishable
2. **Appropriate Mask Size**: Make the mask region large enough to accommodate the subject
3. **Consistent Lighting**: Works best when reference and target have similar lighting conditions
4. **Descriptive Prompts**: Be specific about how the subject should appear in the target context

## Creating Custom Masks

For complex editing regions, create a custom mask:
1. Open your target image in an image editor
2. Create a new layer
3. Paint white (255) where you want to edit
4. Paint black (0) where you want to preserve
5. Save as grayscale PNG

## System Requirements

- GPU with >40GB VRAM (for FLUX model)
- CUDA-capable GPU
- Python 3.10+
- Dependencies from requirements.txt