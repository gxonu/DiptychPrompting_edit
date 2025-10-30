import numpy as np
from PIL import Image
import argparse
import os

def create_editing_mask(width, height, mask_type='rectangle', **kwargs):
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
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask[dist <= radius] = 255

    elif mask_type == 'custom':
        mask_path = kwargs.get('mask_path')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Custom mask path not found: {mask_path}")
        mask = Image.open(mask_path).convert("L").resize((width, height))
        mask = np.array(mask)
        mask = np.repeat(mask[..., None], 3, axis=-1)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", type=str, required=True)
    parser.add_argument("--target_image", type=str, required=True)
    parser.add_argument("--mask_x", type=int, required=True)
    parser.add_argument("--mask_y", type=int, required=True)
    parser.add_argument("--mask_w", type=int, required=True)
    parser.add_argument("--mask_h", type=int, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--pixel_offset", type=int, default=0)
    args = parser.parse_args()

    # ===== 실제 Diptych 코드와 동일한 크기 계산 =====
    width  = args.width  + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2

    # ===== 좌우 패널 구성 =====
    ref = Image.open(args.reference_image).resize((width, height)).convert("RGB")
    tgt = Image.open(args.target_image).resize((width, height)).convert("RGB")

    # Diptych (2W x H)
    diptych = np.concatenate([np.array(ref), np.array(tgt)], axis=1)

    # ===== 마스크 생성 (오른쪽 패널 기준 좌표) =====
    right_mask_rgb = create_editing_mask(width, height, mask_type='rectangle',
                                         x=args.mask_x, y=args.mask_y,
                                         w=args.mask_w, h=args.mask_h)
    right_mask_L = Image.fromarray(right_mask_rgb[..., 0]).convert("L")

    full_mask_L = Image.new("L", (width * 2, height))
    full_mask_L.paste(right_mask_L, (width, 0))

    # ===== 시각화: 마스크 위치를 diptych 위에 덮어보기 =====
    overlay = np.array(diptych).copy()
    m = np.array(full_mask_L)
    overlay[m > 0] = [128, 128, 128]  # 마스크 영역을 회색으로 표시

    out_dir = "mask_check"
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(overlay).save(os.path.join(out_dir, "debug_mask_overlay.png"))
    full_mask_L.save(os.path.join(out_dir, "debug_full_mask_L.png"))

    # --- 추가 코드: 오른쪽 패널(= target + gray mask)만 따로 저장 ---
    tgt_np = np.array(tgt).copy()
    mask_np = np.array(right_mask_L)
    tgt_np[mask_np > 0] = [128, 128, 128]  # 마스크 부분 회색 처리
    masked_target = Image.fromarray(tgt_np)

    masked_target_output_path = os.path.join(out_dir, "debug_masked_target.png")
    masked_target.save(masked_target_output_path)
    print(f"✅ Masked target (right panel only) saved to: {masked_target_output_path}")
    
    print("✅ Saved:")
    print(" - mask_check/debug_mask_overlay.png (회색영역 = 적용 마스크)")
    print(" - mask_check/debug_full_mask_L.png (full binary mask)")
