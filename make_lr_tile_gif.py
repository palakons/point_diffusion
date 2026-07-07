#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil

VIEWS = [
    ('man-full', 12),
    ('man-full', 18),
    ('man-full', 21),
    ('man-full', 75),
    ('man-full', 83),
    ('man-full', 86),
    ('man-full', 88),
    ('man-full', 106),
    ('man-full', 107),
    ('man-full', 116),
    ('man-full', 118),
    ('man-full', 131),
    ('man-full', 140),
    ('man-full', 147),
    ('man-full', 148),
    ('man-full', 155),
    ('man-full', 172),
    ('man-full', 177),
    ('man-full', 179),
    ('man-full', 186),
    ('man-full', 188),
    ('man-full', 193),
    ('man-full', 218),
    ('man-full', 227),
    ('man-full', 253),
    ('man-full', 266),
    ('man-full', 268),
    ('man-full', 278),
    ('man-full', 282),
    ('man-full', 288),
    ('man-full', 294),
    ('man-full', 299),
    ('man-full', 309),
    ('man-full', 312),
    ('man-full', 313),
    ('man-full', 319),
    ('man-full', 324),
    ('man-full', 329),
    ('man-full', 346),
    ('man-full', 359),
    ('man-full', 384),
    ('man-full', 399),
    ('man-full', 430),
    ('man-full', 447),
    ('man-full', 451),
    ('man-full', 455),
    ('man-full', 459),
    ('man-full', 472),
    ('man-full', 478),
    ('man-full', 480),
    ('man-full', 481),
    ('man-full', 488),
    ('man-full', 513),
    ('man-full', 525),
    ('man-full', 547),
    ('man-full', 556),
    ('man-full', 559),
    ('man-full', 569),
    ('man-full', 571),
    ('man-full', 593),
]

BASE_DIR = Path(
    "/data/palakons/ddpm_cond_slow/"
    "xattn-B256_dim768_samplemse-lr1e-4-constant-500k-smooth-infer/"
    "inference"
)

OUT_DIR = BASE_DIR / "tiles_lr_frames"
GIF_PATH = BASE_DIR / "tile_lr_fr002-037_hd_4x.gif"

FRAME_IDS = range(2, 38)   # 2..37 inclusive
SIDES = ["left", "right"]

SCENE_GRID_W = 8
SCENE_GRID_H = 8

# Because each scene has left/right pair:
GRID_W = SCENE_GRID_W * 2  # 16 image columns
GRID_H = SCENE_GRID_H      # 8 rows

# Set this if original tiles are too large.
# For 1920-wide GIF, 16 columns means 120 px per image tile.
# Example: TILE_SIZE = (120, 120)
TILE_SIZE = None

LABEL_H = 24
BG_COLOR = (245, 245, 245)
MISSING_COLOR = (230, 230, 230)
TEXT_COLOR = (0, 0, 0)
MISSING_TEXT_COLOR = (120, 120, 120)

GIF_FPS = 6
GIF_WIDTH = 1920*4


def data_file_to_token(data_file: str) -> str:
    """
    ('man-full', 12) -> combo_man-full_left_sc-12_fr-2.png

    If your actual filenames use combo_man_full instead of combo_man-full,
    change this return line to:
        return data_file.replace("-", "_")
    """
    return data_file


def image_path(data_file: str, side: str, scene_id: int, frame_id: int) -> Path:
    token = data_file_to_token(data_file)
    return BASE_DIR / f"combo_{token}_{side}_sc-{scene_id}_fr-{frame_id}.png"


def all_candidate_paths():
    for frame_id in FRAME_IDS:
        for data_file, scene_id in VIEWS:
            for side in SIDES:
                yield image_path(data_file, side, scene_id, frame_id)


def infer_tile_size():
    for p in all_candidate_paths():
        if p.exists():
            with Image.open(p) as im:
                return im.size

    raise RuntimeError(
        "No existing image files found. Cannot infer tile size. "
        "Set TILE_SIZE manually, e.g. TILE_SIZE = (120, 120)."
    )


def load_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def draw_one_frame(frame_id: int, tile_w: int, tile_h: int, font):
    canvas_w = GRID_W * tile_w
    canvas_h = GRID_H * (tile_h + LABEL_H)

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    missing = []

    total_slots = SCENE_GRID_W * SCENE_GRID_H

    for idx in range(total_slots):
        scene_row = idx // SCENE_GRID_W
        scene_col = idx % SCENE_GRID_W

        if idx >= len(VIEWS):
            # blank left/right pair for excess cells
            for side_offset in [0, 1]:
                x = (scene_col * 2 + side_offset) * tile_w
                y = scene_row * (tile_h + LABEL_H)
                draw.rectangle(
                    [x, y, x + tile_w - 1, y + tile_h + LABEL_H - 1],
                    fill=MISSING_COLOR,
                )
            continue

        data_file, scene_id = VIEWS[idx]

        for side_offset, side in enumerate(SIDES):
            col = scene_col * 2 + side_offset
            row = scene_row

            x = col * tile_w
            y = row * (tile_h + LABEL_H)

            p = image_path(data_file, side, scene_id, frame_id)
            label = f"sc-{scene_id} {side} fr-{frame_id}"

            if p.exists():
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if im.size != (tile_w, tile_h):
                        im = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    canvas.paste(im, (x, y))

                draw.rectangle(
                    [x, y + tile_h, x + tile_w - 1, y + tile_h + LABEL_H - 1],
                    fill=BG_COLOR,
                )
                draw.text((x + 4, y + tile_h + 3), label, fill=TEXT_COLOR, font=font)

            else:
                missing.append(str(p))
                draw.rectangle(
                    [x, y, x + tile_w - 1, y + tile_h + LABEL_H - 1],
                    fill=MISSING_COLOR,
                )
                draw.text((x + 4, y + 4), "MISSING", fill=MISSING_TEXT_COLOR, font=font)
                draw.text(
                    (x + 4, y + tile_h + 3),
                    label,
                    fill=MISSING_TEXT_COLOR,
                    font=font,
                )

    out_path = OUT_DIR / f"tile_lr_fr{frame_id:03d}.png"
    canvas.save(out_path)

    print(f"saved: {out_path}")
    if missing:
        print(f"missing {len(missing)} examples:")
        for p in missing:
            print("  ", p)

    return out_path


def make_gif_with_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. PNG frames were created, but GIF was skipped.")
        return

    input_pattern = str(OUT_DIR / "tile_lr_fr%03d.png")
    palette_path = str(OUT_DIR / "palette.png")

    # Palette pass: better colors, smaller GIF.
    cmd_palette = [
        "ffmpeg",
        "-y",
        "-framerate", str(GIF_FPS),
        "-start_number", "2",
        "-i", input_pattern,
        "-vf", f"scale={GIF_WIDTH}:-1:flags=lanczos,palettegen",
        palette_path,
    ]

    # GIF pass.
    cmd_gif = [
        "ffmpeg",
        "-y",
        "-framerate", str(GIF_FPS),
        "-start_number", "2",
        "-i", input_pattern,
        "-i", palette_path,
        "-lavfi", f"scale={GIF_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        str(GIF_PATH),
    ]

    print("making GIF palette...")
    subprocess.run(cmd_palette, check=True)

    print("making GIF...")
    subprocess.run(cmd_gif, check=True)

    print(f"saved GIF: {GIF_PATH}")

def crop_top_left(input_gif_path, output_gif_path, out_dir, crop_factor=4):
    palette_path = str(out_dir / "crop_palette.png")

    # Pass 1: Generate palette for cropped area
    cmd_palette = [
        "ffmpeg", "-y",
        "-i", str(input_gif_path),
        "-vf", f"crop=iw/{crop_factor}:ih/{crop_factor}:0:0,palettegen",
        "-update", "1",
        palette_path
    ]

    # Pass 2: Apply crop and palette
    cmd_gif = [
        "ffmpeg", "-y",
        "-i", str(input_gif_path),
        "-i", palette_path,
        "-lavfi", f"crop=iw/{crop_factor}:ih/{crop_factor}:0:0[x];[x][1:v]paletteuse",
        str(output_gif_path)
    ]

    import subprocess
    print("Generating crop palette...")
    subprocess.run(cmd_palette, check=True)
    print("Cropping GIF...")
    subprocess.run(cmd_gif, check=True)
    print(f"Successfully cropped top-left 1/{crop_factor} to: {output_gif_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if TILE_SIZE is None:
        tile_w, tile_h = infer_tile_size()
    else:
        tile_w, tile_h = TILE_SIZE

    print(f"tile size: {tile_w}x{tile_h}")
    print(f"grid: {GRID_W}x{GRID_H}")
    print(f"frames: {min(FRAME_IDS)}..{max(FRAME_IDS)}")

    font = load_font()

    for frame_id in FRAME_IDS:
        draw_one_frame(frame_id, tile_w, tile_h, font)

    make_gif_with_ffmpeg()

    crop_top_left(GIF_PATH, BASE_DIR / "tile_lr_fr002-037_hd_4x_crop4.gif", OUT_DIR,4)




if __name__ == "__main__":
    main()