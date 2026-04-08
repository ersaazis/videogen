"""
video.py — Podcast Video Generator
====================================
Resolusi  : 1080 × 1920 px  (9:16 vertikal)
Background : #df2963
Padding   : top=240px, left/right=70px, bottom=390px
Layout (bawah ke atas):
  - Subtitle paling bawah (anchor di PADDING_BOTTOM)
  - Avatar langsung di atas subtitle

Speaker TED   → slide in dari KIRI, slide out ke KIRI,  teks rata KIRI,  bg subtitle #353a47
Speaker EDDY  → slide in dari KANAN, slide out ke KANAN, teks rata KANAN, bg subtitle #313222

Subtitle : Roboto 55px, putih, Capitalize
"""

import os
import json
import math
import textwrap
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoClip,
    AudioFileClip,
    VideoFileClip,
    CompositeVideoClip,
    ColorClip,
)

# ─── KONSTANTA DESAIN ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1080, 1920
PADDING_TOP    = 240   # px — atas
PADDING_H      = 70    # px — kiri & kanan
PADDING_BOTTOM = 390   # px — bawah
BG_COLOR = (223, 41, 99)     # #df2963

# Avatar width = 100% dari content_width (dihitung dinamis di VideoFrameBuilder)

# Posisi vertikal FIXED (dari top canvas)
AVATAR_Y   = 490    # px — top avatar dari atas canvas
SUBTITLE_Y = 1385   # px — top subtitle dari atas canvas

# Efek transisi avatar saat berganti speaker
TRANSITION_DURATION = 0.4   # detik durasi slide+fade in

SUBTITLE_FONT_SIZE = 55
SUBTITLE_COLOR = (255, 255, 255)
SUBTITLE_BG_TED  = (53, 58, 71)    # #353a47
SUBTITLE_BG_EDDY = (49, 50, 34)    # #313222
SUBTITLE_PADDING_X = 32           # padding dalam kotak subtitle
SUBTITLE_PADDING_Y = 24
SUBTITLE_BORDER_RADIUS = 18

# Watermark
WATERMARK_TEXT      = "Broll provided by Pexels"
WATERMARK_FONT_SIZE = 25
WATERMARK_LEFT      = 70    # px dari kiri canvas (tidak ikut padding)
WATERMARK_BOTTOM    = 70    # px dari bawah canvas
WATERMARK_PAD_X     = 14   # padding horizontal dalam kotak
WATERMARK_PAD_Y     = 8    # padding vertikal dalam kotak

FONT_PATH_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Fallback: cari Roboto jika tersedia
_ROBOTO_CANDIDATES = [
    "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/Roboto-Regular.ttf",
    "/usr/local/share/fonts/Roboto-Regular.ttf",
    os.path.expanduser("~/.fonts/Roboto-Regular.ttf"),
    os.path.expanduser("~/.local/share/fonts/Roboto-Regular.ttf"),
]

def _find_font():
    for p in _ROBOTO_CANDIDATES:
        if os.path.exists(p):
            return p
    # fallback ke DejaVu
    if os.path.exists(FONT_PATH_REGULAR):
        return FONT_PATH_REGULAR
    return None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── BROLL VALIDATION ────────────────────────────────────────────────────────

def validate_and_fix_broll(broll_data: list, total_duration: float) -> list:
    """
    Validasi broll_data dan isi gap secara otomatis.
    Strategi improvisasi:
      - Gap di awal / tengah  → extend segment sebelumnya (loop)
      - Gap di akhir          → extend segment terakhir
      - File tidak ada        → skip dan tandai sebagai gap, lalu improvisasi
      - Overlap               → potong segment sebelumnya
    Return list bersih tanpa gap, diurutkan by segment_start.
    """
    if not broll_data:
        print("⚠️  broll.json kosong, background akan solid.")
        return []

    # 1. Filter file yang tidak ada
    valid = []
    for seg in broll_data:
        if os.path.exists(seg["local_path"]):
            valid.append(dict(seg))
        else:
            print(f"   ⚠️  File tidak ditemukan, di-skip: {seg['local_path']}")

    if not valid:
        print("⚠️  Semua file broll tidak ditemukan.")
        return []

    # 2. Sort by start
    valid.sort(key=lambda s: s["segment_start"])

    fixed = []
    cursor = 0.0

    for seg in valid:
        seg_start = seg["segment_start"]
        seg_end   = seg["segment_end"]

        # Gap sebelum segment ini?
        if seg_start > cursor + 0.01:
            gap_start = cursor
            gap_end   = seg_start
            print(f"   🔧 Gap {gap_start:.2f}s–{gap_end:.2f}s → improvisasi dari segment sebelumnya")
            if fixed:
                # Loop segment terakhir untuk isi gap
                filler = dict(fixed[-1])
                filler["segment_start"] = gap_start
                filler["segment_end"]   = gap_end
                filler["duration"]      = round(gap_end - gap_start, 3)
                fixed.append(filler)
            else:
                # Tidak ada segment sebelumnya: pakai segment ini sebagai filler
                filler = dict(seg)
                filler["segment_start"] = gap_start
                filler["segment_end"]   = gap_end
                filler["duration"]      = round(gap_end - gap_start, 3)
                fixed.append(filler)

        # Overlap: potong mulai cursor
        if seg_start < cursor:
            seg["segment_start"] = cursor
            seg["duration"] = round(seg_end - cursor, 3)
            if seg["duration"] <= 0:
                continue

        seg["segment_start"] = round(seg["segment_start"], 3)
        seg["segment_end"]   = round(seg_end, 3)
        seg["duration"]      = round(seg_end - seg["segment_start"], 3)
        fixed.append(dict(seg))
        cursor = seg_end

    # 3. Gap di akhir
    if cursor < total_duration - 0.01:
        print(f"   🔧 Gap akhir {cursor:.2f}s–{total_duration:.2f}s → loop segment terakhir")
        if fixed:
            filler = dict(fixed[-1])
            filler["segment_start"] = cursor
            filler["segment_end"]   = total_duration
            filler["duration"]      = round(total_duration - cursor, 3)
            fixed.append(filler)

    print(f"✅ Broll validated: {len(fixed)} segmen, coverage {fixed[-1]['segment_end']:.2f}s / {total_duration:.2f}s")
    return fixed


# ─── UTILITY ─────────────────────────────────────────────────────────────────

def load_avatar(image_path: str, size: int) -> Image.Image:
    """Buka avatar PNG (with transparency) dan resize ke size×size (square)."""
    full_path = os.path.join(BASE_DIR, image_path)
    img = Image.open(full_path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Pecah teks agar muat dalam max_width piksel."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip() if current else word
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_subtitle_image(
    text: str,
    speaker: str,
    font: ImageFont.FreeTypeFont,
    content_width: int,       # lebar area konten (inside padding)
) -> Image.Image:
    """
    Buat gambar kotak subtitle sesuai speaker.
    Lebar = content_width (full width konten), tinggi dinamis.
    """
    is_ted = speaker.upper() == "TED"
    bg_color = SUBTITLE_BG_TED if is_ted else SUBTITLE_BG_EDDY
    text_align = "left" if is_ted else "right"

    # Judul format: UPPERCASE untuk tampilan modern
    display_text = text.upper()
    lines = wrap_text(display_text, font, content_width - 2 * SUBTITLE_PADDING_X)

    # Hitung dimensi teks
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 12
    text_block_h = line_height * len(lines)
    box_w = content_width
    box_h = text_block_h + 2 * SUBTITLE_PADDING_Y

    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw_rounded_rect(draw, (0, 0, box_w, box_h), SUBTITLE_BORDER_RADIUS, fill=bg_color)

    text_area_w = box_w - 2 * SUBTITLE_PADDING_X

    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        y = SUBTITLE_PADDING_Y + i * line_height

        if text_align == "left":
            x = SUBTITLE_PADDING_X
        else:
            x = SUBTITLE_PADDING_X + (text_area_w - line_w)

        draw.text((x, y), line, font=font, fill=SUBTITLE_COLOR)

    return img


# ─── FRAME BUILDER ───────────────────────────────────────────────────────────

class VideoFrameBuilder:
    def __init__(self, cc_data: list, char_data: list, broll_data: list,
                 font_path: str | None, total_duration: float):
        self.cc_data   = cc_data
        self.char_data = char_data
        self.font = ImageFont.truetype(font_path, SUBTITLE_FONT_SIZE) if font_path else ImageFont.load_default()
        self.content_width = WIDTH - 2 * PADDING_H    # 1080 - 140 = 940
        self.avatar_size = 650

        # ── Broll: validasi & load clips ──────────────────────────────────────
        print("🎥 Validating broll timeline...")
        self.broll_data = validate_and_fix_broll(broll_data, total_duration)
        self._broll_clips: dict[str, VideoFileClip] = {}
        for seg in self.broll_data:
            p = seg["local_path"]
            if p not in self._broll_clips:
                try:
                    self._broll_clips[p] = VideoFileClip(p)
                    print(f"   ✅ Loaded broll: {os.path.basename(p)}")
                except Exception as e:
                    print(f"   ⚠️  Gagal load broll {p}: {e}")

        # Pre-load semua avatar yang dibutuhkan
        print("🖼️  Pre-loading avatars...")
        print(f"   Avatar size: {self.avatar_size}×{self.avatar_size} px")
        self._avatar_cache: dict[str, Image.Image] = {}
        for entry in char_data:
            ip = entry["image_path"]
            if ip not in self._avatar_cache:
                try:
                    self._avatar_cache[ip] = load_avatar(ip, self.avatar_size)
                    print(f"   ✅ {ip}")
                except Exception as e:
                    print(f"   ⚠️  Gagal load {ip}: {e}")

        # Base canvas solid (fallback jika broll tidak ada)
        self._solid_bg = Image.new("RGBA", (WIDTH, HEIGHT), (*BG_COLOR, 255))

        # Pre-render watermark
        self._watermark = self._make_watermark(font_path)

    def _get_broll_seg_at(self, t: float) -> dict | None:
        """Cari segmen broll yang aktif pada waktu t."""
        for seg in self.broll_data:
            if seg["segment_start"] <= t < seg["segment_end"]:
                return seg
        return None

    def _get_broll_frame(self, t: float) -> Image.Image:
        """
        Ambil frame dari video broll pada waktu global t.
        Clip ke ukuran 1080×1920 (crop center jika perlu).
        Loop video jika t_dalam_clip > durasi video.
        """
        seg = self._get_broll_seg_at(t)
        if not seg:
            return self._solid_bg

        clip = self._broll_clips.get(seg["local_path"])
        if not clip:
            return self._solid_bg

        # Waktu dalam clip (loop jika clip lebih pendek dari segment)
        offset = t - seg["segment_start"]
        clip_dur = clip.duration
        t_in_clip = offset % clip_dur  # loop

        # Extract frame
        frame_np = clip.get_frame(t_in_clip)  # RGB numpy (H, W, 3)
        frame_img = Image.fromarray(frame_np).convert("RGBA")

        # Resize + crop center ke 1080×1920
        orig_w, orig_h = frame_img.size
        scale = max(WIDTH / orig_w, HEIGHT / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        frame_img = frame_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - WIDTH)  // 2
        top  = (new_h - HEIGHT) // 2
        frame_img = frame_img.crop((left, top, left + WIDTH, top + HEIGHT))

        return frame_img

    def _make_watermark(self, font_path: str | None) -> Image.Image:
        """Buat kotak watermark hitam dengan teks putih."""
        wm_font = (
            ImageFont.truetype(font_path, WATERMARK_FONT_SIZE)
            if font_path else ImageFont.load_default()
        )
        bbox = wm_font.getbbox(WATERMARK_TEXT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        box_w = tw + 2 * WATERMARK_PAD_X
        box_h = th + 2 * WATERMARK_PAD_Y
        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 200))  # hitam semi-transparan
        draw = ImageDraw.Draw(img)
        draw.text((WATERMARK_PAD_X, WATERMARK_PAD_Y - bbox[1]), WATERMARK_TEXT,
                  font=wm_font, fill=(255, 255, 255, 255))
        return img

    def _get_char_at(self, t: float) -> dict | None:
        """Cari karakter yang aktif pada waktu t."""
        for entry in self.char_data:
            if entry["start"] <= t < entry["end"]:
                return entry
        # fallback: ambil entry terakhir kalau t >= end terakhir
        if self.char_data and t >= self.char_data[-1]["end"]:
            return self.char_data[-1]
        return None

    def _get_cc_at(self, t: float) -> dict | None:
        """Cari caption yang aktif pada waktu t."""
        for item in self.cc_data:
            if item["start"] <= t <= item["end"]:
                return item
        return None

    @staticmethod
    def _ease_out_cubic(x: float) -> float:
        """Ease-out-cubic: cepat di awal, melambat di akhir (untuk slide IN)."""
        return 1.0 - (1.0 - x) ** 3

    @staticmethod
    def _ease_in_cubic(x: float) -> float:
        """Ease-in-cubic: lambat di awal, cepat di akhir (untuk slide OUT)."""
        return x ** 3

    def _calc_avatar_x(self, speaker: str, char: dict, t: float) -> int:
        """
        Hitung posisi X avatar berdasarkan slide-in / slide-out horizontal.

        TED  → slide IN dari kiri  (x: -avatar_size → content_left)
               slide OUT ke kiri   (x: content_left → -avatar_size)
        EDDY → slide IN dari kanan (x: WIDTH        → content_left)
               slide OUT ke kanan  (x: content_left → WIDTH)
        """
        is_ted     = (speaker == "TED")
        rest_x_left  = PADDING_H
        rest_x_right = WIDTH - PADDING_H - self.avatar_size
        
        rest_x     = rest_x_left if is_ted else rest_x_right
        off_left   = -self.avatar_size                  # off-screen kiri
        off_right  = WIDTH                              # off-screen kanan
        
        origin     = off_left  if is_ted else off_right # asal slide-in
        dest       = off_left  if is_ted else off_right # tujuan slide-out

        elapsed   = t - char["start"]
        remaining = char["end"] - t

        # Slide IN
        if elapsed < TRANSITION_DURATION:
            progress = elapsed / TRANSITION_DURATION
            eased    = self._ease_out_cubic(progress)
            return int(origin + (rest_x - origin) * eased)

        # Slide OUT
        if remaining < TRANSITION_DURATION and remaining >= 0:
            progress = 1.0 - (remaining / TRANSITION_DURATION)
            eased    = self._ease_in_cubic(progress)
            return int(rest_x + (dest - rest_x) * eased)

        # Diam
        return rest_x

    def build_frame(self, t: float) -> "np.ndarray":
        """Render satu frame pada waktu t (detik). Return numpy array RGB."""

        # ── Background: broll frame (atau solid fallback) ──────────────────────
        canvas = self._get_broll_frame(t).copy()

        # Semi-transparent dark overlay agar konten tetap mudah dibaca
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 90))
        canvas = Image.alpha_composite(canvas, overlay)

        draw = ImageDraw.Draw(canvas)

        char = self._get_char_at(t)
        cc   = self._get_cc_at(t)

        if char:
            speaker    = char["speaker"].upper()
            avatar_img = self._avatar_cache.get(char["image_path"])

            # Posisi FIXED — tidak berubah walaupun subtitle ada/tidak
            avatar_y = AVATAR_Y    # 490px dari top
            sub_y    = SUBTITLE_Y  # 1385px dari top

            # --- Posisi X avatar (horizontal slide in/out) ---
            if avatar_img:
                avatar_x = self._calc_avatar_x(speaker, char, t)

                clip_left  = max(avatar_x, 0)
                clip_right = min(avatar_x + self.avatar_size, WIDTH)
                if clip_right > clip_left:
                    src_left  = clip_left - avatar_x
                    src_right = src_left + (clip_right - clip_left)
                    cropped   = avatar_img.crop((src_left, 0, src_right, self.avatar_size))
                    canvas.paste(cropped, (clip_left, avatar_y), cropped)

            # --- Subtitle (posisi fixed di sub_y) ---
            if cc and cc.get("speaker", "").upper() == speaker:
                sub_img = make_subtitle_image(
                    cc["text"], speaker, self.font, self.content_width
                )
                canvas.paste(sub_img, (PADDING_H, sub_y), sub_img)

        # --- Watermark: selalu di pojok kiri bawah ---
        wm = self._watermark
        wm_x = WATERMARK_LEFT
        wm_y = HEIGHT - WATERMARK_BOTTOM - wm.height
        canvas.paste(wm, (wm_x, wm_y), wm)

        # Convert ke RGB numpy array
        frame_np = np.array(canvas.convert("RGB"))
        return frame_np


# ─── MAIN ────────────────────────────────────────────────────────────────────

def build_video(
    cc_path: str,
    char_path: str,
    audio_path: str,
    output_path: str,
    broll_path: str | None = None,
    fps: int = 30,
):
    print("🎬 Memulai proses render video...")

    # Validasi file input
    missing = []
    for label, path in [("cc.json", cc_path), ("character.json", char_path), ("audio", audio_path)]:
        if not os.path.exists(path):
            missing.append(f"  ❌ {label}: {path}")
    if missing:
        print("\n⛔  File berikut tidak ditemukan:")
        for m in missing:
            print(m)
        print("\n💡 Jalankan dulu: python3 automate.py")
        print("   (untuk generate cc.json, character.json, dan audio)")
        return

    # Load data
    with open(cc_path, "r") as f:
        cc_data = json.load(f)
    with open(char_path, "r") as f:
        char_data = json.load(f)

    # Load broll (opsional)
    broll_data = []
    if broll_path and os.path.exists(broll_path):
        with open(broll_path, "r") as f:
            broll_data = json.load(f)
        print(f"📹 Loaded {len(broll_data)} broll segments dari {broll_path}")
    else:
        print("⚠️  Tidak ada broll.json, background akan solid.")

    # Durasi video = durasi audio
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    print(f"⏱️  Durasi: {duration:.2f} detik")

    font_path = _find_font()
    if font_path:
        print(f"🔤 Font: {font_path}")
    else:
        print("⚠️  Font Roboto tidak ditemukan, menggunakan default PIL")

    builder = VideoFrameBuilder(cc_data, char_data, broll_data, font_path, duration)

    print(f"🖥️  Rendering {int(duration * fps)} frames pada {fps} FPS...")
    video_clip = VideoClip(builder.build_frame, duration=duration)
    video_clip = video_clip.with_audio(audio_clip)

    video_clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        ffmpeg_params=["-crf", "18"],
        logger="bar",
    )

    audio_clip.close()
    video_clip.close()
    # Tutup semua broll clips
    for clip in builder._broll_clips.values():
        clip.close()
    print(f"\n✅ Video berhasil dibuat: {output_path}")


def _find_audio(temp_dir: str) -> str:
    """Auto-detect file *_joined.mp3 terbaru di temp_dir."""
    import glob
    candidates = glob.glob(os.path.join(temp_dir, "*_joined.mp3"))
    if candidates:
        # Ambil yang paling baru
        return max(candidates, key=os.path.getmtime)
    # Fallback lama
    return os.path.join(temp_dir, "test1_joined.mp3")


def parse_args():
    _temp = os.path.join(BASE_DIR, "temp")
    _audio_default = _find_audio(_temp)

    parser = argparse.ArgumentParser(
        description="🎬 Podcast Video Generator (9:16, 1080×1920)"
    )
    parser.add_argument(
        "--cc",
        default=os.path.join(_temp, "cc.json"),
        help="Path ke cc.json (default: temp/cc.json)",
    )
    parser.add_argument(
        "--char",
        default=os.path.join(_temp, "character.json"),
        help="Path ke character.json (default: temp/character.json)",
    )
    parser.add_argument(
        "--audio",
        default=_audio_default,
        help=f"Path ke file audio (auto-detected: {os.path.basename(_audio_default)})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_temp, "output_video.mp4"),
        help="Path output video (default: temp/output_video.mp4)",
    )
    parser.add_argument(
        "--broll",
        default=os.path.join(_temp, "broll.json"),
        help="Path ke broll.json (default: temp/broll.json)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame per second (default: 30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_video(
        cc_path=args.cc,
        char_path=args.char,
        audio_path=args.audio,
        output_path=args.output,
        broll_path=args.broll,
        fps=args.fps,
    )
