"""
video.py — Podcast Video Generator (Refactored)
==============================================
Class-based rendering engine for vertical 9:16 podcast shorts.
Features: 
- Multi-layer rendering (B-roll, Overlay, Social handles, Avatar, Subtitles, Watermark)
- Dynamic frame building with PIL and MoviePy
- Automatic B-roll timeline synchronization
"""

import os
import re
import json
import math
import textwrap
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip, VideoFileClip, CompositeVideoClip, ColorClip
from moviepy.video.fx import FadeIn, FadeOut # For future use

# ─── DESIGN CONSTANTS ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1080, 1920
PADDING_TOP    = 240
PADDING_H      = 70
PADDING_BOTTOM = 390
BG_COLOR = (223, 41, 99) # #df2963

# Positioning
AVATAR_Y   = 490
SUBTITLE_Y = 1460 # Moved down to create gap from avatar bottom (~1430)
TRANSITION_DURATION = 0.4

# Typography
SUBTITLE_FONT_SIZE = 55
SUBTITLE_COLOR = (255, 255, 255)
SUBTITLE_BG_TED  = (53, 58, 71)
SUBTITLE_BG_EDDY = (49, 50, 34)
SUBTITLE_PADDING_X, SUBTITLE_PADDING_Y = 32, 24
SUBTITLE_BORDER_RADIUS = 18

# Social Media Overlay
SOCIAL_FONT_SIZE = 40
SOCIAL_ICON_SIZE = 40
SOCIAL_BAR_Y = 1640 # Final static position shifted up for better gap

# Watermark
WATERMARK_TEXT = "video provided by Pexels"
WATERMARK_FONT_SIZE = 25
WATERMARK_LEFT, WATERMARK_BOTTOM = 70, 70
WATERMARK_PAD_X, WATERMARK_PAD_Y = 14, 8

# Fonts
FONT_PATH_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ROBOTO_CANDIDATES = [
    "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/Roboto-Regular.ttf",
    "/usr/local/share/fonts/Roboto-Regular.ttf",
]

def find_best_font(size):
    for p in ROBOTO_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    if os.path.exists(FONT_PATH_REGULAR):
        return ImageFont.truetype(FONT_PATH_REGULAR, size)
    return ImageFont.load_default()

# ─── UTILITIES ───────────────────────────────────────────────────────────────

def wrap_text(text, font, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip() if current else word
        bbox = font.getbbox(test)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            if current: lines.append(current)
            current = word
    if current: lines.append(current)
    return lines

def draw_rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

# ─── RENDERING ENGINE ──────────────────────────────────────────────────────────

class VideoFrameBuilder:
    def __init__(self, cc_data, char_data, broll_data, social_data, total_duration, base_dir):
        self.cc_data = cc_data
        self.char_data = char_data
        self.social_data = social_data or {}
        self.total_duration = total_duration
        self.base_dir = base_dir
        
        # Fonts
        self.font_sub = find_best_font(SUBTITLE_FONT_SIZE)
        self.font_social = find_best_font(SOCIAL_FONT_SIZE)
        self.font_wm = find_best_font(WATERMARK_FONT_SIZE)
        
        self.content_width = WIDTH - 2 * PADDING_H
        self.avatar_size = self.content_width
        
        # Pre-load Assets
        self._load_broll_clips(broll_data)
        self._load_avatars()
        self._load_social_icons()
        self._watermark = self._make_watermark()
        self._social_overlay = self._make_social_overlay()
        self._solid_bg = Image.new("RGBA", (WIDTH, HEIGHT), (*BG_COLOR, 255))

    def _load_broll_clips(self, broll_data):
        from video_planning import VideoPlanner # Internal use for validation if needed
        # We reuse the validation logic here
        self.broll_timeline = self._validate_broll(broll_data)
        self._broll_clips = {}
        for seg in self.broll_timeline:
            path = seg["local_path"]
            if path not in self._broll_clips and os.path.exists(path):
                try:
                    self._broll_clips[path] = VideoFileClip(path)
                except: pass

    def _validate_broll(self, broll_data):
        if not broll_data: return []
        valid = [s for s in broll_data if os.path.exists(s["local_path"])]
        if not valid: return []
        valid.sort(key=lambda s: s["segment_start"])
        fixed, cursor = [], 0.0
        for seg in valid:
            if seg["segment_start"] > cursor + 0.01:
                filler = dict(fixed[-1] if fixed else seg)
                filler.update({"segment_start": cursor, "segment_end": seg["segment_start"]})
                fixed.append(filler)
            if seg["segment_start"] < cursor:
                seg["segment_start"] = cursor
            seg["duration"] = round(seg["segment_end"] - seg["segment_start"], 3)
            if seg["duration"] > 0:
                fixed.append(seg)
                cursor = seg["segment_end"]
        if cursor < self.total_duration:
             filler = dict(fixed[-1] if fixed else valid[0])
             filler.update({"segment_start": cursor, "segment_end": self.total_duration})
             fixed.append(filler)
        return fixed

    def _load_avatars(self):
        self._avatar_cache = {}
        for entry in self.char_data:
            path = os.path.join(self.base_dir, entry["image_path"])
            if path not in self._avatar_cache and os.path.exists(path):
                img = Image.open(path).convert("RGBA")
                self._avatar_cache[entry["image_path"]] = img.resize((self.avatar_size, self.avatar_size), Image.LANCZOS)

    def _load_social_icons(self):
        self._icons = {}
        icon_names = ["instagram", "youtube", "tiktok", "threads"]
        asset_dir = os.path.join(self.base_dir, "assets")
        for name in icon_names:
            path = os.path.join(asset_dir, f"{name}.png")
            if os.path.exists(path):
                img = Image.open(path).convert("RGBA")
                # Resize to SOCIAL_ICON_SIZE square
                self._icons[name] = img.resize((SOCIAL_ICON_SIZE, SOCIAL_ICON_SIZE), Image.LANCZOS)

    def _make_watermark(self):
        bbox = self.font_wm.getbbox(WATERMARK_TEXT)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img = Image.new("RGBA", (tw + 2*WATERMARK_PAD_X, th + 2*WATERMARK_PAD_Y), (0,0,0,180))
        draw = ImageDraw.Draw(img)
        draw.text((WATERMARK_PAD_X, WATERMARK_PAD_Y - bbox[1]), WATERMARK_TEXT, font=self.font_wm, fill=(255,255,255,230))
        return img

    def _make_social_overlay(self):
        """Creates a 2-column grid of social media icons + handles."""
        # Row-based mapping as requested: (Col 1, Col 2)
        grid_config = [
            [("youtube", self.social_data.get("youtube")), ("tiktok", self.social_data.get("tiktok"))],
            [("instagram", self.social_data.get("instagram")), ("threads", self.social_data.get("threads"))]
        ]
        
        # Prepare processed items
        rows = []
        for pair in grid_config:
            row_items = []
            for name, handle in pair:
                if handle and name in self._icons:
                    display_handle = handle if handle.startswith("@") else f"@{handle}"
                    row_items.append((self._icons[name], display_handle))
                else:
                    row_items.append(None)
            if any(row_items):
                rows.append(row_items)
        
        if not rows: return None
        
        # Grid parameters
        col_width = self.content_width // 2
        gap_inner = 12
        row_height = max(SOCIAL_ICON_SIZE, SOCIAL_FONT_SIZE) + 10
        total_h = len(rows) * row_height
        
        img = Image.new("RGBA", (self.content_width, total_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for r_idx, row in enumerate(rows):
            curr_y = r_idx * row_height
            for c_idx, item in enumerate(row):
                if not item: continue
                icon, text = item
                curr_x = c_idx * col_width
                
                # Icon
                img.paste(icon, (curr_x, curr_y), icon)
                
                # Text
                bbox = self.font_social.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                text_y = curr_y + (SOCIAL_ICON_SIZE - th) // 2 - bbox[1]
                draw.text((curr_x + icon.width + gap_inner, text_y), text, font=self.font_social, fill=(255, 255, 255, 200))
                
        return img

    def _get_broll_frame(self, t):
        seg = next((s for s in self.broll_timeline if s["segment_start"] <= t < s["segment_end"]), None)
        clip = self._broll_clips.get(seg["local_path"]) if seg else None
        if not clip: return self._solid_bg
        
        t_in_clip = (t - seg["segment_start"]) % clip.duration
        frame = Image.fromarray(clip.get_frame(t_in_clip)).convert("RGBA")
        
        # Fit to 1080x1920 with Center Crop
        orig_w, orig_h = frame.size
        scale = max(WIDTH / orig_w, HEIGHT / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame = frame.resize((new_w, new_h), Image.LANCZOS)
        left, top = (new_w - WIDTH)//2, (new_h - HEIGHT)//2
        return frame.crop((left, top, left+WIDTH, top+HEIGHT))

    def build_frame(self, t):
        # 1. B-roll (Bottom)
        canvas = self._get_broll_frame(t).copy()
        
        # 2. Black Overlay (Opacity 0.15)
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, int(255 * 0.20)))
        canvas = Image.alpha_composite(canvas, overlay)
        
        # 3. Social Media Layer (Static position below avatar chin area)
        if self._social_overlay:
            canvas.paste(self._social_overlay, (PADDING_H, SOCIAL_BAR_Y), self._social_overlay)
            
        # 4. Avatar
        char = next((c for c in self.char_data if c["start"] <= t < c["end"]), self.char_data[-1] if self.char_data else None)
        if char:
            avatar_img = self._avatar_cache.get(char["image_path"])
            if avatar_img:
                # Slide Logic
                elapsed, remaining = t - char["start"], char["end"] - t
                is_ted = char["speaker"].upper() == "TED"
                rest_x = PADDING_H if is_ted else (WIDTH - PADDING_H - self.avatar_size)
                origin = -self.avatar_size if is_ted else WIDTH
                
                x = rest_x
                if elapsed < TRANSITION_DURATION:
                    x = int(origin + (rest_x - origin) * (1 - (1 - elapsed/TRANSITION_DURATION)**3))
                elif remaining < TRANSITION_DURATION:
                    x = int(rest_x + (origin - rest_x) * ((1 - remaining/TRANSITION_DURATION)**3))
                canvas.paste(avatar_img, (x, AVATAR_Y), avatar_img)
        # 5. Subtitle
        cc = next((item for item in self.cc_data if item["start"] <= t <= item["end"]), None)
        if cc and char and cc["speaker"].upper() == char["speaker"].upper():
            sub_img = self._make_subtitle_image(cc["text"], cc["speaker"])
            canvas.paste(sub_img, (PADDING_H, SUBTITLE_Y), sub_img)
            
        # 6. Watermark (Top)
        wm_y = HEIGHT - WATERMARK_BOTTOM - self._watermark.height
        canvas.paste(self._watermark, (WATERMARK_LEFT, wm_y), self._watermark)
        
        return np.array(canvas.convert("RGB"))

    def _make_subtitle_image(self, text, speaker):
        is_ted = speaker.upper() == "TED"
        bg_color = SUBTITLE_BG_TED if is_ted else SUBTITLE_BG_EDDY
        lines = wrap_text(text.upper(), self.font_sub, self.content_width - 2*SUBTITLE_PADDING_X)
        line_h = self.font_sub.getbbox("Ag")[3] - self.font_sub.getbbox("Ag")[1] + 12
        box_h = line_h * len(lines) + 2*SUBTITLE_PADDING_Y
        img = Image.new("RGBA", (self.content_width, box_h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw_rounded_rect(draw, (0, 0, self.content_width, box_h), SUBTITLE_BORDER_RADIUS, fill=bg_color)
        for i, line in enumerate(lines):
            lw = self.font_sub.getbbox(line)[2] - self.font_sub.getbbox(line)[0]
            x = SUBTITLE_PADDING_X if is_ted else (self.content_width - SUBTITLE_PADDING_X - lw)
            draw.text((x, SUBTITLE_PADDING_Y + i*line_h), line, font=self.font_sub, fill=SUBTITLE_COLOR)
        return img

# ─── PUBLIC RENDERER CLASS ──────────────────────────────────────────────────

class VideoRenderer:
    def __init__(self, project_id, base_dir=None):
        self.project_id = project_id
        self.base_dir = base_dir or os.getcwd()
        self.safe_pid = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
        self.project_dir = os.path.join(self.base_dir, "projects", self.safe_pid)
        self.output_dir = os.path.join(self.project_dir, "output")
        self.temp_dir = os.path.join(self.project_dir, "temp")
        
        # Load necessary data
        self.project_json = self._load_json("project.json")
        self.cc_data = self._load_json("cc.json")
        self.char_data = self._load_json("character.json")
        self.broll_data = self._load_json("broll.json") or []
        
        # Paths
        self.audio_path = self._find_audio()
        self.output_path = os.path.join(self.output_dir, f"{self.safe_pid}_final_video.mp4")

    def _load_json(self, name):
        for d in [self.output_dir, self.temp_dir, self.project_dir]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                with open(p, "r") as f: return json.load(f)
        return None

    def _find_audio(self):
        name = f"{self.safe_pid}_joined.mp3"
        for d in [self.output_dir, self.temp_dir]:
            p = os.path.join(d, name)
            if os.path.exists(p): return p
        return None

    def render(self, fps=30, progress_callback=None):
        """Executes the full rendering pipeline."""
        if not self.audio_path or not self.cc_data or not self.char_data:
            raise FileNotFoundError("Master audio, cc.json, or character.json missing.")
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        with AudioFileClip(self.audio_path) as audio:
            duration = audio.duration
            builder = VideoFrameBuilder(
                self.cc_data, self.char_data, self.broll_data, 
                self.project_json.get("social_media"), duration, self.base_dir
            )
            
            video = VideoClip(builder.build_frame, duration=duration)
            video = video.with_audio(audio)
            
            video.write_videofile(
                self.output_path,
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset="fast",
                ffmpeg_params=["-crf", "18"],
                logger="bar"
            )
            
            # Cleanup
            video.close()
            for clip in builder._broll_clips.values(): clip.close()
            
        print(f"✅ Video created: {self.output_path}")
        return self.output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    renderer = VideoRenderer(args.project)
    renderer.render(fps=args.fps)
