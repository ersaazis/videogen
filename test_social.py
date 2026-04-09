import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── DESIGN CONSTANTS (Sync with video.py) ───────────────────────────────────
WIDTH, HEIGHT = 1080, 1920
PADDING_TOP    = 240
PADDING_H      = 70
BG_COLOR = (223, 41, 99) # #df2963

AVATAR_Y   = 490
SUBTITLE_Y = 1460 
SOCIAL_BAR_Y = 1610 

SUBTITLE_FONT_SIZE = 50
SOCIAL_FONT_SIZE = 45
SOCIAL_ICON_SIZE = 45
WATERMARK_FONT_SIZE = 25
WATERMARK_TEXT = "video provided by Pexels"
WATERMARK_BOTTOM = 70
WATERMARK_LEFT = 70

# ─── UTILS ───────────────────────────────────────────────────────────────────
def find_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

# ─── PREVIEW BUILDER ──────────────────────────────────────────────────────────
class PreviewBuilder:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.font_sub = find_font(SUBTITLE_FONT_SIZE)
        self.font_social = find_font(SOCIAL_FONT_SIZE)
        self.font_wm = find_font(WATERMARK_FONT_SIZE)
        self.content_width = WIDTH - 2 * PADDING_H
        
        # Mock Data
        self.social_data = {
            "youtube": "TedEddyShow",
            "tiktok": "ted_eddy_x",
            "instagram": "ted_eddy_insta",
            "threads": "ted_threads"
        }
        
    def build_preview(self, output_path="preview_layout.png"):
        # 1. Background
        canvas = Image.new("RGBA", (WIDTH, HEIGHT), (*BG_COLOR, 255))
        draw = ImageDraw.Draw(canvas)
        
        # 2. Avatar Placeholder (Mock)
        avatar_size = self.content_width
        draw.rectangle([PADDING_H, AVATAR_Y, PADDING_H + avatar_size, AVATAR_Y + avatar_size], 
                       fill=(255, 255, 255, 40), outline=(255, 255, 255, 100), width=5)
        draw.text((PADDING_H + 20, AVATAR_Y + 20), "AVATAR POSITION", font=self.font_sub, fill=(255, 255, 255, 150))
        
        # 3. Subtitle Placeholder
        sub_text = "PREVIEW OF THE SUBTITLE"
        sub_bg = (53, 58, 71)
        sub_pad_x, sub_pad_y = 32, 24
        bbox = self.font_sub.getbbox(sub_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_w = tw + 2 * sub_pad_x
        box_h = th + 2 * sub_pad_y
        draw_rounded_rect(draw, (PADDING_H, SUBTITLE_Y, PADDING_H + box_w, SUBTITLE_Y + box_h), 18, fill=sub_bg)
        draw.text((PADDING_H + sub_pad_x, SUBTITLE_Y + sub_pad_y - bbox[1]), sub_text, font=self.font_sub, fill=(255, 255, 255))

        # 4. Social Media Overlay (Mock Generation)
        self._load_icons()
        social_overlay = self._make_social_overlay()
        if social_overlay:
            canvas.paste(social_overlay, (PADDING_H, SOCIAL_BAR_Y), social_overlay)

        # 5. Watermark / Credit
        bbox_wm = self.font_wm.getbbox(WATERMARK_TEXT)
        tw_wm, th_wm = bbox_wm[2] - bbox_wm[0], bbox_wm[3] - bbox_wm[1]
        draw.rectangle([WATERMARK_LEFT, HEIGHT - WATERMARK_BOTTOM - th_wm - 20, WATERMARK_LEFT + tw_wm + 40, HEIGHT - WATERMARK_BOTTOM], fill=(0,0,0,100))
        draw.text((WATERMARK_LEFT + 20, HEIGHT - WATERMARK_BOTTOM - th_wm - 10 - bbox_wm[1]), WATERMARK_TEXT, font=self.font_wm, fill=(255, 255, 255, 180))

        canvas.save(output_path)
        print(f"✅ Preview saved to: {output_path}")

    def _load_icons(self):
        self._icons = {}
        for name in ["instagram", "youtube", "tiktok", "threads"]:
            path = os.path.join(self.base_dir, "assets", f"{name}.png")
            if os.path.exists(path):
                img = Image.open(path).convert("RGBA")
                self._icons[name] = img.resize((SOCIAL_ICON_SIZE, SOCIAL_ICON_SIZE), Image.LANCZOS)
            else:
                # Create placeholder icon if missing
                self._icons[name] = Image.new("RGBA", (SOCIAL_ICON_SIZE, SOCIAL_ICON_SIZE), (255, 255, 255, 100))

    def _make_social_overlay(self):
        grid_config = [
            [("youtube", self.social_data.get("youtube")), ("tiktok", self.social_data.get("tiktok"))],
            [("instagram", self.social_data.get("instagram")), ("threads", self.social_data.get("threads"))]
        ]
        col_width = self.content_width // 2
        row_height = max(SOCIAL_ICON_SIZE, SOCIAL_FONT_SIZE) + 10
        img = Image.new("RGBA", (self.content_width, row_height * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for r_idx, row in enumerate(grid_config):
            for c_idx, (name, handle) in enumerate(row):
                curr_x = c_idx * col_width
                curr_y = r_idx * row_height
                icon = self._icons.get(name)
                img.paste(icon, (curr_x, curr_y), icon)
                draw.text((curr_x + SOCIAL_ICON_SIZE + 12, curr_y + 5), handle, font=self.font_social, fill=(255, 255, 255, 200))
        return img

if __name__ == "__main__":
    PreviewBuilder().build_preview()
