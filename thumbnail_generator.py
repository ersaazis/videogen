import os
import json
import re
import requests
import random
import uuid
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# Reuse design constants if possible, or define thumbnail specific ones
WIDTH, HEIGHT = 1080, 1920

# Fonts (Same as video.py)
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ROBOTO_CANDIDATES = [
    "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/Roboto-Regular.ttf",
    "/usr/local/share/fonts/Roboto-Regular.ttf",
]

def find_best_font(size, bold=False):
    if bold and os.path.exists(FONT_PATH_BOLD):
        return ImageFont.truetype(FONT_PATH_BOLD, size)
    for p in ROBOTO_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    if os.path.exists(FONT_PATH_REGULAR):
        return ImageFont.truetype(FONT_PATH_REGULAR, size)
    return ImageFont.load_default()

class ThumbnailGenerator:
    def __init__(self, api_key=None, pexels_api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.api_key = api_key
        self.pexels_api_key = pexels_api_key
        self.base_url = base_url
        self.model = model

    def get_thumbnail_plan(self, chat_data, title):
        if not self.api_key: return None
        full_text = " ".join([c.get("message", "") for c in chat_data[:20]]) # First 20 lines for context
        
        prompt = f"""
        You are a YouTube thumbnail designer. Plan a viral vertical thumbnail for this podcast.
        Title: {title}
        Script excerpt: {full_text}
        
        TASK:
        1. Provide a specific search keyword for a Pexels background photo (e.g., "dark city alley", "neon office", "serene mountain").
        2. Pick the best facial expressions for Ted and Eddy from this list: afraid, angry, disgusted, happy, nauseated, normal, sad, surprised.
        
        Return ONLY a JSON object:
        {{
          "keyword": "background keyword",
          "ted_expression": "expression",
          "eddy_expression": "expression"
        }}
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            content = response.choices[0].message.content
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except: return {"keyword": "podcast studio", "ted_expression": "happy", "eddy_expression": "surprised"}

    def search_pexels_photo(self, query):
        if not self.pexels_api_key: return None
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip() or "abstract"
        url = f"https://api.pexels.com/v1/search?query={clean_query}&per_page=5&orientation=portrait"
        headers = {"Authorization": self.pexels_api_key}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                photos = response.json().get("photos", [])
                if photos:
                    # Prefer large or original
                    return photos[0]["src"]["large2x"]
        except: pass
        return None

    def generate(self, project_id, plan, social_data, title, output_dir):
        # 1. Background
        bg_url = self.search_pexels_photo(plan.get("keyword", "podcast"))
        bg_image = None
        if bg_url:
            try:
                res = requests.get(bg_url, timeout=20)
                from io import BytesIO
                bg_image = Image.open(BytesIO(res.content)).convert("RGBA")
            except: pass
        
        if not bg_image:
            bg_image = Image.new("RGBA", (WIDTH, HEIGHT), (31, 41, 55, 255)) # Dark gray default
        
        # Fit bg
        orig_w, orig_h = bg_image.size
        scale = max(WIDTH / orig_w, HEIGHT / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        bg_image = bg_image.resize((new_w, new_h), Image.LANCZOS)
        left, top = (new_w - WIDTH)//2, (new_h - HEIGHT)//2
        canvas = bg_image.crop((left, top, left+WIDTH, top+HEIGHT))
        
        # 2. Dark Overlay
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 80))
        canvas = Image.alpha_composite(canvas, overlay)
        
        # 3. Avatars
        # Ted (Left)
        ted_expr = plan.get("ted_expression", "happy")
        ted_var = random.randint(0, 3)
        ted_path = os.path.join(os.getcwd(), "avatar", "ted", f"{ted_expr}{ted_var}.png")
        if os.path.exists(ted_path):
            ted_img = Image.open(ted_path).convert("RGBA")
            # Resize avatar to be consistent with video (Width - 2*H_PADDING = 1080 - 140 = 940?)
            # Actually let's make them slightly smaller to fit both
            avatar_size = 800
            ted_img = ted_img.resize((avatar_size, avatar_size), Image.LANCZOS)
            canvas.paste(ted_img, (-150, 600), ted_img)
            
        # Eddy (Right)
        eddy_expr = plan.get("eddy_expression", "surprised")
        eddy_var = random.randint(0, 3)
        eddy_path = os.path.join(os.getcwd(), "avatar", "eddy", f"{eddy_expr}{eddy_var}.png")
        if os.path.exists(eddy_path):
            eddy_img = Image.open(eddy_path).convert("RGBA")
            avatar_size = 800
            eddy_img = eddy_img.resize((avatar_size, avatar_size), Image.LANCZOS)
            canvas.paste(eddy_img, (WIDTH - avatar_size + 150, 600), eddy_img)
            
        # 4. Text Overlays
        draw = ImageDraw.Draw(canvas)
        
        # Title (Centered, Top-ish)
        title_font = find_best_font(110, bold=True)
        max_title_w = WIDTH - 100
        from video import wrap_text # Reuse if possible or redefine
        def local_wrap(text, font, max_w):
            words = text.split()
            lines, current = [], ""
            for word in words:
                test = f"{current} {word}".strip() if current else word
                bbox = font.getbbox(test)
                if (bbox[2] - bbox[0]) <= max_w: current = test
                else:
                    if current: lines.append(current)
                    current = word
            if current: lines.append(current)
            return lines

        title_lines = local_wrap(title.upper(), title_font, max_title_w)
        current_y = 250
        for line in title_lines:
            bbox = title_font.getbbox(line)
            lw = bbox[2] - bbox[0]
            # Draw shadow
            draw.text(((WIDTH - lw)//2 + 5, current_y + 5), line, font=title_font, fill=(0, 0, 0, 255))
            # Draw text
            draw.text(((WIDTH - lw)//2, current_y), line, font=title_font, fill=(255, 255, 255, 255))
            current_y += (bbox[3] - bbox[1]) + 20
            
        # Credits (Bottom)
        info_font = find_best_font(45)
        credit_text = "Produced by Ted & Eddy"
        
        bbox_cre = info_font.getbbox(credit_text)
        draw.text(((WIDTH - (bbox_cre[2]-bbox_cre[0]))//2, HEIGHT - 150), credit_text, font=info_font, fill=(255, 255, 255, 180))
        
        # 5. Save
        out_path = os.path.join(output_dir, "thumbnail.jpg")
        canvas.convert("RGB").save(out_path, "JPEG", quality=95)
        return out_path
