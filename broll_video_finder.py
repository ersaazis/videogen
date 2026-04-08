import os
import json
import re
import requests
from openai import OpenAI
from moviepy import AudioFileClip

class BrollFinder:
    def __init__(self, api_key=None, pexels_api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.api_key = api_key
        self.pexels_api_key = pexels_api_key
        self.base_url = base_url
        self.model = model

    def search_pexels_videos(self, query):
        if not self.pexels_api_key: return []
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip() or "podcast"
        url = f"https://api.pexels.com/videos/search?query={clean_query}&per_page=10&orientation=portrait"
        headers = {"Authorization": self.pexels_api_key}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("videos", [])
        except: pass
        return []

    def download_video(self, url, dest_path):
        if os.path.exists(dest_path): return True
        try:
            res = requests.get(url, stream=True, timeout=30)
            if res.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=1024*1024):
                        if chunk: f.write(chunk)
                return True
        except: pass
        return False

    def get_broll_plan_from_llm(self, cc_data):
        if not self.api_key: return None
        full_text = " ".join([c["text"] for c in cc_data])
        total_duration = cc_data[-1]["end"] if cc_data else 0
        prompt = f"""
        You are a video editor. Plan the background B-roll segments for a vertical short video.
        The video is {total_duration} seconds long.
        
        Transcript:
        {full_text}
        
        TASK:
        Divide the video into dynamic B-roll segments (each around 5-30 seconds). 
        For each segment, provide a specific search keyword for Pexels.
        The segments must cover the entire {total_duration} seconds without gaps.
        
        Return ONLY a valid JSON list of objects:
        [
          {{"start": 0.0, "end": 5.0, "keyword": "rocket launch"}},
          ...
        ]
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            content = response.choices[0].message.content
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            plan = json.loads(content)
            if isinstance(plan, dict) and "segments" in plan: plan = plan["segments"]
            return plan
        except: return None

    def generate_broll_json(self, valid_items, output_dir, cc_data=None):
        if not cc_data:
            cc_path = os.path.join(output_dir, "cc.json")
            if not os.path.exists(cc_path): return []
            with open(cc_path, 'r') as f: cc_data = json.load(f)
            
        plan = self.get_broll_plan_from_llm(cc_data)
        if not plan:
            plan = []
            current_t = 0
            for item in valid_items:
                path = item.get("audio_path")
                with AudioFileClip(path) as clip: d = clip.duration
                plan.append({"start": current_t, "end": current_t + d, "keyword": item.get("message", "podcast")[:50]})
                current_t += d

        project_dir = os.path.dirname(output_dir)
        broll_video_dir = os.path.join(project_dir, "broll_videos")
        os.makedirs(broll_video_dir, exist_ok=True)

        final_broll = []
        for seg in plan:
            start_t, end_t, keyword = seg["start"], seg["end"], seg["keyword"]
            target_dur = end_t - start_t
            videos = self.search_pexels_videos(keyword)
            acc_dur = 0
            for v in videos:
                v_id, v_dur = v.get("id"), v.get("duration", 0)
                video_url = next((f.get("link") for f in v.get("video_files", []) if f.get("width") in [720, 1080]), None) or (v.get("video_files", [])[0].get("link") if v.get("video_files") else None)
                if video_url:
                    local_path = os.path.join(broll_video_dir, f"pexels_{v_id}.mp4")
                    if self.download_video(video_url, local_path):
                        use_dur = min(v_dur, target_dur - acc_dur)
                        final_broll.append({"segment_start": round(start_t + acc_dur, 2), "segment_end": round(start_t + acc_dur + use_dur, 2), "duration": round(use_dur, 2), "local_path": local_path, "pexels_id": v_id, "query": keyword})
                        acc_dur += use_dur
                if acc_dur >= target_dur: break
        
        broll_path = os.path.join(output_dir, "broll.json")
        with open(broll_path, "w") as f: json.dump(final_broll, f, indent=4)
        return final_broll
