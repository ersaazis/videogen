import os
import json
import uuid
from openai import OpenAI
from moviepy import AudioFileClip
import whisper_timestamped as whisper

class VideoPlanner:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._whisper_model = None

    def _get_whisper_model(self):
        if self._whisper_model is None:
            print("Loading Whisper Model (base)...")
            self._whisper_model = whisper.load_model("base", device="cpu")
        return self._whisper_model

    def get_exact_timestamps(self, audio_path):
        model = self._get_whisper_model()
        audio = whisper.load_audio(audio_path)
        result = whisper.transcribe(model, audio, language="en")
        
        words_data = []
        for segment in result["segments"]:
            for word in segment["words"]:
                words_data.append({
                    "text": word["text"].strip(),
                    "start": word["start"],
                    "end": word["end"]
                })
        return words_data

    def fix_transcription_with_ai(self, unfixed_segments, original_script):
        if not self.api_key:
            return unfixed_segments

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            transcription_only = [seg["text"] for seg in unfixed_segments]
            
            prompt = f"""
            You are a subtitle editor. I have a transcription from Whisper that might contain misspellings or slightly wrong words.
            I also have the original script (Source of Truth).
            
            Original Script: "{original_script}"
            
            Mistyped Transcription Segments:
            {json.dumps(transcription_only, indent=2)}
            
            TASK:
            Fix the transcription segments so they match the spelling and words of the Original Script. 
            IMPORTANT: 
            1. Keep the exact same number of segments as provided in the list.
            2. Do not merge or split segments.
            3. Return ONLY a valid JSON list of strings.
            
            Example Output: ["Corrected word 1", "Corrected word 2", ...]
            """
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            fixed_texts = json.loads(content)
            if isinstance(fixed_texts, dict) and "segments" in fixed_texts:
                 fixed_texts = fixed_texts["segments"]
                 
            if len(fixed_texts) == len(unfixed_segments):
                for i in range(len(unfixed_segments)):
                    unfixed_segments[i]["text"] = fixed_texts[i]
                return unfixed_segments
        except Exception as e:
            print(f"  ⚠️ Error AI misspelling fix: {e}")
        return unfixed_segments

    def generate_cc_json(self, valid_items, output_dir):
        all_captions = []
        current_global_time = 0
        
        for item in valid_items:
            path = item.get("audio_path")
            if not path or not os.path.exists(path):
                continue
                
            try:
                exact_words = self.get_exact_timestamps(path)
                words_per_group = 3
                item_captions = []
                
                for i in range(0, len(exact_words), words_per_group):
                    group = exact_words[i:i + words_per_group]
                    group_text = " ".join([w["text"] for w in group])
                    group_start = current_global_time + group[0]["start"]
                    group_end = current_global_time + group[-1]["end"]

                    item_captions.append({
                        "start": round(group_start, 2),
                        "end": round(group_end, 2),
                        "text": group_text,
                        "speaker": item.get("speaker")
                    })
                
                if item.get("message"):
                    item_captions = self.fix_transcription_with_ai(item_captions, item["message"])
                all_captions.extend(item_captions)
            except Exception as e:
                print(f"  ⚠️ Gagal transkripsi exact untuk {path}: {e}")
                
            with AudioFileClip(path) as clip:
                current_global_time += clip.duration
            
        cc_path = os.path.join(output_dir, "cc.json")
        with open(cc_path, "w") as f:
            json.dump(all_captions, f, indent=4)
        return all_captions

    def generate_character_json(self, valid_items, output_dir):
        current_time = 0
        transitions = []
        for item in valid_items:
            path = item.get("audio_path")
            if path and os.path.exists(path):
                with AudioFileClip(path) as clip:
                    duration = clip.duration
                    speaker_lower = item.get("speaker", "Ted").lower()
                    expr = item.get("expression", "normal")
                    image_path = f"avatar/{speaker_lower}/{expr}.png"
                    
                    transitions.append({
                        "speaker": item.get("speaker"),
                        "expression": expr,
                        "image_path": image_path,
                        "start": round(current_time, 2),
                        "end": round(current_time + duration, 2),
                        "duration": round(duration, 2)
                    })
                    current_time += duration
                    
        json_path = os.path.join(output_dir, "character.json")
        with open(json_path, "w") as f:
            json.dump(transitions, f, indent=4)
        return transitions

    def generate_video_description(self, chat_data, output_dir):
        """
        Generates marketing captions (General and Social Media) in description.md.
        """
        if not self.api_key:
            return None
            
        full_script = "\n".join([f"{item.get('speaker', 'Unknown')}: {item.get('message', '')}" for item in chat_data])
        
        prompt = f"""
        You are a social media manager and content creator. Based on this AI podcast script between Ted and Eddy, 
        generate a compelling video description for use during posting.
        
        Script:
        ---
        {full_script}
        ---
        
        TASK:
        Generate exactly two sections:
        
        1. **General Description**: A professional and detailed summary of what is discussed in the video. 
        Focus on the "mind-blowing" facts or deep insights.
        
        2. **Social Media Caption**: A punchy, casual, brolike, and extremely engaging short caption. 
        This is for use in descriptions where excitement is key.
        
        MANDATORY RULES:
        - USE ENGLISH ONLY.
        - DO NOT USE ANY HASHTAGS (#).
        - DO NOT USE SYMBOLS OTHER THAN . , ' ! ? (matching the podcast vibe).
        - KEEP THE SOCIAL MEDIA CAPTION UNDER 30 WORDS.
        
        Output Format:
        # General Description
        [Content here]
        
        # Social Media Caption
        [Content here]
        """
        
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            description_text = response.choices[0].message.content
            
            os.makedirs(output_dir, exist_ok=True)
            desc_file = os.path.join(output_dir, "description.md")
            with open(desc_file, "w", encoding="utf-8") as f:
                f.write(description_text)
                
            return description_text
        except Exception as e:
            print(f"  ⚠️ Error generating description: {e}")
            return None
