import gradio as gr
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
import math
import traceback
import os
import json
import urllib.parse
import requests
import uuid
import numpy as np
from dotenv import load_dotenv
import shutil
import time
import moviepy
from moviepy import AudioFileClip, concatenate_audioclips
import whisper_timestamped as whisper
from df.enhance import enhance, init_df, load_audio, save_audio
import torch
import video

# Load environment variables
load_dotenv()

# --- Configuration ---
DEFAULT_DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', "")
DEFAULT_REVOICER_SESSION = os.environ.get('REVOICER_CI_SESSION', "")
PIXELS_API_KEY = os.environ.get('PIXELS_API_KEY', "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

def update_env_file(key, value):
    """
    Updates or adds a key-value pair in the .env file.
    """
    env_path = os.path.join(os.getcwd(), ".env")
    lines = []
    found = False
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
            
    if not found:
        lines.append(f"{key}={value}\n")
        
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return f"Updated {key} in .env"

def get_project_list():
    """
    Returns a list of project IDs, sorted by most recently modified.
    """
    projects_dir = os.path.join(os.getcwd(), "projects")
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir, exist_ok=True)
        return []
        
    project_dirs = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
    # Sort by modification time (most recent first)
    project_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(projects_dir, x)), reverse=True)
    return project_dirs

def delete_project_files(project_id):
    """
    Permanently deletes a project directory and its content.
    """
    if not project_id:
        return "Masukkan Project ID yang ingin dihapus.", "", [], None, [], None, get_project_list()
    
    safe_pid = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_pid)
    
    if os.path.exists(project_dir):
        try:
            shutil.rmtree(project_dir)
            # Clear all project-related fields
            return f"Project '{safe_pid}' BERHASIL DIHAPUS PERMANEN.", "", [], None, [], None, get_project_list()
        except Exception as e:
            return f"Gagal menghapus project: {str(e)}", "", [], None, [], None, get_project_list()
    return f"Project '{safe_pid}' tidak ditemukan.", "", [], None, [], None, get_project_list()

def delete_all_projects():
    """
    Permanently deletes ALL projects in the projects folder.
    """
    projects_dir = os.path.join(os.getcwd(), "projects")
    if os.path.exists(projects_dir):
        for d in os.listdir(projects_dir):
            path = os.path.join(projects_dir, d)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except: pass
    return "SEMUA PROJECT TELAH DIHAPUS PERMANEN.", "", [], None, [], None, get_project_list()

# --- Helper Functions ---

# Global model variables for lazy loading
_whisper_model = None
_df_model = None
_df_state = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper Model (base)...")
        _whisper_model = whisper.load_model("base", device="cpu")
    return _whisper_model

def get_df_model():
    global _df_model, _df_state
    if _df_model is None:
        print("Loading DeepFilterNet Model...")
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state

def enhance_audio_file(input_path, output_path):
    model, state = get_df_model()
    audio, _ = load_audio(input_path, sr=state.sr())
    enhanced = enhance(model, state, audio)
    
    # Amplifikasi / Normalisasi (Meningkatkan volume ke level maksimal yang aman)
    max_val = torch.max(torch.abs(enhanced))
    if max_val > 0:
        enhanced = (enhanced / max_val) * 0.9
        
    save_audio(output_path, enhanced, sr=state.sr())
    return output_path

def get_exact_timestamps(audio_path):
    model = get_whisper_model()
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

def fix_transcription_with_ai(unfixed_segments, original_script, api_key):
    """
    Menggunakan AI untuk memperbaiki misspelling pada hasil transkripsi Whisper
    berdasarkan naskah asli (Source of Truth).
    """
    if not api_key:
        return unfixed_segments

    try:
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
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
            model=DEEPSEEK_MODEL,
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

def generate_cc_json(valid_items, temp_dir, api_key):
    all_captions = []
    current_global_time = 0
    
    for item in valid_items:
        path = item.get("audio_path")
        if not path or not os.path.exists(path):
            continue
            
        try:
            exact_words = get_exact_timestamps(path)
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
                item_captions = fix_transcription_with_ai(item_captions, item["message"], api_key)
            all_captions.extend(item_captions)
        except Exception as e:
            print(f"  ⚠️ Gagal transkripsi exact untuk {path}: {e}")
            
        with AudioFileClip(path) as clip:
            current_global_time += clip.duration
        
    cc_path = os.path.join(temp_dir, "cc.json")
    with open(cc_path, "w") as f:
        json.dump(all_captions, f, indent=4)
    return cc_path

def generate_character_json(valid_items, temp_dir):
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
    json_path = os.path.join(temp_dir, "character.json")
    with open(json_path, "w") as f:
        json.dump(transitions, f, indent=4)
    return json_path

def search_pexels_videos(query):
    api_key = os.environ.get("PIXELS_API_KEY")
    if not api_key: return []
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip() or "podcast"
    url = f"https://api.pexels.com/videos/search?query={clean_query}&per_page=10&orientation=portrait"
    headers = {"Authorization": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("videos", [])
    except: pass
    return []

def download_video(url, dest_path):
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

def get_broll_plan_from_llm(cc_data, api_key):
    if not api_key: return None
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
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(model=DEEPSEEK_MODEL, messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        plan = json.loads(content)
        if isinstance(plan, dict) and "segments" in plan: plan = plan["segments"]
        return plan
    except: return None

def generate_broll_json(valid_items, temp_dir, api_key):
    cc_path = os.path.join(temp_dir, "cc.json")
    if not os.path.exists(cc_path): return []
    with open(cc_path, 'r') as f: cc_data = json.load(f)
    plan = get_broll_plan_from_llm(cc_data, api_key)
    if not plan:
        plan = []
        current_t = 0
        for item in valid_items:
            path = item.get("audio_path")
            with AudioFileClip(path) as clip: d = clip.duration
            plan.append({"start": current_t, "end": current_t + d, "keyword": item.get("message", "podcast")[:50]})
            current_t += d

    project_dir = os.path.dirname(temp_dir)
    broll_video_dir = os.path.join(project_dir, "broll_videos")
    os.makedirs(broll_video_dir, exist_ok=True)

    final_broll = []
    for seg in plan:
        start_t, end_t, keyword = seg["start"], seg["end"], seg["keyword"]
        target_dur = end_t - start_t
        videos = search_pexels_videos(keyword)
        acc_dur = 0
        for v in videos:
            v_id, v_dur = v.get("id"), v.get("duration", 0)
            video_url = next((f.get("link") for f in v.get("video_files", []) if f.get("width") in [720, 1080]), None) or (v.get("video_files", [])[0].get("link") if v.get("video_files") else None)
            if video_url:
                local_path = os.path.join(broll_video_dir, f"pexels_{v_id}.mp4")
                if download_video(video_url, local_path):
                    use_dur = min(v_dur, target_dur - acc_dur)
                    final_broll.append({"segment_start": round(start_t + acc_dur, 2), "segment_end": round(start_t + acc_dur + use_dur, 2), "duration": round(use_dur, 2), "local_path": local_path, "pexels_id": v_id, "query": keyword})
                    acc_dur += use_dur
            if acc_dur >= target_dur: break
    
    broll_path = os.path.join(temp_dir, "broll.json")
    with open(broll_path, "w") as f: json.dump(final_broll, f, indent=4)
    return final_broll

def process_studio_mastering(project_id, api_key, progress=gr.Progress()):
    if not project_id: return "Masukkan Project ID.", None
    safe_pid = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_pid)
    chat_path = os.path.join(project_dir, "chat.json")
    if not os.path.exists(chat_path): return f"chat.json tidak ditemukan untuk project {safe_pid}.", None
    
    output_dir = os.path.join(project_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"{safe_pid}_joined_raw.mp3")
    final_path = os.path.join(output_dir, f"{safe_pid}_joined.mp3")
    
    try:
        with open(chat_path, 'r') as f: chat_data = json.load(f)
        audio_clips, valid_items = [], []
        for item in chat_data:
            path = item.get("audio_path")
            if path and os.path.exists(path):
                audio_clips.append(AudioFileClip(path))
                valid_items.append(item)
        
        if not audio_clips: return "Audio tidak ditemukan. Generate audio dulu.", None
        
        progress(0.2, desc="Joining audio...")
        final_audio = concatenate_audioclips(audio_clips)
        final_audio.write_audiofile(raw_path, logger=None)
        
        progress(0.5, desc="Enhancing audio...")
        enhance_audio_file(raw_path, final_path)
        
        progress(0.7, desc="Generating Metadata...")
        generate_character_json(valid_items, output_dir)
        generate_cc_json(valid_items, output_dir, api_key)
        broll_data = generate_broll_json(valid_items, output_dir, api_key)
        
        for clip in audio_clips: clip.close()
        final_audio.close()
        if os.path.exists(raw_path): os.remove(raw_path)
        
        return f"Studio Mastering Selesai!\nAudio: {final_path}\nMetadata (cc.json, character.json, broll.json) ok.", final_path, broll_data
    except Exception as e: return f"Error: {str(e)}", None, []

def process_video_generation(project_id, fps=30, progress=gr.Progress()):
    if not project_id: return "Masukkan Project ID.", None
    safe_pid = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_pid)
    output_dir = os.path.join(project_dir, "output")
    # Juga check temp untuk backward compatibility jika output belum ada
    temp_dir = os.path.join(project_dir, "temp")
    
    # Metadata paths (prioritize output folder)
    cc_path = os.path.join(output_dir, "cc.json") if os.path.exists(os.path.join(output_dir, "cc.json")) else os.path.join(temp_dir, "cc.json")
    char_path = os.path.join(output_dir, "character.json") if os.path.exists(os.path.join(output_dir, "character.json")) else os.path.join(temp_dir, "character.json")
    audio_path = os.path.join(output_dir, f"{safe_pid}_joined.mp3") if os.path.exists(os.path.join(output_dir, f"{safe_pid}_joined.mp3")) else os.path.join(temp_dir, f"{safe_pid}_joined.mp3")
    broll_path = os.path.join(output_dir, "broll.json") if os.path.exists(os.path.join(output_dir, "broll.json")) else os.path.join(temp_dir, "broll.json")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_pid}_final_video.mp4")
    
    if not os.path.exists(cc_path) or not os.path.exists(char_path) or not os.path.exists(audio_path):
        return "File metadata (cc, character, audio) belum lengkap. Selesaikan Step 3 dulu.", None
        
    try:
        video.build_video(
            cc_path=cc_path,
            char_path=char_path,
            audio_path=audio_path,
            output_path=output_path,
            broll_path=broll_path,
            fps=fps
        )
        return f"Video Berhasil Dibuat! Path: {output_path}", output_path
    except Exception as e:
        return f"Gagal render video: {str(e)}", None

def get_video_id(url):
    pattern = r'(?:v=|\/|be\/)([\w-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id, languages=['en', 'id'])
        full_text = " ".join([snippet.text for snippet in transcript_obj])
        return full_text
    except Exception as e:
        return f"Error: Gagal mengambil transkrip: {str(e)}"

def generate_audio_for_message(speaker, message, output_path, audio_id, ci_session, tone="Normal"):
    url = "https://revoicer.app/speak/generate_voice"
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "cookie": f"_ga=GA1.1.954622182.1775418693; lastSelectedAvatarGroup=Talon; ci_session={ci_session}; _ga_WLEQM9ZEZN=GS2.1.s1775463002.o5.g0.t1775463002.j60.l0.h0",
        "origin": "https://revoicer.app",
        "referer": "https://revoicer.app/speak",
        "sec-ch-ua": "\"Chromium\";v=\"146\", \"Not-A.Brand\";v=\"24\", \"Google Chrome\";v=\"146\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "x-requested-with": "XMLHttpRequest"
    }
    
    if speaker.lower() == "ted":
        voice = "verse"
        # Overridden by tone parameter
    elif speaker.lower() == "eddy":
        voice = "onyx"
        # Overridden by tone parameter
    else:
        return None
        
    char_count = len(message)
    word_count = len(message.split())
    text_html = f"<p>{message}</p>**********{tone}||||||||||"
    simple_text = f"{message}\\n"
    
    payload_dict = {
        "languageSelected": "en-US",
        "voiceSelected": voice,
        "toneSelected": tone,
        "languageSelectedClone": "English",
        "speakingRateClone": "default",
        "text": text_html,
        "simpletext": simple_text,
        "charCount": char_count,
        "wordsCount": word_count,
        "campaignId": "109634"
    }
    
    json_payload = json.dumps(payload_dict, separators=(',', ':'))
    encoded_payload = urllib.parse.quote(json_payload)
    data = f"data={encoded_payload}"
    
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("success"):
                download_link = res_json["data"]["voice"]["download_link"]
                download_link = download_link.replace("\\/", "/") 
                
                audio_res = requests.get(download_link)
                if audio_res.status_code == 200:
                    file_name = f"{audio_id}.mp3"
                    full_path = os.path.join(output_path, file_name)
                    with open(full_path, "wb") as f:
                        f.write(audio_res.content)
                    return full_path
    except Exception as e:
        print(f"Audio generation failed for {speaker}: {e}")
    return None

def load_project_data(project_id):
    if not project_id:
        return "", "Ready", "", [], None, [], None
    safe_project_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_project_id)
    project_json_path = os.path.join(project_dir, "project.json")
    
    chat_data = []
    yt_url = ""
    transcript_text = ""
    master_audio = None
    broll_data = []
    final_video = None
    
    if os.path.exists(project_json_path):
        try:
            with open(project_json_path, "r", encoding="utf-8") as f:
                proj_data = json.load(f)
            yt_url = proj_data.get("youtube_url", "")
            
            transcript_path = os.path.join(project_dir, "transcript.md")
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            
            chat_json_path = os.path.join(project_dir, "chat.json")
            if os.path.exists(chat_json_path):
                with open(chat_json_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                
                chat_data = sanitize_chat_data(chat_data)
                
                dirty = False
                for item in chat_data:
                    if "id" not in item:
                        item["id"] = str(uuid.uuid4())
                        dirty = True
                if dirty:
                    with open(chat_json_path, "w", encoding="utf-8") as f:
                        json.dump(chat_data, f, indent=4, ensure_ascii=False)
                
                output_dir = os.path.join(project_dir, "output")
                temp_dir = os.path.join(project_dir, "temp")
                
                # Master Audio
                master_audio_path = os.path.join(output_dir, f"{safe_project_id}_joined.mp3")
                if not os.path.exists(master_audio_path):
                    master_audio_path = os.path.join(temp_dir, f"{safe_project_id}_joined.mp3")
                if os.path.exists(master_audio_path):
                    master_audio = master_audio_path
                
                # B-roll Data
                broll_path = os.path.join(output_dir, "broll.json")
                if not os.path.exists(broll_path):
                    broll_path = os.path.join(temp_dir, "broll.json")
                if os.path.exists(broll_path):
                    with open(broll_path, "r") as f:
                        broll_data = json.load(f)
                
                # Final Video
                video_path = os.path.join(output_dir, f"{safe_project_id}_final_video.mp4")
                if not os.path.exists(video_path):
                    video_path = os.path.join(temp_dir, f"{safe_project_id}_final_video.mp4")
                if os.path.exists(video_path):
                    final_video = video_path

            return yt_url, f"Project '{safe_project_id}' Loaded!", transcript_text, chat_data, master_audio, broll_data, final_video
        except Exception as e:
            return "", f"Error loading: {str(e)}", "", [], None, [], None
    return "", "Ready (Project Baru)", "", [], None, [], None

def load_and_refresh_plist(project_id):
    """
    Wrapper for load_project_data that also refreshes the project list.
    """
    yt_url, status, trans, chat, audio, broll, video = load_project_data(project_id)
    return yt_url, status, trans, chat, audio, broll, video, get_project_list()


def generate_all_audio(project_id, chat_data, ci_session, progress=gr.Progress()):
    if not project_id or not chat_data:
        yield "Project ID atau Data Percakapan kosong.", chat_data
        return
    
    if not ci_session:
        yield "Revoicer CI Session kosong.", chat_data
        return
    
    safe_project_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_project_id)
    audio_dir = os.path.join(project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    success_count = 0
    updated_chat = [dict(item) for item in chat_data] # Create fresh list of dicts
    
    # Initialize all to not processing
    for item in updated_chat:
        item["is_processing"] = False

    for i, item in enumerate(progress.tqdm(updated_chat, desc="Generating Audio")):
        speaker = item.get("speaker")
        message = item.get("message")
        audio_id = item.get("id")
        tone = item.get("tone", "Normal")
        
        if not audio_id:
            audio_id = str(uuid.uuid4())
            updated_chat[i]["id"] = audio_id
            
        existing_path = item.get("audio_path")
        if existing_path and os.path.exists(existing_path):
            continue
        
        # Mark as processing (hanya untuk visual)
        updated_chat[i]["is_processing"] = True
        yield f"Generating Audio {i+1}/{len(updated_chat)}: {speaker}...", updated_chat
            
        path = generate_audio_for_message(speaker, message, audio_dir, audio_id, ci_session, tone=tone)
        
        # Selesai processing
        updated_chat[i]["is_processing"] = False
        if path:
            updated_chat[i]["audio_path"] = path
            success_count += 1
        
        # Yield update state agar audio player terisi
        yield f"Audio {i+1} Ready. Lanjut...", updated_chat
        time.sleep(0.5) # Beri nafas untuk UI render
                
    # Save back to chat.json
    chat_json_path = os.path.join(project_dir, "chat.json")
    try:
        # Clean up processing flags before saving
        save_data = []
        for item in updated_chat:
            clean_item = dict(item)
            clean_item.pop("is_processing", None)
            save_data.append(clean_item)
            
        with open(chat_json_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat data: {e}")
        
    yield f"Selesai! Berhasil generate {success_count} file audio.", updated_chat


# --- Metadata Constants ---
EXPRESSIONS = ["afraid", "angry", "disgusted", "happy", "nauseated", "normal", "sad", "surprised"]
TONES = ['Normal', 'Standard', 'Angry', 'Cheerful', 'Excited', 'Friendly', 'Shouting', 'Terrified']

def sanitize_chat_data(chat_data):
    """
    Ensures all expressions and tones are within allowed choices to prevent Gradio errors.
    """
    for item in chat_data:
        # Sanitize Expression
        expr = item.get("expression", "normal").lower()
        if expr not in EXPRESSIONS:
            # Maybe it's a tone accidentally put in expression field?
            # Or just invalid
            item["expression"] = "normal"
        else:
            item["expression"] = expr
            
        # Sanitize Tone
        tone = item.get("tone", "Normal").capitalize()
        # Handle special case for "Normal" which might keep it lowercase
        if tone == "Normal": tone = "Normal" 
        
        if tone not in TONES:
            item["tone"] = "Normal"
        else:
            item["tone"] = tone
            
        # Sanitize Message Punctuation: Only allow . , ' ! ?
        if "message" in item:
            msg = item["message"]
            # Keep letters, numbers, spaces, and . , ' ! ?
            # Strip everything else like " - ; : ( ) etc.
            sanitized_msg = re.sub(r"[^a-zA-Z0-9\s.,'!?]", "", msg)
            item["message"] = sanitized_msg
            
    return chat_data

def parse_raw_script(raw_result):
    chat_data = []
    # Pattern to match Speaker: [Expression] [Tone] Dialog or variations
    # Example: Ted: [happy] [Excited] Dude, this is awesome!
    pattern = r'(?i)^(Ted|Eddy)\s*:\s*(?:\[(\w+)\])?\s*(?:\[(\w+)\])?\s*(.*)'
    
    for line in raw_result.split("\n"):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            speaker = match.group(1).capitalize()
            val1 = match.group(2)
            val2 = match.group(3)
            message = match.group(4)
            
            expr = "normal"
            tone = "Normal"
            
            # Logic to handle one or two brackets
            if val1 and val2:
                expr = val1.lower()
                tone = val2.capitalize()
            elif val1:
                # Only one bracket. Check if it's an expression or a tone.
                v = val1.lower()
                if v in EXPRESSIONS:
                    expr = v
                elif val1.capitalize() in TONES:
                    tone = val1.capitalize()
                else:
                    expr = "normal" # Default
                    
            chat_data.append({
                "id": str(uuid.uuid4()),
                "speaker": speaker,
                "expression": expr,
                "tone": tone,
                "message": message
            })
        elif line and chat_data:
            chat_data[-1]["message"] += f" \n{line}"
            
    chat_data = sanitize_chat_data(chat_data)
    
    if not chat_data:
        chat_data = [{"id": str(uuid.uuid4()), "speaker": "System", "expression": "normal", "tone": "Normal", "message": "Failed to parse script:\n" + raw_result}]
    return chat_data

def generate_script_only(project_id, youtube_url, api_key):
    if not project_id:
        return "Masukkan Project ID.", "", []
    if not youtube_url:
        return "Masukkan URL YouTube.", "", []
    
    safe_project_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_id)
    project_dir = os.path.join(os.getcwd(), "projects", safe_project_id)
    os.makedirs(project_dir, exist_ok=True)
    
    video_id = get_video_id(youtube_url)
    if not video_id:
        return "URL YouTube tidak valid!", "", []
    
    transcript = get_transcript(video_id)
    if transcript.startswith("Error:"):
        return f"{transcript}", "", []
        
    transcript_path = os.path.join(project_dir, "transcript.md")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript for {safe_project_id}\n\n{transcript}")
        
    project_metadata = {
        "project_id": safe_project_id,
        "youtube_url": youtube_url,
        "video_id": video_id
    }
    project_json_path = os.path.join(project_dir, "project.json")
    with open(project_json_path, "w", encoding="utf-8") as f:
        json.dump(project_metadata, f, indent=4, ensure_ascii=False)
    
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    prompt = f"""
    You are an expert copywriter and podcast script writer. Based on the YouTube transcript below, 
    write a highly engaging, mind-blowing podcast conversation between two close friends: **Ted** and **Eddy**.
    
    Character Background:
    - Ted & Eddy both grew up in the USA. 
    - Their speaking style is HIGHLY INFORMAL, casual, and "bro-like". They frequently use modern American slang, idioms, and speak as if they are in an unscripted, high-energy podcast.
    
    MANDATORY RULES:
    1. The ENTIRE CONVERSATION MUST BE IN ENGLISH. 
    2. Zero formal language. Make it raw, deep, and conversational while fully dissecting the transcript's topic.
    3. THE VERY FIRST LINE OF THE CONVERSATION MUST ALWAYS BE SPOKEN BY TED. Ted MUST open the podcast with a SUPER ENGAGING HOOK.
    4. Keep the flow dynamic, reacting to each other's points.
    5. Each dialog line MUST specify character expression from this list: afraid, angry, disgusted, happy, nauseated, normal, sad, surprised.
    6. Each dialog line MUST also specify a voice tone from this list: ['Normal', 'Standard', 'Angry', 'Cheerful', 'Excited', 'Friendly', 'Shouting', 'Terrified'].
    7. CRITICAL PUNCTUATION RULE: Use ONLY these symbols in the dialog: . , ' ! ? (Period, Comma, Apostrophe, Exclamation, Question). DO NOT use quotation marks ("), dashes (-), semi-colons (;), or any other special symbols.
    8. DURATION CONSTRAINT: The total conversation MUST be under 3 minutes (180 seconds). Keep the total word count for the entire script between 350-450 words.
    9. CONCISE LINES: Each individual dialog line MUST be short and punchy (max 25 words per line). NO LONG MONOLOGUES. Keep it fast-paced.
    10. STRUCTURE: Aim for about 20-30 lines of dialogue in total.
    
    Output Format MUST strictly use these prefixes:
    Ted: [expression] [Tone] (Dialog...)
    Eddy: [expression] [Tone] (Dialog...)
    
    Example:
    Ted: [happy] [Excited] Dude, that was absolutely insane!
    Eddy: [surprised] [Standard] Bro, I didn't even know that was possible.
    
    YouTube Transcript:
    ---
    {transcript}
    """
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant and communications expert."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        raw_result = response.choices[0].message.content
        
        chat_data = parse_raw_script(raw_result)
        chat_json_path = os.path.join(project_dir, "chat.json")
        with open(chat_json_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=4, ensure_ascii=False)
            
        return "Script Successfully Generated! Please edit & generate audio per-line.", transcript, chat_data, get_project_list()
    except Exception as e:
        return f"DeepSeek API Failure: {str(e)}", transcript, [], get_project_list()

def process_auto_generate(project_id, youtube_url, api_key, ci_session, progress=gr.Progress()):
    if not project_id:
        yield "Enter Project ID.", "", [], None, [], None, get_project_list()
        return
    if not youtube_url:
        yield "Enter YouTube URL.", "", [], None, [], None, get_project_list()
        return

    # Step 1: Generate Script
    progress(0, desc="[1/4] Generating Script...")
    status, script, chat_data, plist = generate_script_only(project_id, youtube_url, api_key)
    if "Failed" in status or "error" in status.lower() or not chat_data:
        yield status, script, chat_data, None, [], None, plist
        return
    
    yield f"[1/4] Script OK. Preparing Audio...", script, chat_data, None, [], None, plist

    # Step 2: Generate All Audio
    progress(0.2, desc="[2/4] Generating Audio Studio...")
    final_chat_data = chat_data
    for status_update, updated_chat in generate_all_audio(project_id, chat_data, ci_session, progress):
        final_chat_data = updated_chat
        yield f"[2/4] Audio Studio: {status_update}", script, final_chat_data, None, [], None, plist

    # Step 3: Studio Mastering
    progress(0.7, desc="[3/4] Studio Mastering...")
    status_master, master_audio, broll_data = process_studio_mastering(project_id, api_key, progress)
    if "Error" in status_master:
         yield status_master, script, final_chat_data, master_audio, broll_data, None, plist
         return

    yield f"[3/4] Mastering OK. Rendering Final...", script, final_chat_data, master_audio, broll_data, None, plist

    # Step 4: Auto Render Final 60fps
    progress(0.9, desc="[4/4] Rendering Final 60fps...")
    status_video, video_path = process_video_generation(project_id, fps=60, progress=progress)

    yield f"🚀 Auto Production Complete!\n{status_video}", script, final_chat_data, master_audio, broll_data, video_path, get_project_list()

# --- Style Definitions (Gradio 6.0) ---
CUSTOM_CSS = """
    .container { max-width: 1000px !important; margin: auto; padding-top: 2rem; }
    .title-area { text-align: center; margin-bottom: 2rem; }
    .chat-row { border: 1px solid #374151; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: rgba(31,41,55,0.4); }
    .generating-pulse {
        animation: pulse 1.5s infinite;
        border: 2px solid #60a5fa !important;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .project-selector-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 5px 15px;
        margin-bottom: 25px;
        background: rgba(31, 41, 55, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    .project-item-new {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .project-item-active {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(96, 165, 250, 0.3);
    }
    .project-item {
        background: rgba(55, 65, 81, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
    }
    .project-item:hover {
        background: rgba(75, 85, 99, 0.6) !important;
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
"""
CUSTOM_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

# --- UI Setup ---
with gr.Blocks() as demo:
    chat_state = gr.State([])
    broll_state = gr.State([])
    project_list_state = gr.State(get_project_list())

    # Pre-define core components to avoid NameError in gr.render, but don't render them yet
    project_input = gr.Textbox(label="Project ID", placeholder="e.g. project-gpt", scale=1, render=False)
    yt_input = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=...", scale=2, render=False)
    api_input = gr.Textbox(label="DeepSeek API Key", placeholder="sk_...", value=DEFAULT_DEEPSEEK_API_KEY, type="text", render=False)
    revoicer_session_input = gr.Textbox(label="Revoicer CI Session", placeholder="ci_session value", value=DEFAULT_REVOICER_SESSION, type="text", render=False)
    status_out = gr.Textbox(label="Status", value="Ready", interactive=False, render=False)
    
    with gr.Column(elem_classes=["container"]):
        with gr.Column(elem_classes=["title-area"]):
            gr.Markdown('<div style="text-align:center;"><span style="background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800;">DeepStream Studio ⚓</span><br><span style="color: #6b7280; font-size: 1.1rem;">Podcast Editor & Audio Generator Per-Line</span></div>')

        # Project Selector Row (TOP SECTION)
        @gr.render(inputs=[project_list_state, project_input])
        def render_project_selector(plist, current_pid):
            with gr.Column():
                with gr.Row():
                    # New Project Button
                    btn_new = gr.Button("➕ Create New Project", variant="secondary", size="sm", elem_classes=["project-item-new"], scale=1)
                    btn_clear_all_trigger = gr.Button("🗑️ Clear All Projects", variant="stop", size="sm", scale=1)
                
                with gr.Row(visible=False, elem_classes=["chat-row"]) as confirm_all_row:
                    with gr.Column():
                        gr.Markdown("### ⚠️ CONFIRM GLOBAL DELETE!\nAll project folders, scripts, audio, and video will be permanently deleted. This cannot be undone.")
                        with gr.Row():
                            btn_confirm_all = gr.Button("YES, DELETE ALL DATA PERMANENTLY", variant="stop")
                            btn_cancel_all = gr.Button("CANCEL", variant="secondary")

                with gr.Row(elem_classes=["project-selector-container"]):
                    # Existing Projects
                    for pid in plist:
                        is_active = (pid == current_pid)
                        btn_p = gr.Button(
                            f"⚓ {pid}", 
                            variant="primary" if is_active else "secondary", 
                            size="sm", 
                            elem_classes=["project-item-active" if is_active else "project-item"],
                            scale=1
                        )
                        
                        def make_click_fn(p=pid):
                            def on_click():
                                return p, f"Project '{p}' Loaded!", *load_and_refresh_plist(p)
                            return on_click
                        
                        btn_p.click(
                            fn=make_click_fn(pid), 
                            outputs=[project_input, status_out, yt_input, status_out, trans_out, chat_state, mastering_preview, broll_state, video_out, project_list_state]
                        )
                        
                # Event Listeners for Clear All (inside render because the buttons are dynamic)
                btn_new.click(fn=lambda: ("", "New Project Ready!"), outputs=[project_input, status_out])
                btn_clear_all_trigger.click(fn=lambda: gr.update(visible=True), outputs=[confirm_all_row])
                btn_cancel_all.click(fn=lambda: gr.update(visible=False), outputs=[confirm_all_row])
                btn_confirm_all.click(
                    fn=delete_all_projects,
                    outputs=[status_out, trans_out, chat_state, mastering_preview, broll_state, video_out, project_list_state]
                ).then(fn=lambda: (gr.update(visible=False), ""), outputs=[confirm_all_row, project_input])
        
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=3):
                    project_input.render() # Put it here
                    yt_input = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=...", scale=2)
                with gr.Column(scale=3):
                    api_input = gr.Textbox(label="DeepSeek API Key", placeholder="sk_...", value=DEFAULT_DEEPSEEK_API_KEY, type="text")
                    revoicer_session_input = gr.Textbox(label="Revoicer CI Session", placeholder="ci_session value", value=DEFAULT_REVOICER_SESSION, type="text")
            with gr.Row():
                btn_gen_script = gr.Button("Manual", variant="primary")
                btn_gen_auto = gr.Button("Autopilot", variant="secondary")
        
        status_out.render()
        
        with gr.Tabs():
            with gr.TabItem("1. Source Transcript"):
                trans_out = gr.TextArea(label="Source Transcript", lines=15, interactive=False)

            with gr.TabItem("2. Audio Studio"):
                btn_generate_all = gr.Button("Generate All Audio", variant="primary")
                
                @gr.render(inputs=[chat_state, project_input])
                def render_chat_studio(chat_data, proj_id):
                    def save_chat_silently(state, pid):
                        if not pid: return
                        safe_proj_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', pid)
                        chat_json_path = os.path.join(os.getcwd(), "projects", safe_proj_id, "chat.json")
                        try:
                            with open(chat_json_path, "w", encoding="utf-8") as f:
                                json.dump(state, f, indent=4, ensure_ascii=False)
                        except: pass

                    if not chat_data:
                        gr.Markdown("*Belum ada naskah. Silakan generate atau load Project ID.*")
                        return
                        
                    for i, item in enumerate(chat_data):
                        with gr.Column(elem_classes=["chat-row"]):
                            with gr.Row():
                                with gr.Column(scale=1, min_width=60):
                                    btn_up = gr.Button("Up", size="sm")
                                    btn_down = gr.Button("Down", size="sm")
                                    btn_del = gr.Button("Delete", size="sm", variant="stop")
                                    
                                with gr.Column(scale=6):
                                    with gr.Row():
                                        spk = gr.Dropdown(choices=["Ted", "Eddy"], value=item.get("speaker", "Ted"), label="Speaker", interactive=True, scale=1)
                                        expr = gr.Dropdown(choices=EXPRESSIONS, value=item.get("expression", "normal"), label="Expression", interactive=True, scale=1)
                                        tn = gr.Dropdown(choices=TONES, value=item.get("tone", "Normal"), label="Tone", interactive=True, scale=1)
                                        msg = gr.TextArea(value=item.get("message", ""), label="Text Dialog", lines=2, interactive=True, scale=2)
                                
                                with gr.Column(scale=2):
                                    is_processing = item.get("is_processing", False)
                                    audio_player = gr.Audio(
                                        value=item.get("audio_path", None), 
                                        type="filepath", 
                                        label="PROCESSING..." if is_processing else "Preview Audio", 
                                        interactive=False,
                                        elem_classes=["generating-pulse"] if is_processing else []
                                    )
                                    btn_audio = gr.Button("Generating..." if is_processing else "Generate Audio", variant="primary" if is_processing else "secondary", interactive=not is_processing)
                                    
                            # Auto save logic
                            def on_change(s_val, e_val, t_val, m_val, idx=i):
                                chat_data[idx]["speaker"] = s_val
                                chat_data[idx]["expression"] = e_val
                                chat_data[idx]["tone"] = t_val
                                chat_data[idx]["message"] = m_val
                                save_chat_silently(chat_data, proj_id)
                                    
                            spk.change(fn=on_change, inputs=[spk, expr, tn, msg])
                            expr.change(fn=on_change, inputs=[spk, expr, tn, msg])
                            tn.change(fn=on_change, inputs=[spk, expr, tn, msg])
                            msg.blur(fn=on_change, inputs=[spk, expr, tn, msg])
                            
                            # Generate local Audio
                            def on_gen_local_audio(state, s_val, e_val, t_val, m_val, pid, ci_session, idx=i):
                                if not pid: return None
                                if not ci_session: return None
                                state[idx]["speaker"] = s_val
                                state[idx]["expression"] = e_val
                                state[idx]["tone"] = t_val
                                state[idx]["message"] = m_val
                                
                                safe_proj_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', pid)
                                audio_dir = os.path.join(os.getcwd(), "projects", safe_proj_id, "audio")
                                os.makedirs(audio_dir, exist_ok=True)
                                
                                if "id" not in state[idx]:
                                    state[idx]["id"] = str(uuid.uuid4())
                                
                                path = generate_audio_for_message(s_val, m_val, audio_dir, state[idx]["id"], ci_session, tone=t_val)
                                if path:
                                    state[idx]["audio_path"] = path
                                    
                                save_chat_silently(state, pid)
                                return path
                                
                            btn_audio.click(fn=on_gen_local_audio, inputs=[chat_state, spk, expr, tn, msg, project_input, revoicer_session_input], outputs=[audio_player])

                            # Ordering logic
                            def move_up(state, pid, idx=i):
                                if idx > 0:
                                    state[idx], state[idx-1] = state[idx-1], state[idx]
                                    save_chat_silently(state, pid)
                                return state
                            btn_up.click(fn=move_up, inputs=[chat_state, project_input], outputs=[chat_state])

                            def move_down(state, pid, idx=i):
                                if idx < len(state) - 1:
                                    state[idx], state[idx+1] = state[idx+1], state[idx]
                                    save_chat_silently(state, pid)
                                return state
                            btn_down.click(fn=move_down, inputs=[chat_state, project_input], outputs=[chat_state])

                            def delete_row(state, pid, idx=i):
                                state.pop(idx)
                                save_chat_silently(state, pid)
                                return state
                            btn_del.click(fn=delete_row, inputs=[chat_state, project_input], outputs=[chat_state])

                    btn_add = gr.Button("Add New Dialogue", variant="secondary")
                    def add_row(state, pid):
                        state.append({"id": str(uuid.uuid4()), "speaker": "Ted", "expression": "normal", "tone": "Normal", "message": ""})
                        save_chat_silently(state, pid)
                        return state
                    btn_add.click(fn=add_row, inputs=[chat_state, project_input], outputs=[chat_state])

            with gr.TabItem("3. Studio Mastering"):
                btn_mastering = gr.Button("Join & Mastering Step (Enhance + Metadata)", variant="primary")
                mastering_preview = gr.Audio(label="Preview Master Audio", type="filepath", interactive=False)
                
                @gr.render(inputs=[broll_state])
                def render_broll_previews(data):
                    if not data:
                        return gr.Markdown("*Belum ada data B-roll. Klik mastering untuk menghasilkan naskah broll.*")
                    with gr.Row():
                        for i, seg in enumerate(data):
                            with gr.Column(scale=1, min_width=250):
                                gr.Video(value=seg.get("local_path"), label=f"Segment {i+1} ({seg.get('segment_start')}s - {seg.get('segment_end')}s)", interactive=False)
                                gr.Markdown(f"**Keyword**: {seg.get('query')}")

                btn_mastering.click(
                    fn=process_studio_mastering,
                    inputs=[project_input, api_input],
                    outputs=[status_out, mastering_preview, broll_state]
                )

            with gr.TabItem("4. Video Generator"):
                with gr.Row():
                    fps_input = gr.Slider(minimum=1, maximum=60, value=30, step=1, label="FPS Render")
                    btn_render_video = gr.Button("Render Final Video (9:16)", variant="primary")
                
                video_out = gr.Video(label="Final Video Result")
                
                btn_render_video.click(
                    fn=process_video_generation,
                    inputs=[project_input, fps_input],
                    outputs=[status_out, video_out]
                )
                            

    # Load project data only on Blur or Submit (Enter) to avoid infinite loops during typing
    # Also refreshes project list
    project_input.submit(
        fn=load_and_refresh_plist,
        inputs=[project_input],
        outputs=[yt_input, status_out, trans_out, chat_state, mastering_preview, broll_state, video_out, project_list_state]
    )
    project_input.blur(
        fn=load_and_refresh_plist,
        inputs=[project_input],
        outputs=[yt_input, status_out, trans_out, chat_state, mastering_preview, broll_state, video_out, project_list_state]
    )


    btn_gen_script.click(
        fn=generate_script_only, 
        inputs=[project_input, yt_input, api_input], 
        outputs=[status_out, trans_out, chat_state, project_list_state]
    )

    btn_gen_auto.click(
        fn=process_auto_generate,
        inputs=[project_input, yt_input, api_input, revoicer_session_input],
        outputs=[status_out, trans_out, chat_state, mastering_preview, broll_state, video_out, project_list_state]
    )

    btn_generate_all.click(
        fn=generate_all_audio,
        inputs=[project_input, chat_state, revoicer_session_input],
        outputs=[status_out, chat_state],
        show_progress="full"
    )

    # .env Auto-Persistence
    api_input.change(fn=lambda v: update_env_file("DEEPSEEK_API_KEY", v), inputs=[api_input])
    revoicer_session_input.change(fn=lambda v: update_env_file("REVOICER_CI_SESSION", v), inputs=[revoicer_session_input])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        allowed_paths=[os.getcwd()],
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS
    )
