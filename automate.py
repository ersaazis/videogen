import os
import json
import traceback
import requests
import re
from dotenv import load_dotenv
import whisper_timestamped as whisper
from moviepy import AudioFileClip, concatenate_audioclips
from df.enhance import enhance, init_df, load_audio, save_audio
import torch
from openai import OpenAI

# Load environment variables
load_dotenv()

# Global model variable to avoid reloading
_whisper_model = None
_df_model = None
_df_state = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("🤖 Loading Whisper Model (base)...")
        _whisper_model = whisper.load_model("base", device="cpu")
    return _whisper_model

def get_df_model():
    global _df_model, _df_state
    if _df_model is None:
        print("🎙️ Loading DeepFilterNet Model...")
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state

def enhance_audio_file(input_path, output_path):
    """
    Menggunakan DeepFilterNet untuk membersihkan noise dan meningkatkan kualitas audio,
    kemudian melakukan amplifikasi (normalisasi) agar suara lebih keras.
    """
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
    """
    Menggunakan whisper-timestamped untuk mendapatkan timestamp kata demi kata yang akurat.
    """
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

def fix_transcription_with_ai(unfixed_segments, original_script):
    """
    Menggunakan AI untuk memperbaiki misspelling pada hasil transkripsi Whisper
    berdasarkan naskah asli (Source of Truth).
    """
    if not DEEPSEEK_API_KEY:
        return unfixed_segments

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        # Siapkan data untuk prompt
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
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "DeepSeek-V3" in DEEPSEEK_API_KEY else None
        )
        
        content = response.choices[0].message.content
        # Extract JSON
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
        else:
            print(f"  ⚠️ AI returned {len(fixed_texts)} segments, expected {len(unfixed_segments)}. Skipping fix.")
            return unfixed_segments
            
    except Exception as e:
        print(f"  ⚠️ Error AI misspelling fix: {e}")
        return unfixed_segments

def generate_cc_json(valid_items, temp_dir):
    """
    Menghasilkan cc.json dengan timestamp kata demi kata yang EXACT menggunakan Whisper.
    """
    print("📝 Menghasilkan Closed Captions (cc.json) dengan EXACT timestamps...")
    
    all_captions = []
    current_global_time = 0
    
    for item in valid_items:
        path = item.get("enhanced_audio_path")
        if not path or not os.path.exists(path):
            path = item.get("audio_path")
            
        if not path or not os.path.exists(path):
            continue
            
        print(f"  🔍 Transcribing: {os.path.basename(path)}")
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
            
            # --- AI MISSPELLING FIX STEP ---
            if item.get("message"):
                print(f"    ✨ Fixing misspellings using AI Reference...")
                item_captions = fix_transcription_with_ai(item_captions, item["message"])
                
            all_captions.extend(item_captions)
            
        except Exception as e:
            print(f"  ⚠️ Gagal transkripsi exact untuk {path}: {e}")
            
        with AudioFileClip(path) as clip:
            current_global_time += clip.duration
        
    cc_path = os.path.join(temp_dir, "cc.json")
    with open(cc_path, "w") as f:
        json.dump(all_captions, f, indent=4)
    print(f"✅ cc.json berhasil disimpan di: {cc_path}")

def generate_character_json(project_id, valid_items, temp_dir):
    """
    Menghasilkan data transisi karakter (character.json).
    """
    print("🎬 Menghasilkan data transisi karakter (character.json)...")
    current_time = 0
    transitions = []
    for item in valid_items:
        path = item.get("enhanced_audio_path")
        if not path or not os.path.exists(path):
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
    print(f"✅ character.json berhasil disimpan di: {json_path}")
    return transitions

def search_pexels_videos(query, min_duration=0, per_page=10):
    """
    Mencari video di Pexels berdasarkan query. Mengembalikan list video.
    """
    api_key = os.getenv("PIXELS_API_KEY")
    if not api_key:
        print("⚠️ PIXELS_API_KEY not found in .env")
        return []
    
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()
    if not clean_query:
        clean_query = "podcast"
        
    url = f"https://api.pexels.com/videos/search?query={clean_query}&per_page={per_page}&orientation=portrait"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("videos", [])
    except Exception as e:
        print(f"  ⚠️ Pexels error: {e}")
    return []

def download_video(url, dest_path):
    """
    Download video dari URL ke path tujuan.
    """
    if os.path.exists(dest_path):
        return True
    try:
        print(f"    📥 Downloading: {os.path.basename(dest_path)}...")
        res = requests.get(url, stream=True, timeout=30)
        if res.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception as e:
        print(f"    ❌ Gagal download {url}: {e}")
    return False

def get_broll_plan_from_llm(cc_data):
    """
    Gunakan DeepSeek untuk merencanakan segmen B-roll berdasarkan isi naskah (cc.json).
    Output: list of {start, end, duration, keyword}
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️ DEEPSEEK_API_KEY tidak ditemukan. Menggunakan rencana fallback.")
        return None
    
    # Gabungkan teks untuk konteks
    full_text = " ".join([c["text"] for c in cc_data])
    total_duration = cc_data[-1]["end"] if cc_data else 0
    
    prompt = f"""
    You are a video editor. Plan the background B-roll segments for a vertical short video.
    The video is {total_duration} seconds long.
    
    Transcript:
    {full_text}
    
    TASK:
    Divide the video into dynamic B-roll segments (each around 10-40 seconds). 
    For each segment, provide a specific search keyword for Pexels.
    The segments must cover the entire {total_duration} seconds without gaps.
    
    Return ONLY a valid JSON list of objects:
    [
      {{"start": 0.0, "end": 5.0, "keyword": "rocket launch"}},
      ...
    ]
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "DeepSeek-V3" in api_key else None # Just in case
        )
        content = response.choices[0].message.content
        # Ekstrak JSON jika ada markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        plan = json.loads(content)
        if isinstance(plan, dict) and "segments" in plan: # Handle common wrapping
            plan = plan["segments"]
        return plan
    except Exception as e:
        print(f"  ⚠️ Gagal mendapatkan plan LLM: {e}")
        return None

def generate_broll_json(valid_items, temp_dir):
    """
    Mencari dan mendownload video B-roll dari Pexels dengan perencanaan LLM.
    """
    print("\n🧠 --- DYNAMIC B-ROLL PLANNING (LLM) ---")
    
    # Load cc.json for planning
    cc_path = os.path.join(temp_dir, "cc.json")
    if not os.path.exists(cc_path):
        print("❌ cc.json tidak ditemukan untuk perencanaan B-roll.")
        return []
    with open(cc_path, 'r') as f:
        cc_data = json.load(f)

    # 1. Rencanakan segmen menggunakan LLM
    plan = get_broll_plan_from_llm(cc_data)
    
    # Fallback jika LLM gagal: gunakan per audio segment seperti sebelumnya
    if not plan:
        print("  ⚠️ Menggunakan fallback planning (per segment).")
        plan = []
        current_t = 0
        for item in valid_items:
            path = item.get("enhanced_audio_path") or item.get("audio_path")
            with AudioFileClip(path) as clip:
                d = clip.duration
            plan.append({"start": current_t, "end": current_t + d, "keyword": item.get("message", "podcast")[:50]})
            current_t += d

    # 2. Acquisition & Download
    project_dir = os.path.dirname(temp_dir)
    broll_video_dir = os.path.join(project_dir, "broll_videos")
    os.makedirs(broll_video_dir, exist_ok=True)

    final_broll_metadata = []
    
    for seg in plan:
        start_t = seg["start"]
        end_t = seg["end"]
        target_duration = end_t - start_t
        keyword = seg["keyword"]
        
        print(f"  🎬 Segmen: {start_t:.1f}s - {end_t:.1f}s | Keyword: '{keyword}'")
        
        videos = search_pexels_videos(keyword)
        if not videos:
            videos = search_pexels_videos("cinematic " + keyword.split()[0])
            
        accumulated_duration = 0
        segments_for_this_plan = []
        
        for v in videos:
            v_id = v.get("id")
            v_dur = v.get("duration", 0)
            
            # Download Logic
            files = v.get("video_files", [])
            video_url = None
            for f in files:
                if f.get("width") in [720, 1080]:
                    video_url = f.get("link")
                    break
            if not video_url and files: video_url = files[0].get("link")
            
            if video_url:
                local_path = os.path.join(broll_video_dir, f"pexels_{v_id}.mp4")
                if download_video(video_url, local_path):
                    # Berapa lama kita pakai video ini?
                    remaining_needed = target_duration - accumulated_duration
                    actual_use_duration = min(v_dur, remaining_needed)
                    
                    segments_for_this_plan.append({
                        "segment_start": round(start_t + accumulated_duration, 2),
                        "segment_end": round(start_t + accumulated_duration + actual_use_duration, 2),
                        "duration": round(actual_use_duration, 2),
                        "local_path": local_path,
                        "pexels_id": v_id,
                        "query": keyword
                    })
                    accumulated_duration += actual_use_duration
            
            if accumulated_duration >= target_duration:
                break
        
        # Jika masih gagal cover, pakai placeholder/generic
        if accumulated_duration < target_duration:
            print(f"    ⚠️ Gagal cover penuh ({accumulated_duration:.1f}/{target_duration:.1f}s)")
            
        final_broll_metadata.extend(segments_for_this_plan)

    broll_path = os.path.join(temp_dir, "broll.json")
    with open(broll_path, "w") as f:
        json.dump(final_broll_metadata, f, indent=4)
    print(f"\n✅ Perencanaan selesai. {len(final_broll_metadata)} potongan video disiapkan.")
    return final_broll_metadata

def join_audio(project_id, temp_dir):
    print(f"🚀 Memulai penggabungan & peningkatan audio untuk project: {project_id}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "projects", project_id)
    chat_path = os.path.join(project_dir, "chat.json")
    if not os.path.exists(chat_path):
        print(f"❌ Error: chat.json tidak ditemukan di {chat_path}")
        return

    os.makedirs(temp_dir, exist_ok=True)
    raw_joined_path = os.path.join(temp_dir, f"{project_id}_joined_raw.mp3")
    final_output_path = os.path.join(temp_dir, f"{project_id}_joined.mp3")

    try:
        with open(chat_path, 'r') as f:
            chat_data = json.load(f)

        audio_clips = []
        valid_items = []
        print("🔍 Mengecek file audio...")
        for i, item in enumerate(chat_data):
            path = item.get("enhanced_audio_path")
            if not path or not os.path.exists(path):
                path = item.get("audio_path")
            if path and os.path.exists(path):
                clip = AudioFileClip(path)
                audio_clips.append(clip)
                valid_items.append(item)
            else:
                print(f"  [!] Audio untuk baris {i+1} tidak ditemukan.")

        if not audio_clips:
            print("❌ Error: Tidak ada audio ditemukan.")
            return

        # 1. Join Audio
        print(f"🔗 Menggabungkan audio...")
        final_audio = concatenate_audioclips(audio_clips)
        final_audio.write_audiofile(raw_joined_path)
        
        # 2. Enhance Audio (Studio Quality)
        print(f"✨ Meningkatkan kualitas audio (Enhancing)...")
        enhance_audio_file(raw_joined_path, final_output_path)
        
        # 3. Metadata
        generate_character_json(project_id, valid_items, temp_dir)
        generate_cc_json(valid_items, temp_dir)
        generate_broll_json(valid_items, temp_dir)
        
        # Cleanup
        for clip in audio_clips:
            clip.close()
        final_audio.close()
        if os.path.exists(raw_joined_path):
            os.remove(raw_joined_path)
            
        print(f"\n✅ Selesai! Audio Clean/Studio Edition disimpan di: {final_output_path}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {str(e)}")
        traceback.print_exc()

PROJECT_NAME = "supermodel"
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

if __name__ == "__main__":
    join_audio(PROJECT_NAME, TEMP_DIR)
