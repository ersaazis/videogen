import os
import shutil
import torch
from moviepy import AudioFileClip, concatenate_audioclips, CompositeAudioClip
from moviepy.audio.fx import AudioLoop
from df.enhance import enhance, init_df, load_audio, save_audio

class AudioEnhancer:
    def __init__(self):
        self._df_model = None
        self._df_state = None

    def _get_df_model(self):
        if self._df_model is None:
            print("Loading DeepFilterNet Model...")
            self._df_model, self._df_state, _ = init_df()
        return self._df_model, self._df_state

    def enhance_audio_file(self, input_path, output_path):
        """
        Enhances and normalizes the audio file using DeepFilterNet.
        """
        model, state = self._get_df_model()
        audio, _ = load_audio(input_path, sr=state.sr())
        enhanced = enhance(model, state, audio)
        
        # Amplifikasi / Normalisasi (Meningkatkan volume ke level maksimal yang aman)
        max_val = torch.max(torch.abs(enhanced))
        if max_val > 0:
            enhanced = (enhanced / max_val) * 0.9
            
        save_audio(output_path, enhanced, sr=state.sr())
        return output_path

    def join_and_enhance(self, audio_paths, output_dir, safe_pid, bg_music_path=None, bg_music_volume=0.1, progress_callback=None):
        """
        Joins multiple audio files and enhances the resulting master track.
        """
        raw_path = os.path.join(output_dir, f"{safe_pid}_joined_raw.mp3")
        final_path = os.path.join(output_dir, f"{safe_pid}_joined.mp3")
        
        audio_clips = []
        for path in audio_paths:
            if os.path.exists(path):
                audio_clips.append(AudioFileClip(path))
        
        if not audio_clips:
            return None
        
        try:
            if progress_callback: progress_callback(0.2, desc="Joining audio...")
            final_audio = concatenate_audioclips(audio_clips)
            final_audio.write_audiofile(raw_path, logger=None)
            
            if progress_callback: progress_callback(0.5, desc="Enhancing audio...")
            self.enhance_audio_file(raw_path, final_path)
            
            # --- Mix background music if provided ---
            if bg_music_path and os.path.exists(bg_music_path):
                if progress_callback: progress_callback(0.8, desc="Mixing background music...")
                voice_audio = AudioFileClip(final_path)
                bg_music = AudioFileClip(bg_music_path).with_volume_scaled(bg_music_volume)
                
                # Loop or trim background music to match voice duration
                if bg_music.duration < voice_audio.duration:
                    bg_music = bg_music.with_effects([AudioLoop(duration=voice_audio.duration)])
                else:
                    bg_music = bg_music.subclipped(0, voice_audio.duration)
                
                # Mix them together
                mixed_audio = CompositeAudioClip([voice_audio, bg_music])
                mixed_path = os.path.join(output_dir, f"{safe_pid}_joined_mixed.mp3")
                mixed_audio.write_audiofile(mixed_path, logger=None)
                
                # Close all clips before file operations
                voice_audio.close()
                bg_music.close()
                mixed_audio.close()
                
                # Replace the original clean audio with the mixed one
                if os.path.exists(final_path):
                    os.remove(final_path)
                shutil.move(mixed_path, final_path)
            
            # Cleanup intermediate files
            for clip in audio_clips: clip.close()
            final_audio.close()
            if os.path.exists(raw_path):
                os.remove(raw_path)
                
            return final_path
        except Exception as e:
            print(f"Error in AudioEnhancer: {e}")
            return None

if __name__ == "__main__":
    # Example usage / debug
    # enhancer = AudioEnhancer()
    # enhancer.join_and_enhance(["path/to/1.mp3", "path/to/2.mp3"], "output/", "test_project")
    pass
