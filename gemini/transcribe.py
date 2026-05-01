import sys
import os
import signal
import wave
import time
import numpy as np
from moonshine_voice import MicTranscriber, ModelArch, get_model_for_language, TranscriptEventListener

def save_wav(audio_data, samplerate, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(folder, f"recording_{timestamp}.wav")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes)
        wf.setframerate(samplerate)
        # Convert float32 back to int16
        audio_int16 = (np.array(audio_data) * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    
    return filename

class TranscriptionCollector(TranscriptEventListener):
    def __init__(self, silence_timeout=1.5):
        self.transcript_parts = []
        self.last_update_time = time.time()
        self.silence_timeout = silence_timeout
        self.is_finished = False
        self.audio_buffer = []

    def on_line_completed(self, event):
        self.transcript_parts.append(event.line.text.strip())
        self.last_update_time = time.time()
        # For simple voice input, we often just want one line.
        # But we'll wait for the silence timeout instead of exiting immediately.

    def on_line_text_changed(self, event):
        self.last_update_time = time.time()

def main():
    # Configuration
    model_arch_type = ModelArch.MEDIUM_STREAMING
    language = "en"
    recordings_folder = "/home/kaiser/.gemini/extensions/gemini-moonshine/recordings"
    silence_timeout = 1.5
    
    def signal_handler(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        model_path, model_arch = get_model_for_language(language, model_arch_type)
        
        # Initialize transcriber
        transcriber = MicTranscriber(
            model_path=model_path,
            model_arch=model_arch,
        )
        
        collector = TranscriptionCollector(silence_timeout=silence_timeout)
        transcriber.add_listener(collector)

        # We want to capture audio too. Since MicTranscriber doesn't expose it easily,
        # we'll wrap its internal audio_callback or just accept we might not save it for now.
        # Actually, let's try to capture it by wrapping the callback if we really want it.
        
        original_callback = transcriber._start_listening
        
        def wrapped_start_listening():
            # This is a bit hacky but it works to capture audio
            def audio_callback(in_data, frames, time_info, status):
                if not transcriber._should_listen:
                    return
                if in_data is not None:
                    audio_data = in_data.astype(np.float32).flatten()
                    collector.audio_buffer.extend(audio_data.tolist())
                    transcriber.mic_stream.add_audio(audio_data, transcriber._samplerate)
            
            import sounddevice as sd
            transcriber._sd_stream = sd.InputStream(
                samplerate=transcriber._samplerate,
                blocksize=transcriber._blocksize,
                device=transcriber._device,
                channels=transcriber._channels,
                dtype="float32",
                callback=audio_callback,
            )
            transcriber._sd_stream.start()

        transcriber._start_listening = wrapped_start_listening
        
        transcriber.start()
        
        # Wait for transcription and silence
        start_time = time.time()
        max_duration = 30 # absolute timeout
        
        while time.time() - start_time < max_duration:
            if collector.transcript_parts and (time.time() - collector.last_update_time > silence_timeout):
                break
            time.sleep(0.1)
        
        transcriber.stop()
        
        # Save audio if captured
        if collector.audio_buffer:
            saved_path = save_wav(collector.audio_buffer, 16000, recordings_folder)
            print(f"Audio saved to: {saved_path}", file=sys.stderr)

        final_transcript = " ".join(collector.transcript_parts).strip()
        if final_transcript:
            print(final_transcript)
        else:
            sys.exit(1)
            
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            transcriber.close()
        except:
            pass

if __name__ == "__main__":
    main()
