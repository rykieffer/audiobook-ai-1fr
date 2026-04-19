
import os
import gradio as gr
from pathlib import Path

from audiobook_ai.core.epub_parser import EPUBParser
from audiobook_ai.tts.qwen_engine import TTSEngine
from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer

class AudiobookGUI:
    def __init__(self):
        self._book_title = "Audiobook"
        self._project_dir = os.path.expanduser("~/audiobook_output")
        self._voice_model_variant_base = "Qwen3-4B"
        self.narrator_wav_path = None
        self.narrator_ref_text = None
        self._chapters_list = []
        self._tags = {}
        self._auto_save_timer = None
        self.segments_dir = os.path.join(self._project_dir, "segments")
        self.engine = TTSEngine()
        self.analyzer = None
        self.voice_strategy = "single_narrator"

    def build(self):
        with gr.Blocks() as demo:
            gr.Markdown("# AI Audiobook Generator - Simplified")
            
            with gr.Tabs():
                # Tab 1: Analysis
                with gr.Tab("1. Analysis"):
                    file_input = gr.File(label="Upload EPUB", file_types=[".epub"])
                    analyze_btn = gr.Button("Analyze")
                    status_text = gr.Textbox(label="Status", interactive=False)
                    analyze_btn.click(
                        fn=self.analyze_epub,
                        inputs=[file_input],
                        outputs=[status_text]
                    )
                
                # Tab 2: Voices - Simplified to single narrator
                with gr.Tab("2. Voices"):
                    gr.Markdown("### Narrator Voice")
                    narrator_desc = gr.Textbox(
                        label="Narrator Description",
                        value="A calm, clear narrator voice"
                    )
                    narrator_ref_audio = gr.File(
                        label="Reference Audio (optional)",
                        type="filepath"
                    )
                    gen_btn = gr.Button("Generate Audiobook")
                    output_audio = gr.Audio(label="Generated Audiobook")
                
                # Tab 3 removed for simplicity
            
            # Connect generation
            gen_btn.click(
                fn=self.start_generation,
                inputs=[],  # Uses internal state
                outputs=[output_audio]
            )
        
        return demo

    def analyze_epub(self, file):
        if file is None:
            return "Please upload an EPUB file"
        
        epub_path = file.name
        parser = EPUBParser(epub_path)
        segments = parser.parse()
        
        self._chapters_list = parser.get_chapters()
        self._tags = parser.get_tags()
        self.epub_path = epub_path
        self._analyzed = True
        
        return f"Analyzed {len(segments)} segments"

    def start_generation(self):
        if not hasattr(self, '_analyzed') or not self._analyzed:
            return "Please analyze an EPUB first"
        
        engine = self._get_engine()
        engine.load_model(self._voice_model_variant_base)
        
        parser = EPUBParser(self.epub_path)
        all_segs = parser.parse()
        
        self.segments_dir = os.path.join(self._project_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)
        
        total = len(all_segs)
        generated = 0
        failed = 0
        
        for i, seg in enumerate(all_segs):
            seg_id = f"ch0_s{i}"
            out_path = os.path.join(self.segments_dir, f"{seg_id}.wav")
            
            if os.path.exists(out_path):
                continue
            
            text = seg.text if hasattr(seg, 'text') else ""
            if not text.strip():
                continue
            
            # Single narrator mode - simplified
            ref_audio = self.narrator_wav_path
            
            from audiobook_ai.analysis.character_analyzer import EMOTION_INSTRUCTIONS_FR
            emotion_instr = EMOTION_INSTRUCTIONS_FR.get("calm", "")
            
            ref_text = self.narrator_ref_text or text
            
            try:
                gen_path = engine.generate_voice_clone(
                    text=text,
                    ref_audio_path=ref_audio,
                    ref_text=ref_text,
                    language="french",
                    emotion_instruction=emotion_instr,
                    output_path=out_path,
                )
                if gen_path:
                    generated += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"Failed: {e}")
        
        engine.unload_model()
        
        # Simple assembly using sox
        wav_files = sorted([
            os.path.join(self.segments_dir, f)
            for f in os.listdir(self.segments_dir)
            if f.endswith('.wav')
        ])
        
        if wav_files:
            output_path = os.path.join(self._project_dir, f"{self._book_title.replace(' ', '_')}.m4a")
            self._concat_wavs(wav_files, output_path)
            return output_path
        return None

    def _concat_wavs(self, wav_files, output_path):
        """Simple WAV concatenation using sox"""
        import subprocess
        cmd = ["sox"] + wav_files + [output_path]
        subprocess.run(cmd, check=False)

    def _get_engine(self):
        return self.engine

# Global instance
app_instance = None

def create_app():
    global app_instance
    if app_instance is None:
        app_instance = AudiobookGUI()
    return app_instance.build()

app = create_app()
