# AIGUIBook - AI Audiobook Generator

Transform EPUB ebooks into multi-voice audiobooks (M4A) with AI character voices, emotion detection, and voice acting.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/rykieffer/audiobook-ai-1fr.git
cd audiobook-ai-1fr

# Install dependencies
pip install -r requirements.txt
pip install faster-qwen3-tts

# Run the application
python main.py
```

## 📚 Project Overview

**AIGUIBook** is an AI-powered audiobook generator that transforms EPUB ebooks into professional-quality audiobooks with multiple character voices and emotion acting.

### Architecture

```
EPUB → Parse → Segment → LLM Analysis (characters/emotions) → Voice Design → Voice Clone + Acting → M4A
```

### Two Production Modes

1. **Single Narrator** (Recommended): One voice reads everything. The AI dynamically applies emotions (angry, sad, whisper, tense) to act out different characters. Fast and consistent.

2. **Full Ensemble**: Each character gets a unique AI-designed voice via VoiceDesign. Characters are instantly recognizable by their distinct vocal identity.

## 🔧 Key Technologies

- **TTS Engine**: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — 10x faster Qwen3-TTS using CUDA graph capture (no flash-attn required)
- **Voice Design**: Qwen3-TTS VoiceDesign model creates unique voices from text descriptions
- **Voice Clone + Acting**: Qwen3-TTS Base model clones voices and applies `instruct`-based emotion acting
- **Analysis**: Local LLM (LM Studio/Ollama) or OpenRouter detects speakers and emotions per segment
- **Output**: M4A with loudness normalization (LUFS -16), AAC encoding

## 📖 How It Works

### Step-by-Step Workflow

1. **EPUB Parsing**: Extract text, metadata, and structure from EPUB files
2. **Text Extraction**: Convert HTML to plain text
3. **Sentence Splitting**: Split text into TTS-friendly segments (NEW: Now handles French dialogue correctly!)
4. **Character Analysis**: Detect speakers, classify as dialogue/narration, assign emotions
5. **Voice Design**: Generate voice descriptions or clone character voices
6. **TTS Generation**: Create audio segments with emotion acting
7. **Assembly**: Combine segments into final audiobook

### French Dialogue Fix

**Problem**: Previously, French dialogue with em-dashes was not properly split:
```
— Comment vas-tu? demanda-t-il.
```

**Solution**: The new `_split_into_sentences()` method correctly handles:
- Em-dash dialogue lines
- Question marks followed by lowercase (French grammar)
- Mixed dialogue and narration

## 🎤 Voice Configuration

### Narration Voices
- `narrator_male`: Default male narrator
- `narrator_female`: Default female narrator
- Emotion: Neutral (configurable)

### Character Voices
- Automatically detected from dialogue
- LLM generates unique voice per character
- Emotion based on context (excited, sad, angry, etc.)

### Available Emotions
- calm, excited, angry, sad, whisper, tense, urgent, amused, contemptuous, surprised, neutral

## 📁 Project Structure

```
audiobook-ai-1fr/
├── audiobook_ai/
│   ├── core/
│   │   ├── epub_parser.py          # EPUB parsing with French dialogue fix
│   │   ├── config.py
│   │   ├── project.py
│   │   └── text_segmenter.py
│   ├── analysis/
│   │   ├── character_analyzer.py   # Voice design & emotion detection
│   │   └── __init__.py
│   ├── audio/
│   │   ├── assembly.py
│   │   └── validation.py
│   ├── tts/
│   │   ├── qwen_engine.py
│   │   └── voice_manager.py
│   └── __init__.py
├── cli.py
├── main.py
├── requirements.txt
├── pyproject.toml
├── README.md (this file)
└── tests/
```

## 🧪 Testing

```bash
# Run the test suite
python -m pytest tests/
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- LLM backend (lmstudio, ollama, openrouter)
- Model parameters
- Output directory
- Voice settings

## 📝 License

MIT License

## 💡 Contributing

Contributions are welcome! Please ensure:
- All tests pass
- French dialogue handling is preserved
- Voice design system remains backward compatible
