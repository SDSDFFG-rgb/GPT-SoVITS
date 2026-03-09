import argparse
import os
import tempfile

import torchaudio
from moonshine_voice import Transcriber, get_model_for_language
from moonshine_voice.utils import load_wav_file


def prepare_audio_for_moonshine(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    peak = float(wav.abs().max()) if wav.numel() else 0.0
    if peak > 1e-6:
        wav = wav / peak * 0.8
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    torchaudio.save(temp_path, wav, sr, encoding='PCM_S', bits_per_sample=16)
    return temp_path, sr


def transcribe_streaming(transcriber, audio_data, sample_rate):
    transcriber.start()
    chunk_size = max(1, int(0.1 * sample_rate))
    for i in range(0, len(audio_data), chunk_size):
        transcriber.add_audio(audio_data[i:i + chunk_size], sample_rate)
    transcript = transcriber.stop()
    return ' '.join([line.text.strip() for line in transcript.lines if line.text.strip()]).strip()


def transcribe_one(audio_path, cache_dir):
    normalized_path = None
    try:
        if cache_dir:
            os.environ['MOONSHINE_VOICE_CACHE'] = cache_dir
        model_path, model_arch = get_model_for_language('ja')
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch, options={'max_tokens_per_second': 13.0})
        normalized_path, _ = prepare_audio_for_moonshine(audio_path)
        audio_data, sr = load_wav_file(normalized_path)
        transcript = transcriber.transcribe_without_streaming(audio_data, sr)
        text = ' '.join([line.text.strip() for line in transcript.lines if line.text.strip()]).strip()
        if text:
            return text
        text = transcribe_streaming(transcriber, audio_data, sr)
        if text:
            return text
        for pad_sec in [0.25, 0.5, 1.0]:
            silence = [0.0] * max(1, int(pad_sec * sr))
            padded = silence + audio_data + silence
            text = transcribe_streaming(transcriber, padded, sr)
            if text:
                return text
        return ''
    finally:
        if normalized_path and os.path.exists(normalized_path):
            try:
                os.remove(normalized_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-path', required=True)
    parser.add_argument('--cache-dir', default='')
    args = parser.parse_args()
    print(transcribe_one(args.audio_path, args.cache_dir), end='')


if __name__ == '__main__':
    main()
