import configparser
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import wave
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import inference_webui as base

_whisper_model = None
EMOTIONS = ["neutral", "happy", "surprise", "annoyed", "sorrow", "relax"]
EMOTION_LABELS = {
    "neutral": "neutral / 通常",
    "happy": "happy / 嬉しい",
    "surprise": "surprise / 驚き",
    "annoyed": "annoyed / いら立ち",
    "sorrow": "sorrow / 悲しい",
    "relax": "relax / 落ち着き",
}
CONFIG_PATH = Path(ROOT_DIR) / "emotion_presets.txt"
SCRIPT_EXAMPLE = "[happy]おはよう。[relax]今日はゆっくり話そう。[surprise]えっ、本当に？"
DEFAULT_CONFIG_TEXT = """# GPT-SoVITS emotion presets\n# neutral は常に UI の speed / top_p / temperature を使います。\n# 他の感情はこのファイルの絶対値を使います。\n\n[neutral]\npitch_shift = 0.0\nspeed = ui\ntop_p = ui\ntemperature = ui\nvolume_gain_db = 0.0\n\n[happy]\npitch_shift = 1.0\nspeed = 1.10\ntop_p = 1.00\ntemperature = 1.00\nvolume_gain_db = 0.5\n\n[surprise]\npitch_shift = 0.5\nspeed = 1.20\ntop_p = 1.00\ntemperature = 1.00\nvolume_gain_db = 0.4\n\n[annoyed]\npitch_shift = 0.5\nspeed = 1.20\ntop_p = 0.70\ntemperature = 0.90\nvolume_gain_db = 1.2\n\n[sorrow]\npitch_shift = -1.0\nspeed = 0.85\ntop_p = 0.90\ntemperature = 0.80\nvolume_gain_db = -0.6\n\n[relax]\npitch_shift = -0.5\nspeed = 0.90\ntop_p = 0.90\ntemperature = 0.85\nvolume_gain_db = -0.2\n"""


def ensure_emotion_config_file():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return CONFIG_PATH


def load_emotion_config():
    ensure_emotion_config_file()
    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH, encoding="utf-8")
    config = {}
    for emotion in EMOTIONS:
        if emotion not in parser:
            raise ValueError(f"設定ファイルに [{emotion}] セクションがありません: {CONFIG_PATH}")
        section = parser[emotion]
        speed_value = section.get("speed", "ui").strip()
        top_p_value = section.get("top_p", "ui").strip()
        temperature_value = section.get("temperature", "ui").strip()
        config[emotion] = {
            "pitch_shift": float(section.get("pitch_shift", "0.0")),
            "speed": speed_value if speed_value == "ui" else float(speed_value),
            "top_p": top_p_value if top_p_value == "ui" else float(top_p_value),
            "temperature": temperature_value if temperature_value == "ui" else float(temperature_value),
            "volume_gain_db": float(section.get("volume_gain_db", "0.0")),
        }
    return config


def summarize_manifest_for_ui(manifest):
    if not manifest:
        return {}
    refs = manifest.get("generated_refs") or {}
    return {
        "session": Path(manifest.get("session_dir", "")).name,
        "config": Path(manifest.get("config_path", "")).name if manifest.get("config_path") else "",
        "source_ref": Path(manifest.get("source_ref", "")).name if manifest.get("source_ref") else "",
        "refs": {emotion: Path(path).name for emotion, path in refs.items()},
        "segment_count": len(manifest.get("segments") or []),
        "final_output": Path(manifest.get("final_output", "")).name if manifest.get("final_output") else "",
    }


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_name = os.environ.get("reference_whisper_model", "turbo")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _whisper_model


def transcribe_reference_audio(audio_path):
    if not audio_path:
        return (
            gr.update(),
            gr.update(value=base.i18n("日文")),
            gr.update(value="参照音声をアップロードすると、ここにWhisperの自動書き起こしを表示します。"),
            gr.update(value=False),
        )

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        transcript = "".join(segment.text.strip() for segment in segments).strip()
        if transcript:
            status = (
                f"Whisperで自動書き起こししました。検出言語: {info.language} "
                f"(confidence={info.language_probability:.2f})"
            )
        else:
            status = "Whisperで音声を検出できませんでした。参照音声の内容を手動で入力してください。"
        return (
            gr.update(value=transcript),
            gr.update(value=base.i18n("日文")),
            gr.update(value=status),
            gr.update(value=False),
        )
    except Exception as exc:
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(value=base.i18n("日文")),
            gr.update(value=f"Whisperの自動書き起こしに失敗しました: {exc}"),
            gr.update(value=False),
        )


def open_emotion_config():
    config_path = ensure_emotion_config_file()
    os.startfile(str(config_path))
    return f"設定ファイルを開きました: {config_path}"


def ensure_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")
    return ffmpeg_path


def get_sample_rate(audio_path):
    try:
        return int(torchaudio.info(audio_path).sample_rate)
    except Exception:
        return 44100


def build_atempo_filters(target_tempo):
    filters = []
    remaining = target_tempo
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return filters


def build_emotion_ref(audio_path, emotion, out_path, preset):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if emotion == "neutral":
        shutil.copy2(audio_path, out_path)
        return

    ffmpeg_path = ensure_ffmpeg()
    sample_rate = get_sample_rate(audio_path)
    pitch_ratio = math.pow(2.0, preset["pitch_shift"] / 12.0)
    speed_value = float(preset["speed"])
    tempo_correction = max(0.5, min(2.0, speed_value / pitch_ratio))
    volume_ratio = math.pow(10.0, preset["volume_gain_db"] / 20.0)
    filter_chain = [
        f"asetrate={sample_rate}*{pitch_ratio:.8f}",
        f"aresample={sample_rate}",
        *build_atempo_filters(tempo_correction),
        f"volume={volume_ratio:.6f}",
    ]
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-filter:a",
        ",".join(filter_chain),
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{emotion} 参照音声の生成に失敗しました。")


def make_session_dir():
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    session_dir = Path(ROOT_DIR) / "outputs" / "emotion_sessions" / session_id
    (session_dir / "refs").mkdir(parents=True, exist_ok=True)
    (session_dir / "segments").mkdir(parents=True, exist_ok=True)
    return session_dir


def empty_preview_updates():
    return [gr.update(value=None) for _ in EMOTIONS]


def write_wav_file(path, sample_rate, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.int16)
    if audio.ndim > 1:
        audio = audio.squeeze()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(audio.tobytes())


def parse_tagged_script(script):
    text = (script or "").strip()
    if not text:
        raise ValueError("読み上げ台本を入力してください。")
    token_re = re.compile(r"\[([A-Za-z0-9_]+)\]")
    segments = []
    current_emotion = "neutral"
    cursor = 0
    for match in token_re.finditer(text):
        chunk = text[cursor : match.start()].strip()
        if chunk:
            segments.append({"emotion": current_emotion, "text": chunk})
        current_emotion = match.group(1).lower()
        if current_emotion not in EMOTIONS:
            raise ValueError(f"未対応の感情タグです: [{current_emotion}]")
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        segments.append({"emotion": current_emotion, "text": tail})
    if not segments:
        raise ValueError("タグの後ろに読み上げ本文がありません。")
    return segments


def build_manifest(session_dir, source_ref, refs, emotion_config):
    return {
        "session_dir": str(session_dir),
        "config_path": str(CONFIG_PATH),
        "source_ref": source_ref,
        "generated_refs": refs,
        "emotion_config": emotion_config,
        "segments": [],
        "final_output": None,
    }


def get_infer_values(emotion, emotion_config, ui_speed, ui_top_p, ui_temperature):
    preset = emotion_config[emotion]
    if emotion == "neutral":
        return {
            "speed": float(ui_speed),
            "top_p": float(ui_top_p),
            "temperature": float(ui_temperature),
        }
    return {
        "speed": float(preset["speed"]),
        "top_p": float(preset["top_p"]),
        "temperature": float(preset["temperature"]),
    }


def split_emotion_refs(audio_path):
    if not audio_path:
        return (
            None,
            "参照音声をアップロードしてください。",
            gr.update(value={"error": "reference audio missing"}),
            *empty_preview_updates(),
        )
    try:
        ensure_ffmpeg()
        emotion_config = load_emotion_config()
        session_dir = make_session_dir()
        refs = {}
        for emotion in EMOTIONS:
            out_path = session_dir / "refs" / f"{emotion}.wav"
            build_emotion_ref(audio_path, emotion, out_path, emotion_config[emotion])
            refs[emotion] = str(out_path)
        manifest = build_manifest(session_dir, audio_path, refs, emotion_config)
        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        status = f"感情参照音声を生成しました: {session_dir}"
        return (
            manifest,
            status,
            gr.update(value=summarize_manifest_for_ui(manifest)),
            *[gr.update(value=refs[emotion]) for emotion in EMOTIONS],
        )
    except Exception as exc:
        traceback.print_exc()
        return (
            None,
            f"感情参照音声の生成に失敗しました: {exc}",
            gr.update(value={"error": str(exc)[:200]}),
            *empty_preview_updates(),
        )


def run_base_tts(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut,
    top_k,
    top_p,
    temperature,
    ref_text_free,
    speed,
    if_freeze,
    sample_steps,
    if_sr,
    pause_second,
):
    result = None
    generator = base.get_tts_wav(
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        how_to_cut,
        top_k,
        top_p,
        temperature,
        ref_text_free,
        speed,
        if_freeze,
        None,
        sample_steps,
        if_sr,
        pause_second,
    )
    for result in generator:
        pass
    if result is None:
        raise RuntimeError("推論結果を取得できませんでした。")
    return result


def infer_emotion_script(
    session_state,
    prompt_text,
    prompt_language,
    script_text,
    text_language,
    how_to_cut,
    top_k,
    ui_top_p,
    ui_temperature,
    ref_text_free,
    ui_speed,
    if_freeze,
    sample_steps,
    if_sr,
    pause_second,
):
    if not session_state or not session_state.get("generated_refs"):
        return None, [], gr.update(value={"error": "emotion refs missing"}), "先に感情分割を実行してください。"
    if not prompt_text or not prompt_text.strip():
        return None, [], gr.update(value={"error": "prompt text missing"}), "参照音声の書き起こしを入力してください。"

    try:
        emotion_config = load_emotion_config()
        segments = parse_tagged_script(script_text)
        refs = session_state["generated_refs"]
        session_dir = Path(session_state["session_dir"])
        segment_rows = []
        rendered_segments = []
        final_sr = None
        for index, segment in enumerate(segments):
            emotion = segment["emotion"]
            emotion_ref = refs.get(emotion) or refs["neutral"]
            infer_values = get_infer_values(emotion, emotion_config, ui_speed, ui_top_p, ui_temperature)
            sr, audio = run_base_tts(
                emotion_ref,
                prompt_text,
                prompt_language,
                segment["text"],
                text_language,
                how_to_cut,
                top_k,
                infer_values["top_p"],
                infer_values["temperature"],
                ref_text_free,
                infer_values["speed"],
                if_freeze,
                sample_steps,
                if_sr,
                pause_second,
            )
            final_sr = sr
            segment_path = session_dir / "segments" / f"{index:03d}_{emotion}.wav"
            write_wav_file(segment_path, sr, audio)
            rendered_segments.append(np.asarray(audio, dtype=np.int16))
            segment_rows.append(
                [
                    index,
                    emotion,
                    f"{infer_values['speed']:.2f}",
                    f"{infer_values['top_p']:.2f}",
                    f"{infer_values['temperature']:.2f}",
                    segment["text"],
                    emotion_ref,
                    str(segment_path),
                ]
            )

        final_audio = np.concatenate(rendered_segments) if rendered_segments else np.zeros(0, dtype=np.int16)
        final_path = session_dir / "final.wav"
        write_wav_file(final_path, final_sr, final_audio)
        manifest = dict(session_state)
        manifest["emotion_config"] = emotion_config
        manifest["segments"] = [
            {
                "index": row[0],
                "emotion": row[1],
                "speed": row[2],
                "top_p": row[3],
                "temperature": row[4],
                "text": row[5],
                "ref_wav": row[6],
                "output_wav": row[7],
            }
            for row in segment_rows
        ]
        manifest["final_output"] = str(final_path)
        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        status = f"{len(segment_rows)} 区間を推論し、最終音声を保存しました: {final_path}"
        return (final_sr, final_audio), segment_rows, gr.update(value=summarize_manifest_for_ui(manifest)), status
    except Exception as exc:
        traceback.print_exc()
        return None, [], gr.update(value={"error": str(exc)[:200]}), f"感情タグ推論に失敗しました: {exc}"


ensure_emotion_config_file()

with gr.Blocks(title="GPT-SoVITS 1C Emotion Japanese Inference", analytics_enabled=False, js=base.js, css=base.css) as app:
    emotion_state = gr.State(value=None)
    gr.HTML(
        base.top_html.format(
            "GPT-SoVITS 1C 感情推論 UI。参照音声から感情別参照 WAV を生成し、[happy] 形式のタグ付き台本を連続推論します。"
        ),
        elem_classes="markdown",
    )
    with gr.Group():
        gr.Markdown(base.html_center("モデル選択", "h3"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label="GPTモデル候補",
                choices=sorted(base.GPT_names, key=base.custom_sort_key),
                value=base.gpt_path,
                interactive=True,
                scale=14,
            )
            SoVITS_dropdown = gr.Dropdown(
                label="SoVITSモデル候補",
                choices=sorted(base.SoVITS_names, key=base.custom_sort_key),
                value=base.sovits_path,
                interactive=True,
                scale=14,
            )
            refresh_button = gr.Button("モデル一覧を更新", variant="primary", scale=14)
            refresh_button.click(fn=base.change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

        gr.Markdown(base.html_center("参照音声と感情分割", "h3"))
        with gr.Row():
            inp_ref = gr.Audio(label="参照音声をアップロード (3〜10秒推奨)", type="filepath", scale=13)
            with gr.Column(scale=13):
                ref_text_free = gr.Checkbox(
                    label=base.i18n("开启无参考文本模式。不填参考文本亦相当于开启。")
                    + base.i18n("v3暂不支持该模式，使用了会报错。"),
                    value=False,
                    interactive=True if base.model_version not in base.v3v4set else False,
                    show_label=True,
                    scale=1,
                )
                gr.Markdown(
                    base.html_left(
                        "参照音声は Whisper で自動書き起こしされます。感情設定は emotion_presets.txt から読み込みます。"
                    )
                )
                prompt_text = gr.Textbox(
                    label="参照音声の書き起こし",
                    value="",
                    lines=5,
                    max_lines=5,
                    scale=1,
                    placeholder="参照音声をアップロードすると自動入力されます。",
                )
                whisper_status = gr.Textbox(
                    label="Whisperステータス",
                    value="参照音声をアップロードすると、ここにWhisperの自動書き起こしを表示します。",
                    interactive=False,
                    lines=2,
                    max_lines=4,
                )
                with gr.Row():
                    transcribe_button = gr.Button("Whisperで再書き起こし", variant="secondary")
                    split_button = gr.Button("感情分割", variant="primary")
                    open_config_button = gr.Button("Open Config", variant="secondary")
                split_status = gr.Textbox(label="感情分割ステータス", interactive=False, lines=3)
            with gr.Column(scale=14):
                prompt_language = gr.Dropdown(
                    label="参照音声の言語",
                    choices=list(base.dict_language.keys()),
                    value=base.i18n("日文"),
                )
                inp_refs = (
                    gr.File(
                        label="追加参照音声 (この画面では未使用、既存UI互換のため保持)",
                        file_count="multiple",
                        visible=False,
                    )
                    if base.model_version not in base.v3v4set
                    else gr.File(label="追加参照音声", file_count="multiple", visible=False)
                )
                sample_steps = (
                    gr.Radio(
                        label=base.i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        value=32 if base.model_version == "v3" else 8,
                        choices=[4, 8, 16, 32, 64, 128] if base.model_version == "v3" else [4, 8, 16, 32],
                        visible=True,
                    )
                    if base.model_version in base.v3v4set
                    else gr.Radio(
                        label=base.i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        choices=[4, 8, 16, 32, 64, 128] if base.model_version == "v3" else [4, 8, 16, 32],
                        visible=False,
                        value=32 if base.model_version == "v3" else 8,
                    )
                )
                if_sr_Checkbox = gr.Checkbox(
                    label=base.i18n("v3输出如果觉得闷可以试试开超分"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    visible=False if base.model_version != "v3" else True,
                )
                manifest_view = gr.JSON(label="セッション情報")

        with gr.Row():
            preview_components = []
            for emotion in EMOTIONS:
                preview_components.append(gr.Audio(label=f"{EMOTION_LABELS[emotion]} 参照音声", type="filepath"))

        gr.Markdown(base.html_center("感情タグ付き台本", "h3"))
        gr.Markdown(
            base.html_left(
                "タグ記法: `[happy]こんにちは。[sorrow]少し寂しい。` のように書きます。neutral は UI の値、他の感情は emotion_presets.txt の絶対値を使います。"
            )
        )
        with gr.Row():
            with gr.Column(scale=13):
                text = gr.Textbox(label="読み上げる台本", value=SCRIPT_EXAMPLE, lines=22, max_lines=22)
            with gr.Column(scale=7):
                text_language = gr.Dropdown(
                    label="テキストの言語",
                    choices=list(base.dict_language.keys()),
                    value=base.i18n("日文"),
                    scale=1,
                )
                how_to_cut = gr.Dropdown(
                    label="テキスト分割方法",
                    choices=[
                        base.i18n("不切"),
                        base.i18n("凑四句一切"),
                        base.i18n("凑50字一切"),
                        base.i18n("按中文句号。切"),
                        base.i18n("按英文句号.切"),
                        base.i18n("按标点符号切"),
                    ],
                    value=base.i18n("按标点符号切"),
                    interactive=True,
                    scale=1,
                )
                if_freeze = gr.Checkbox(
                    label=base.i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    scale=1,
                )
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.6,
                        maximum=1.65,
                        step=0.05,
                        label="neutral の話速",
                        value=1.0,
                        interactive=True,
                        scale=1,
                    )
                    pause_second_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label="文ごとの間 (秒)",
                        value=0.3,
                        interactive=True,
                        scale=1,
                    )
                gr.Markdown(base.html_center("neutral 用 GPT サンプリング設定"))
                top_k = gr.Slider(minimum=1, maximum=100, step=1, label="top_k (全感情共通)", value=15, interactive=True, scale=1)
                top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label="neutral の top_p", value=0.8, interactive=True, scale=1)
                temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="neutral の temperature", value=0.8, interactive=True, scale=1)

        with gr.Row():
            inference_button = gr.Button(value="感情タグで音声を合成", variant="primary", size="lg", scale=25)
            output = gr.Audio(label="最終生成音声", scale=14)

        segment_table = gr.Dataframe(
            headers=["index", "emotion", "speed", "top_p", "temperature", "text", "ref_wav", "output_wav"],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="セグメント一覧",
        )
        inference_status = gr.Textbox(label="推論ステータス", interactive=False, lines=3)

        inference_button.click(
            infer_emotion_script,
            [
                emotion_state,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                ref_text_free,
                speed,
                if_freeze,
                sample_steps,
                if_sr_Checkbox,
                pause_second_slider,
            ],
            [output, segment_table, manifest_view, inference_status],
        )
        split_button.click(
            split_emotion_refs,
            [inp_ref],
            [emotion_state, split_status, manifest_view, *preview_components],
        )
        open_config_button.click(open_emotion_config, [], [split_status])
        SoVITS_dropdown.change(
            base.change_sovits_weights,
            [SoVITS_dropdown, prompt_language, text_language],
            [
                prompt_language,
                text_language,
                prompt_text,
                prompt_language,
                text,
                text_language,
                sample_steps,
                inp_refs,
                ref_text_free,
                if_sr_Checkbox,
                inference_button,
            ],
        )
        GPT_dropdown.change(base.change_gpt_weights, [GPT_dropdown], [])
        inp_ref.upload(
            transcribe_reference_audio,
            [inp_ref],
            [prompt_text, prompt_language, whisper_status, ref_text_free],
        )
        inp_ref.change(
            transcribe_reference_audio,
            [inp_ref],
            [prompt_text, prompt_language, whisper_status, ref_text_free],
        )
        transcribe_button.click(
            transcribe_reference_audio,
            [inp_ref],
            [prompt_text, prompt_language, whisper_status, ref_text_free],
        )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=base.is_share,
        server_port=base.infer_ttswebui,
    )
