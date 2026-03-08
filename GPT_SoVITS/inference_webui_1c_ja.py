import os
import sys
import traceback

import gradio as gr
import torch
from faster_whisper import WhisperModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import inference_webui as base

_whisper_model = None


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


with gr.Blocks(title="GPT-SoVITS 1C Japanese Inference", analytics_enabled=False, js=base.js, css=base.css) as app:
    gr.HTML(
        base.top_html.format(
            "GPT-SoVITS 1C 推論専用 UI。参照音声アップロード時に Whisper で自動書き起こしします。"
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

        gr.Markdown(base.html_center("参照音声", "h3"))
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
                        "アップロードした参照音声はWhisperで自動書き起こしされます。必要に応じて下のテキストを修正してください。"
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
                transcribe_button = gr.Button("Whisperで再書き起こし", variant="secondary")
            with gr.Column(scale=14):
                prompt_language = gr.Dropdown(
                    label="参照音声の言語",
                    choices=list(base.dict_language.keys()),
                    value=base.i18n("日文"),
                )
                inp_refs = (
                    gr.File(
                        label="任意: 複数の参照音声を追加して音色を平均化します。未入力なら左の単一参照音声のみを使います。",
                        file_count="multiple",
                    )
                    if base.model_version not in base.v3v4set
                    else gr.File(
                        label="任意: 複数の参照音声を追加して音色を平均化します。未入力なら左の単一参照音声のみを使います。",
                        file_count="multiple",
                        visible=False,
                    )
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

        gr.Markdown(base.html_center("読み上げテキスト", "h3"))
        with gr.Row():
            with gr.Column(scale=13):
                text = gr.Textbox(label="読み上げるテキスト", value="", lines=26, max_lines=26)
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
                gr.Markdown(value=base.html_center("話速"))
                if_freeze = gr.Checkbox(
                    label=base.i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    scale=1,
                )
                with gr.Row():
                    speed = gr.Slider(minimum=0.6, maximum=1.65, step=0.05, label="話速", value=1, interactive=True, scale=1)
                    pause_second_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label="文ごとの間 (秒)",
                        value=0.3,
                        interactive=True,
                        scale=1,
                    )
                gr.Markdown(base.html_center("GPTサンプリング設定"))
                top_k = gr.Slider(minimum=1, maximum=100, step=1, label="top_k", value=15, interactive=True, scale=1)
                top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label="top_p", value=1, interactive=True, scale=1)
                temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="temperature", value=1, interactive=True, scale=1)

        with gr.Row():
            inference_button = gr.Button(value="音声を合成", variant="primary", size="lg", scale=25)
            output = gr.Audio(label="生成音声", scale=14)

        inference_button.click(
            base.get_tts_wav,
            [
                inp_ref,
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
                inp_refs,
                sample_steps,
                if_sr_Checkbox,
                pause_second_slider,
            ],
            [output],
        )
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
