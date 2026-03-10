import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import psutil
import torch
import torchaudio
import yaml
from faster_whisper import WhisperModel
from moonshine_voice.utils import load_wav_file
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "ja_JP"
os.environ["language"] = language
i18n = I18nAuto(language=language)

from config import (
    GPU_INDEX,
    GPU_INFOS,
    GPT_weight_version2root,
    SoVITS_weight_version2root,
    exp_root,
    infer_device,
    is_half,
    pretrained_gpt_name,
    pretrained_sovits_name,
)

SESSION_ROOT = ROOT_DIR / "outputs" / "minute_training"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
TEMP_ROOT = ROOT_DIR / "TEMP" / "minute_training"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
SUPPORTED_VERSIONS = ["v1", "v2", "v4", "v2Pro", "v2ProPlus"]
DEFAULT_VERSION = os.environ.get("version", "v2Pro")
CURRENT_PROCESS = None
CURRENT_PROCESS_NAME = ""
WHISPER_MODEL = None
SESSION_KIND = '1min'


def sanitize_name(value):
    value = re.sub(r'[\\/:*?"<>|]+', '_', (value or '').strip())
    value = re.sub(r'\s+', '_', value)
    return value or 'minute_train'


def derive_speaker(save_name, speaker_name):
    return (speaker_name or '').strip() or sanitize_name(save_name)


def gpu_default():
    if infer_device.type == 'cuda' and infer_device.index is not None:
        return str(infer_device.index)
    return str(sorted(GPU_INDEX)[0]) if GPU_INDEX else '0'


def fix_gpu_number(value):
    try:
        value = int(str(value).strip())
        if value in GPU_INDEX:
            return value
    except Exception:
        pass
    return int(gpu_default())


def kill_proc_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        os.kill(parent.pid, signal.SIGTERM)
    except OSError:
        pass


def kill_process(pid):
    if platform.system() == 'Windows':
        subprocess.run(f'taskkill /t /f /pid {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)


def ensure_session(save_name, speaker_name, language_code, temporary=False):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    base_root = TEMP_ROOT if temporary else SESSION_ROOT
    base = base_root / f"{sid}_{SESSION_KIND}_{sanitize_name(save_name)}"
    for name in ['uploads', 'edited', 'dataset', 'transcripts']:
        (base / name).mkdir(parents=True, exist_ok=True)
    return {
        'session_dir': str(base),
        'save_name': sanitize_name(save_name),
        'speaker_name': derive_speaker(save_name, speaker_name),
        'language_code': language_code or 'ja',
        'clips': [],
        'dataset': {},
        'latest_weights': {},
        'session_kind': SESSION_KIND,
        'is_temp_session': temporary,
    }


def duration_of(path):
    audio, sr = torchaudio.load(path)
    if sr <= 0:
        raise ValueError(f'Invalid sample rate: {sr}')
    return round(audio.shape[1] / sr, 4)


def clip_choice(clip):
    return f"{clip['id']:03d} | {clip['name']} | {clip['duration']:.2f}s"


def find_clip(state, choice):
    clips = (state or {}).get('clips') or []
    if not clips:
        return None
    if not choice:
        return clips[0]
    clip_id = int(str(choice).split('|', 1)[0].strip())
    for clip in clips:
        if clip['id'] == clip_id:
            return clip
    return clips[0]


def clip_rows(state):
    rows = []
    for clip in (state or {}).get('clips', []):
        rows.append([clip['id'], '採用' if clip['keep'] else '除外', clip['name'], f"{clip['duration']:.2f}", clip['text']])
    return rows


def state_summary(state):
    if not state:
        return {}
    ds = state.get('dataset', {})
    return {
        'session': Path(state['session_dir']).name,
        'save_name': state['save_name'],
        'speaker_name': state['speaker_name'],
        'language': state['language_code'],
        'clip_count': len(state['clips']),
        'kept_count': sum(1 for c in state['clips'] if c['keep']),
        'list_path': ds.get('list_path', ''),
        'wav_dir': ds.get('wav_dir', ''),
        'latest_weights': state.get('latest_weights', {}),
    }


def editor_outputs(state, message=''):
    ds = (state or {}).get('dataset', {})
    return (
        state,
        message,
        clip_rows(state),
        state_summary(state),
        ds.get('list_path', ''),
        ds.get('wav_dir', ''),
    )



def session_kind_matches(session_dir):
    state_path = session_dir / 'session_state.json'
    fallback_path = session_dir / 'dataset' / 'session.json'
    target = state_path if state_path.exists() else fallback_path
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding='utf-8'))
            kind = data.get('session_kind')
            if kind:
                return kind == SESSION_KIND
            return 'sources' not in data
        except Exception:
            pass
    return f'_{SESSION_KIND}_' in session_dir.name


def session_label(session_dir):
    state_path = session_dir / 'session_state.json'
    fallback_path = session_dir / 'dataset' / 'session.json'
    target = state_path if state_path.exists() else fallback_path
    save_name = session_dir.name
    speaker_name = ''
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding='utf-8'))
            save_name = data.get('save_name') or save_name
            speaker_name = data.get('speaker_name') or ''
        except Exception:
            pass
    stamp = session_dir.name.split('_', 1)[0]
    if len(stamp) >= 13 and '-' in stamp:
        display_stamp = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
    else:
        display_stamp = stamp
    if speaker_name and speaker_name != save_name:
        return f'{save_name} / {speaker_name} / {display_stamp} | {session_dir.name}'
    return f'{save_name} / {display_stamp} | {session_dir.name}'


def session_name_from_choice(choice):
    if not choice:
        return ''
    text = str(choice)
    if '|' in text:
        return text.rsplit('|', 1)[-1].strip()
    return text.strip()
def session_options():
    sessions = []
    for session_dir in sorted(SESSION_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if session_dir.is_dir() and session_kind_matches(session_dir):
            sessions.append(session_label(session_dir))
    return sessions


def normalize_loaded_state(state):
    if not state:
        return None
    state.setdefault('save_name', 'minute_train_ja')
    state.setdefault('speaker_name', state.get('save_name', 'minute_train_ja'))
    state.setdefault('language_code', 'ja')
    state.setdefault('clips', [])
    state.setdefault('dataset', {})
    state.setdefault('latest_weights', {})
    state.setdefault('session_kind', SESSION_KIND)
    session_dir = Path(state['session_dir'])
    dataset_dir = session_dir / 'dataset'
    ds = state['dataset']
    list_path = dataset_dir / 'annotations.list'
    wav_dir = dataset_dir / 'audio'
    session_json = dataset_dir / 'session.json'
    if list_path.exists():
        ds['list_path'] = str(list_path)
    if wav_dir.exists():
        ds['wav_dir'] = str(wav_dir)
    if session_json.exists():
        ds['session_json'] = str(session_json)
    missing_trimmed = 0
    for clip in state['clips']:
        clip.setdefault('trimmed_path', '')
        clip.setdefault('text', '')
        clip.setdefault('keep', True)
        clip.setdefault('trim_start', 0.0)
        clip.setdefault('trim_end', clip.get('duration', 0.0))
        trimmed_path = clip.get('trimmed_path') or ''
        if trimmed_path and not Path(trimmed_path).exists():
            clip['trimmed_path'] = ''
            missing_trimmed += 1
    return state, missing_trimmed


def refresh_session_list(selected_name=None):
    choices = session_options()
    value = selected_name if selected_name in choices else (choices[0] if choices else None)
    message = f'保存済みセッションを {len(choices)} 件見つけました。' if choices else '保存済みセッションはまだありません。'
    return gr.update(choices=choices, value=value), message


def load_saved_session(session_name):
    session_name = session_name_from_choice(session_name)
    choices = session_options()
    if not session_name:
        return (
            None,
            '読み込むセッションを選んでください。',
            [],
            {},
            '',
            '',
            'minute_train_ja',
            'minute_train_ja',
            'ja',
            gr.update(choices=choices, value=None),
        )
    session_dir = SESSION_ROOT / session_name
    state_path = session_dir / 'session_state.json'
    fallback_path = session_dir / 'dataset' / 'session.json'
    target = state_path if state_path.exists() else fallback_path
    if not target.exists():
        return (
            None,
            f'セッション情報が見つかりません: {session_name}',
            [],
            {},
            '',
            '',
            'minute_train_ja',
            'minute_train_ja',
            'ja',
            gr.update(choices=choices, value=session_name if session_name in choices else None),
        )
    state = __import__('json').loads(target.read_text(encoding='utf-8'))
    state, missing_trimmed = normalize_loaded_state(state)
    message = f'セッションを読み込みました: {session_name}'
    if missing_trimmed:
        message += f'。一時トリム音声 {missing_trimmed} 件は元ファイル参照に戻しています。'
    ds = state.get('dataset', {})
    return (
        state,
        message,
        clip_rows(state),
        state_summary(state),
        ds.get('list_path', ''),
        ds.get('wav_dir', ''),
        state.get('save_name', 'minute_train_ja'),
        state.get('speaker_name', state.get('save_name', 'minute_train_ja')),
        state.get('language_code', 'ja'),
        gr.update(choices=choices, value=session_name),
    )


def delete_saved_session(session_name):
    session_name = session_name_from_choice(session_name)
    choices_before = session_options()
    if not session_name:
        return gr.update(choices=choices_before, value=None), '削除するセッションを選んでください。'
    session_dir = SESSION_ROOT / session_name
    if not session_dir.exists():
        choices_after = session_options()
        return gr.update(choices=choices_after, value=(choices_after[0] if choices_after else None)), f'セッションが見つかりません: {session_name}'
    shutil.rmtree(session_dir, ignore_errors=True)
    choices_after = session_options()
    return gr.update(choices=choices_after, value=(choices_after[0] if choices_after else None)), f'セッションを削除しました: {session_name}'


def clone_for_new_session(value, old_session_dir, new_session_dir):
    if isinstance(value, dict):
        return {k: clone_for_new_session(v, old_session_dir, new_session_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_for_new_session(v, old_session_dir, new_session_dir) for v in value]
    if isinstance(value, str) and old_session_dir in value:
        return value.replace(old_session_dir, new_session_dir)
    return value



def materialize_session(state, save_name, speaker_name, language_code):
    if not state or not state.get('session_dir'):
        return state, False
    if not state.get('is_temp_session'):
        return state, False
    new_state = ensure_session(save_name, speaker_name, language_code, temporary=False)
    old_dir = Path(state['session_dir'])
    new_dir = Path(new_state['session_dir'])
    for child in old_dir.iterdir():
        target = new_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    migrated = clone_for_new_session(json.loads(json.dumps(state, ensure_ascii=False)), str(old_dir), str(new_dir))
    migrated['session_dir'] = str(new_dir)
    migrated['save_name'] = sanitize_name(save_name)
    migrated['speaker_name'] = derive_speaker(save_name, speaker_name)
    migrated['language_code'] = (language_code or 'ja').strip() or 'ja'
    migrated['session_kind'] = SESSION_KIND
    migrated['is_temp_session'] = False
    shutil.rmtree(old_dir, ignore_errors=True)
    save_state(migrated)
    return migrated, True


def clone_current_session(state, save_name, speaker_name, language_code):
    if not state or not state.get('session_dir'):
        return None, '先に複製元のセッションを読み込んでください。', [], {}, '', '', gr.update(choices=session_options(), value=None)
    new_state = ensure_session(save_name, speaker_name, language_code, temporary=False)
    old_dir = Path(state['session_dir'])
    new_dir = Path(new_state['session_dir'])
    for folder in ['uploads', 'edited', 'transcripts', 'dataset']:
        src_dir = old_dir / folder
        dst_dir = new_dir / folder
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    cloned = clone_for_new_session(json.loads(json.dumps(state, ensure_ascii=False)), str(old_dir), str(new_dir))
    cloned['session_dir'] = str(new_dir)
    cloned['save_name'] = sanitize_name(save_name)
    cloned['speaker_name'] = derive_speaker(save_name, speaker_name)
    cloned['language_code'] = (language_code or 'ja').strip() or 'ja'
    cloned['session_kind'] = SESSION_KIND
    if cloned.get('dataset', {}).get('list_path'):
        list_path = Path(cloned['dataset']['list_path'])
        if list_path.exists():
            rows = []
            for row in list_path.read_text(encoding='utf-8').splitlines():
                if not row.strip():
                    continue
                parts = row.split('|', 3)
                if len(parts) == 4:
                    parts[1] = cloned['speaker_name']
                    parts[2] = cloned['language_code']
                    rows.append('|'.join(parts))
                else:
                    rows.append(row)
            list_path.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
    (Path(cloned['session_dir']) / 'session_state.json').write_text(json.dumps(cloned, ensure_ascii=False, indent=2), encoding='utf-8')
    choices = session_options()
    selected = next((choice for choice in choices if choice.endswith('| ' + new_dir.name)), None)
    ds = cloned.get('dataset', {})
    return cloned, f'新しい 1分学習セッションとして保存しました: {new_dir.name}', clip_rows(cloned), state_summary(cloned), ds.get('list_path', ''), ds.get('wav_dir', ''), gr.update(choices=choices, value=selected)
def load_files(files, save_name, speaker_name, language_code):
    if not files:
        return editor_outputs(None, '音声ファイルをドロップしてください。')
    state = ensure_session(save_name, speaker_name, language_code, temporary=True)
    upload_dir = Path(state['session_dir']) / 'uploads'
    clips = []
    for i, file_obj in enumerate(files):
        src = Path(file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
        ext = src.suffix or '.wav'
        dst = upload_dir / f'{i:03d}{ext}'
        shutil.copy2(src, dst)
        dur = duration_of(str(dst))
        clips.append({'id': i, 'name': src.name, 'source_path': str(dst), 'trimmed_path': '', 'duration': dur, 'trim_start': 0.0, 'trim_end': dur, 'text': '', 'keep': True})
    state['clips'] = clips
    return editor_outputs(state, f'{len(clips)} 個の音声を読み込みました。Whisper で書き起こしを実行してください。')


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        model_name = os.environ.get('reference_whisper_model', 'turbo')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return WHISPER_MODEL


def prepare_whisper():
    get_whisper_model()
    model_name = os.environ.get('reference_whisper_model', 'turbo')
    return f'Whisper モデルを準備しました: {model_name}'


def transcribe_clip_with_whisper(audio_path):
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language='ja',
        task='transcribe',
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = ''.join(segment.text.strip() for segment in segments).strip()
    return text, info.language if info else 'ja', getattr(info, 'language_probability', 0.0)


def transcribe_all(state):
    if not state or not state.get('clips'):
        return editor_outputs(state, '先に音声を読み込んでください。')
    try:
        transcript_dir = Path(state['session_dir']) / 'transcripts'
        transcript_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        success_count = 0
        empty_count = 0
        for clip in state['clips']:
            target = clip['trimmed_path'] or clip['source_path']
            text, language_name, confidence = transcribe_clip_with_whisper(target)
            clip['text'] = text
            transcript_path = transcript_dir / f"{clip['id']:03d}.txt"
            transcript_path.write_text(text + ('\n' if text else ''), encoding='utf-8')
            preview = text if text else '<<空文字>>'
            if len(preview) > 60:
                preview = preview[:60] + '...'
            summaries.append(f"- {clip['name']}: {preview}")
            if text:
                success_count += 1
            else:
                empty_count += 1
        session_state_path = Path(state['session_dir']) / 'session_state.json'
        session_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        summary_text = '\n'.join(summaries[:12])
        if len(summaries) > 12:
            summary_text += f"\n- 他 {len(summaries) - 12} 件"
        message = (
            f"Whisper 書き起こし完了: {success_count}/{len(state['clips'])} 件。空文字 {empty_count} 件。  \\n"
            f"保存先: {transcript_dir}  \\n"
            f"{summary_text}"
        )
        return editor_outputs(state, message)
    except Exception as exc:
        traceback.print_exc()
        return editor_outputs(state, f'Whisper 書き起こしに失敗しました: {exc}')


def save_clip(state, choice, trim_start, trim_end, text, keep):
    if not state or not state.get('clips'):
        return editor_outputs(state, '先に音声を読み込んでください。')
    clip = find_clip(state, choice)
    edited = Path(state['session_dir']) / 'edited' / f"{clip['id']:03d}.wav"
    audio, sr = torchaudio.load(clip['source_path'])
    start_frame = max(0, int(float(trim_start) * sr))
    end_frame = min(audio.shape[1], int(float(trim_end) * sr))
    if end_frame <= start_frame:
        return editor_outputs(state, '終了位置は開始位置より後ろにしてください。')
    torchaudio.save(str(edited), audio[:, start_frame:end_frame], sr)
    clip['trim_start'] = round(float(trim_start), 4)
    clip['trim_end'] = round(float(trim_end), 4)
    clip['trimmed_path'] = str(edited)
    clip['text'] = (text or '').strip()
    clip['keep'] = bool(keep)
    return editor_outputs(state, f"{clip['name']} の編集を保存しました。")


def save_editor_rows(state, *values):
    if not state or not state.get('clips'):
        return editor_outputs(state, '先に音声を読み込んでください。')
    expected = len(state['clips']) * 3
    if len(values) != expected:
        return state, '編集欄の数が一致しません。画面を再読み込みしてもう一度試してください。', clip_rows(state), state_summary(state)
    idx = 0
    for clip in state['clips']:
        audio_path = values[idx]
        text_value = values[idx + 1]
        keep_value = values[idx + 2]
        clip['trimmed_path'] = str(audio_path) if audio_path else clip.get('trimmed_path', '')
        clip['text'] = (text_value or '').strip()
        clip['keep'] = bool(keep_value)
        idx += 3
    session_state_path = Path(state['session_dir']) / 'session_state.json'
    session_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    kept = sum(1 for clip in state['clips'] if clip['keep'])
    filled = sum(1 for clip in state['clips'] if clip['text'])
    prefix = '新規保存しました。' if materialized else '一覧の編集を保存しました。'
    return state, f'{prefix} 採用 {kept} 件 / テキスト入力済み {filled} 件。', clip_rows(state), state_summary(state)


def export_dataset(state, save_name, speaker_name, language_code):
    if not state or not state.get('clips'):
        return editor_outputs(state, '先に音声を読み込んでください。')
    state['save_name'] = sanitize_name(save_name)
    state['speaker_name'] = derive_speaker(save_name, speaker_name)
    state['language_code'] = (language_code or 'ja').strip() or 'ja'
    dataset_dir = Path(state['session_dir']) / 'dataset'
    audio_dir = dataset_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in state['clips']:
        if not clip['keep'] or not clip['text'].strip():
            continue
        src = clip['trimmed_path'] or clip['source_path']
        dst = audio_dir / f"{clip['id']:03d}.wav"
        shutil.copy2(src, dst)
        rows.append(f"{dst.name}|{state['speaker_name']}|{state['language_code']}|{clip['text'].strip()}")
    if not rows:
        return editor_outputs(state, '採用クリップとテキストが必要です。')
    list_path = dataset_dir / 'annotations.list'
    list_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    session_json = dataset_dir / 'session.json'
    session_json.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    state['dataset'] = {'list_path': str(list_path), 'wav_dir': str(audio_dir), 'session_json': str(session_json)}
    return editor_outputs(state, f'学習フォーマットを生成しました。採用クリップ数: {len(rows)}')


def default_weight(mapping, version):
    if isinstance(mapping, dict):
        if version in mapping:
            return mapping[version]
        if 'v2Pro' in mapping and version == 'v2ProPlus':
            return mapping['v2Pro']
        return next(iter(mapping.values()))
    return mapping


def default_sovits_d_weight(version):
    if version == 'v2Pro':
        return 'GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth'
    if version == 'v2ProPlus':
        return 'GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth'
    return ''


def version_weight_defaults(version):
    return (
        default_weight(pretrained_gpt_name, version),
        default_weight(pretrained_sovits_name, version),
        default_sovits_d_weight(version),
    )


def version_guide(version):
    s1, s2g, s2d = version_weight_defaults(version)
    sovits_root = SoVITS_weight_version2root.get(version, '')
    gpt_root = GPT_weight_version2root.get(version, '')
    if version == 'v4':
        return (
            '### 現在のモデル構成\n'
            f'- 学習モデル版: `{version}`\n'
            f'- GPT の土台: `{s1}`\n'
            f'- SoVITS-G の土台: `{s2g}`\n'
            f'- SoVITS-D: 不使用\n'
            f'- 保存先: `{sovits_root}` / `{gpt_root}`\n\n'
            'v4 は `s1v3.ckpt + s2Gv4.pth` を使います。`SoVITS-D` は空欄のままで正常です。'
        )
    return (
        '### 現在のモデル構成\n'
        f'- 学習モデル版: `{version}`\n'
        f'- GPT の土台: `{s1}`\n'
        f'- SoVITS-G の土台: `{s2g}`\n'
        f'- SoVITS-D の土台: `{s2d or "なし"}`\n'
        f'- 保存先: `{sovits_root}` / `{gpt_root}`\n\n'
        '通常は下の重み欄を手で変更する必要はありません。学習モデル版を切り替えると自動で入ります。'
    )


def version_weight_updates(version, custom_pretrained=False):
    s1, s2g, s2d = version_weight_defaults(version)
    editable = bool(custom_pretrained)
    use_s2d = version in {'v2Pro', 'v2ProPlus'}
    return (
        gr.update(value=s1, interactive=editable),
        gr.update(value=s2g, interactive=editable),
        gr.update(value=s2d if use_s2d else '', interactive=editable, visible=use_s2d),
        version_guide(version),
    )


def custom_pretrained_updates(custom_pretrained, version):
    editable = bool(custom_pretrained)
    use_s2d = version in {'v2Pro', 'v2ProPlus'}
    return (
        gr.update(interactive=editable),
        gr.update(interactive=editable),
        gr.update(interactive=editable, visible=use_s2d),
        version_guide(version),
    )


def stream_process(command, env, process_name):
    global CURRENT_PROCESS, CURRENT_PROCESS_NAME
    merged_env = os.environ.copy()
    merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    lines = [f'[{process_name}] {command}']
    CURRENT_PROCESS_NAME = process_name
    CURRENT_PROCESS = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
        encoding='utf-8',
        errors='replace',
    )
    try:
        for line in CURRENT_PROCESS.stdout:
            lines.append(line.rstrip())
    finally:
        CURRENT_PROCESS.wait()
        code = CURRENT_PROCESS.returncode
        CURRENT_PROCESS = None
        CURRENT_PROCESS_NAME = ''
    if code != 0:
        raise RuntimeError('\n'.join(lines[-80:]))
    return '\n'.join(lines)


def ensure_dataset_ready(state):
    if not state or not state.get('dataset', {}).get('list_path'):
        raise ValueError('先に「学習フォーマットを生成」を実行してください。')
    return state['dataset']


def merge_text_part(opt_dir):
    parts = sorted(Path(opt_dir).glob('2-name2text-*.txt'))
    if not parts:
        return
    lines = []
    for part in parts:
        text = part.read_text(encoding='utf-8').strip()
        if text:
            lines.extend(text.splitlines())
        part.unlink()
    (Path(opt_dir) / '2-name2text.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def merge_semantic_part(opt_dir):
    parts = sorted(Path(opt_dir).glob('6-name2semantic-*.tsv'))
    if not parts:
        return
    lines = ['item_name\tsemantic_audio']
    for part in parts:
        text = part.read_text(encoding='utf-8').strip()
        if text:
            lines.extend([row for row in text.splitlines() if row.strip()])
        part.unlink()
    (Path(opt_dir) / '6-name2semantic.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_preprocess(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path):
    dataset = ensure_dataset_ready(state)
    exp_name = state['save_name']
    opt_dir = Path(exp_root) / exp_name
    opt_dir.mkdir(parents=True, exist_ok=True)
    env = {
        'inp_text': dataset['list_path'],
        'inp_wav_dir': dataset['wav_dir'],
        'exp_name': exp_name,
        'opt_dir': str(opt_dir),
        'bert_pretrained_dir': bert_dir,
        'cnhubert_base_dir': ssl_dir,
        'pretrained_s2G': pretrained_s2g,
        's2config_path': 'GPT_SoVITS/configs/s2.json' if version not in {'v2Pro', 'v2ProPlus'} else f'GPT_SoVITS/configs/s2{version}.json',
        'sv_path': sv_path,
        'i_part': '0',
        'all_parts': '1',
        '_CUDA_VISIBLE_DEVICES': str(fix_gpu_number(gpu_number)),
        'is_half': str(is_half),
        'version': version,
    }
    python = f'"{sys.executable}"'
    logs = []
    logs.append(stream_process(f'{python} -s GPT_SoVITS/prepare_datasets/1-get-text.py', env, '1A テキスト抽出'))
    merge_text_part(opt_dir)
    logs.append(stream_process(f'{python} -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', env, '1B 音声特徴抽出'))
    if 'Pro' in version:
        logs.append(stream_process(f'{python} -s GPT_SoVITS/prepare_datasets/2-get-sv.py', env, '1B 話者埋め込み抽出'))
    logs.append(stream_process(f'{python} -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', env, '1C セマンティック抽出'))
    merge_semantic_part(opt_dir)
    return '\n\n'.join(logs), str(opt_dir)


def run_sovits_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d):
    ensure_dataset_ready(state)
    exp_name = state['save_name']
    config_file = ROOT_DIR / ('GPT_SoVITS/configs/s2.json' if version not in {'v2Pro', 'v2ProPlus'} else f'GPT_SoVITS/configs/s2{version}.json')
    data = json.loads(config_file.read_text(encoding='utf-8'))
    s2_dir = Path(exp_root) / exp_name
    (s2_dir / f'logs_s2_{version}').mkdir(parents=True, exist_ok=True)
    if not is_half:
        data['train']['fp16_run'] = False
        batch_size = max(1, int(batch_size) // 2)
    data['train']['batch_size'] = int(batch_size)
    data['train']['epochs'] = int(total_epoch)
    data['train']['text_low_lr_rate'] = float(text_low_lr_rate)
    data['train']['pretrained_s2G'] = pretrained_s2g
    data['train']['pretrained_s2D'] = pretrained_s2d
    data['train']['if_save_latest'] = bool(if_save_latest)
    data['train']['if_save_every_weights'] = bool(if_save_every_weights)
    data['train']['save_every_epoch'] = int(save_every_epoch)
    data['train']['gpu_numbers'] = str(fix_gpu_number(gpu_number))
    data['train']['grad_ckpt'] = bool(if_grad_ckpt)
    data['train']['lora_rank'] = int(lora_rank)
    data['model']['version'] = version
    data['data']['exp_dir'] = str(s2_dir)
    data['s2_ckpt_dir'] = str(s2_dir)
    data['save_weight_dir'] = SoVITS_weight_version2root[version]
    Path(data['save_weight_dir']).mkdir(parents=True, exist_ok=True)
    data['name'] = exp_name
    data['version'] = version
    tmp_config = TEMP_ROOT / 'tmp_s2_1m.json'
    tmp_config.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    python = f'"{sys.executable}"'
    script = 'GPT_SoVITS/s2_train.py' if version in ['v1', 'v2', 'v2Pro', 'v2ProPlus'] else 'GPT_SoVITS/s2_train_v3_lora.py'
    env = {'_CUDA_VISIBLE_DEVICES': str(fix_gpu_number(gpu_number)), 'version': version, 'is_half': str(is_half)}
    return stream_process(f'{python} -s {script} --config "{tmp_config}"', env, 'SoVITS 学習')


def run_gpt_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1):
    ensure_dataset_ready(state)
    exp_name = state['save_name']
    config_file = ROOT_DIR / ('GPT_SoVITS/configs/s1longer.yaml' if version == 'v1' else 'GPT_SoVITS/configs/s1longer-v2.yaml')
    data = yaml.load(config_file.read_text(encoding='utf-8'), Loader=yaml.FullLoader)
    s1_dir = Path(exp_root) / exp_name
    (s1_dir / f'logs_s1_{version}').mkdir(parents=True, exist_ok=True)
    if not is_half:
        data['train']['precision'] = '32'
        batch_size = max(1, int(batch_size) // 2)
    data['train']['batch_size'] = int(batch_size)
    data['train']['epochs'] = int(total_epoch)
    data['pretrained_s1'] = pretrained_s1
    data['train']['save_every_n_epoch'] = int(save_every_epoch)
    data['train']['if_save_every_weights'] = bool(if_save_every_weights)
    data['train']['if_save_latest'] = bool(if_save_latest)
    data['train']['if_dpo'] = bool(if_dpo)
    data['train']['half_weights_save_dir'] = GPT_weight_version2root[version]
    Path(data['train']['half_weights_save_dir']).mkdir(parents=True, exist_ok=True)
    data['train']['exp_name'] = exp_name
    data['train_semantic_path'] = str(s1_dir / '6-name2semantic.tsv')
    data['train_phoneme_path'] = str(s1_dir / '2-name2text.txt')
    data['output_dir'] = str(s1_dir / f'logs_s1_{version}')
    tmp_config = TEMP_ROOT / 'tmp_s1_1m.yaml'
    tmp_config.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding='utf-8')
    python = f'"{sys.executable}"'
    env = {'_CUDA_VISIBLE_DEVICES': str(fix_gpu_number(gpu_number)), 'hz': '25hz', 'version': version, 'is_half': str(is_half)}
    return stream_process(f'{python} -s GPT_SoVITS/s1_train.py --config_file "{tmp_config}"', env, 'GPT 学習')


def latest_weight_map(version):
    result = {}
    sovits_root = Path(SoVITS_weight_version2root[version])
    gpt_root = Path(GPT_weight_version2root[version])
    sovits_files = sorted(sovits_root.rglob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True) if sovits_root.exists() else []
    gpt_files = sorted(gpt_root.rglob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True) if gpt_root.exists() else []
    result['sovits'] = str(sovits_files[0]) if sovits_files else ''
    result['gpt'] = str(gpt_files[0]) if gpt_files else ''
    return result


def refresh_latest_weights(state, version):
    state = state or ensure_session('minute_train', '', 'ja')
    state['latest_weights'] = latest_weight_map(version)
    return state, state_summary(state)


def stop_current_process():
    if CURRENT_PROCESS is None:
        return '停止中の処理はありません。'
    kill_process(CURRENT_PROCESS.pid)
    return f'{CURRENT_PROCESS_NAME or "処理"} を停止しました。'


def open_inference_ui():
    launcher = ROOT_DIR / 'go-1c-emotion-ja.bat'
    if not launcher.exists():
        return '推論 UI の起動ファイルが見つかりません。'
    if platform.system() == 'Windows':
        os.startfile(str(launcher))
        return '1C 感情推論 UI を別ウィンドウで開きました。'
    return f'Windows 以外では自動起動できません: {launcher}'


def preprocess_action(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path):
    try:
        log, _ = run_preprocess(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path)
        state['latest_weights'] = latest_weight_map(version)
        return state, '前処理が完了しました。SoVITS 学習または GPT 学習に進めます。', state_summary(state), log
    except Exception as exc:
        traceback.print_exc()
        return state, f'前処理に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def sovits_action(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d):
    try:
        log = run_sovits_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d)
        state['latest_weights'] = latest_weight_map(version)
        return state, 'SoVITS 学習が完了しました。', state_summary(state), log
    except Exception as exc:
        traceback.print_exc()
        return state, f'SoVITS 学習に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def gpt_action(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1):
    try:
        log = run_gpt_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1)
        state['latest_weights'] = latest_weight_map(version)
        return state, 'GPT 学習が完了しました。', state_summary(state), log
    except Exception as exc:
        traceback.print_exc()
        return state, f'GPT 学習に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def full_pipeline_action(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2d, gpt_batch_size, gpt_epoch, if_dpo, pretrained_s1):
    try:
        parts = []
        prep_log, _ = run_preprocess(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path)
        parts.append(prep_log)
        parts.append(run_sovits_training(state, version, gpu_number, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d))
        parts.append(run_gpt_training(state, version, gpu_number, gpt_batch_size, gpt_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1))
        state['latest_weights'] = latest_weight_map(version)
        return state, '前処理から学習まで完了しました。', state_summary(state), '\n\n'.join(parts)
    except Exception as exc:
        traceback.print_exc()
        return state, f'一括実行に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def version_defaults(version):
    return (
        default_weight(pretrained_sovits_name, version),
        default_weight(pretrained_gpt_name, version),
    )

DEFAULT_BERT_DIR = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
DEFAULT_SSL_DIR = 'GPT_SoVITS/pretrained_models/chinese-hubert-base'
DEFAULT_SV_PATH = 'GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt'
DEFAULT_PORT = int(os.environ.get('minute_train_webui', '9878'))


with gr.Blocks(title='GPT-SoVITS 1分学習 UI', css=css, js=js) as app:
    gr.HTML(top_html)
    state = gr.State(None)
    gr.Markdown(
        '# 1分学習 UI\n'
        '複数の短い音声クリップを読み込み、Whisper で書き起こし、必要なトリムとテキスト修正を行ってから GPT-SoVITS の学習を開始します。  \n'
        '推論はこの画面では行わず、学習後に既存の **1C 感情推論 UI** を開く構成です。'
    )
    with gr.Row():
        save_name = gr.Textbox(label='保存名', value='minute_train_ja', info='実験名と既定の話者名に使います。半角英数とアンダースコア推奨。')
        speaker_name = gr.Textbox(label='話者名', value='minute_train_ja', info='空欄なら保存名を使います。')
        language_code = gr.Dropdown(label='言語', choices=['ja', 'all_ja', 'auto'], value='ja', info='通常は ja のままで構いません。')
        version = gr.Dropdown(label='学習モデル版', choices=SUPPORTED_VERSIONS, value=DEFAULT_VERSION, info='通常は v2Pro か v4 を選びます。選ぶと下の重みは自動で切り替わります。')
    model_guide = gr.Markdown(version_guide(DEFAULT_VERSION))
    with gr.Row():
        uploads = gr.File(label='音声ファイルをまとめてドロップ', file_count='multiple', file_types=['audio'])
        with gr.Column():
            status = gr.Markdown('ここに進行状況が表示されます。')
            gr.Markdown('Whisper で自動書き起こしします。必要なら各行を手で修正してください。')
            with gr.Row():
                load_btn = gr.Button('音声を読み込む', variant='primary')
                whisper_btn = gr.Button('Whisper モデルを準備')
                transcribe_btn = gr.Button('Whisper で書き起こし')
            gr.Markdown('既に作業済みのセッションを再開したい場合は、下の一覧から読み込めます。')
            with gr.Row():
                session_picker = gr.Dropdown(label='保存済みセッション', choices=session_options(), value=None, allow_custom_value=False)
                refresh_sessions_btn = gr.Button('一覧を更新')
                load_session_btn = gr.Button('選択したセッションを読み込む')
                clone_session_btn = gr.Button('別セッションとして保存')
                delete_session_btn = gr.Button('選択したセッションを削除')
    clip_table = gr.Dataframe(headers=['ID', '状態', 'ファイル', '秒数', 'テキスト'], datatype=['number', 'str', 'str', 'str', 'str'], interactive=False, wrap=True, label='読み込んだファイル一覧')

    gr.Markdown('## クリップ編集\n読み込んだファイルを上から順に確認します。左の波形でそのままトリムし、右で文字を直してください。')
    editor_status = gr.Markdown('音声を読み込むと、ここに編集用の一覧が表示されます。')
    session_json = gr.JSON(label='セッション概要')

    @gr.render(inputs=state)
    def render_clip_editors(current_state):
        if not current_state or not current_state.get('clips'):
            gr.Markdown('まだ音声が読み込まれていません。')
            return
        row_inputs = []
        for clip in current_state['clips']:
            with gr.Row(equal_height=True):
                with gr.Column(scale=6, min_width=420):
                    audio_value = clip['trimmed_path'] or clip['source_path']
                    audio_editor = gr.Audio(
                        value=audio_value,
                        type='filepath',
                        label=f"{clip['name']} ({clip['duration']:.2f}s)",
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                        interactive=True
                    )
                with gr.Column(scale=5, min_width=300):
                    text_editor = gr.Textbox(
                        value=clip['text'],
                        lines=4,
                        label='書き起こし / 修正文',
                        info='誤字、句読点、読み違いをここで直します。'
                    )
                    keep_editor = gr.Checkbox(
                        value=clip['keep'],
                        label='このクリップを学習に使う'
                    )
                row_inputs.extend([audio_editor, text_editor, keep_editor])
            gr.Markdown('---')
        save_all_btn = gr.Button('現在のセッションに上書き保存', variant='primary')
        save_all_btn.click(save_editor_rows, [state] + row_inputs, [state, editor_status, clip_table, session_json])
        gr.Markdown('読み込み済みの既存セッションなら、そのまま同じ session_state.json と 	ranscripts/*.txt に上書き保存されます。')

    gr.Markdown('## 学習データ書き出し\nここで .list と学習用の WAV 一式を作ります。除外したクリップは出力されません。')
    with gr.Row():
        export_btn = gr.Button('学習フォーマットを生成', variant='primary')
        list_path = gr.Textbox(label='生成された annotations.list', interactive=False)
        wav_dir = gr.Textbox(label='学習用 WAV フォルダ', interactive=False)

    gr.Markdown('## 前処理と学習\n迷ったら「前処理から学習まで一括実行」を使ってください。細かく確認したいときは個別ボタンを使います。')
    with gr.Accordion('学習設定', open=False):
        with gr.Row():
            gpu_number = gr.Dropdown(label='GPU', choices=[str(i) for i in sorted(GPU_INDEX)], value=gpu_default(), info='学習に使う GPU を 1 枚選びます。')
            custom_pretrained = gr.Checkbox(label='事前学習重みを手動指定する', value=False, info='通常はオフのままで構いません。学習モデル版に応じて自動入力されます。')
        with gr.Row():
            pretrained_s1 = gr.Textbox(label='事前学習 GPT', value=default_weight(pretrained_gpt_name, DEFAULT_VERSION), interactive=False, info='通常は学習モデル版の切替に合わせて自動で入ります。')
            pretrained_s2g = gr.Textbox(label='事前学習 SoVITS-G', value=default_weight(pretrained_sovits_name, DEFAULT_VERSION), interactive=False, info='通常は学習モデル版の切替に合わせて自動で入ります。')
            pretrained_s2d = gr.Textbox(label='事前学習 SoVITS-D', value=default_sovits_d_weight(DEFAULT_VERSION), interactive=False, visible=DEFAULT_VERSION in {'v2Pro', 'v2ProPlus'}, info='v2Pro系でだけ使います。v4では表示されません。')
        gr.Markdown('上の 3 欄は通常そのままで大丈夫です。v4 は `s1v3.ckpt + s2Gv4.pth`、v2Pro は `s1v3.ckpt + s2Gv2Pro.pth + s2Dv2Pro.pth` が自動で入ります。')
        with gr.Row():
            bert_dir = gr.Textbox(label='BERT モデル', value=DEFAULT_BERT_DIR)
            ssl_dir = gr.Textbox(label='Hubert モデル', value=DEFAULT_SSL_DIR)
            sv_path = gr.Textbox(label='話者認識モデル', value=DEFAULT_SV_PATH)
        with gr.Row():
            sovits_batch_size = gr.Number(label='SoVITS batch_size', value=4, precision=0)
            sovits_epoch = gr.Number(label='SoVITS epoch', value=8, precision=0)
            gpt_batch_size = gr.Number(label='GPT batch_size', value=4, precision=0)
            gpt_epoch = gr.Number(label='GPT epoch', value=15, precision=0)
            save_every_epoch = gr.Number(label='保存間隔(epoch)', value=1, precision=0)
        with gr.Row():
            text_low_lr_rate = gr.Number(label='text_low_lr_rate', value=0.4)
            lora_rank = gr.Number(label='LoRA rank(v4 用)', value=32, precision=0)
            if_save_latest = gr.Checkbox(label='latest を保存', value=True)
            if_save_every_weights = gr.Checkbox(label='各 epoch の重みも保存', value=True, info='オフでも最終 epoch の重みは保存されます。')
            if_grad_ckpt = gr.Checkbox(label='gradient checkpoint', value=False)
            if_dpo = gr.Checkbox(label='DPO を有効化', value=False)
    with gr.Row():
        preprocess_btn = gr.Button('前処理を実行')
        sovits_btn = gr.Button('SoVITS 学習を開始')
        gpt_btn = gr.Button('GPT 学習を開始')
        full_btn = gr.Button('前処理から学習まで一括実行', variant='primary')
        stop_btn = gr.Button('実行中の処理を停止')
    train_log = gr.Textbox(label='ログ', lines=24, max_lines=32, interactive=False)

    gr.Markdown('## 推論\n学習後の確認は既存の 1C 感情推論 UI で行うのが扱いやすいです。最新の学習済み重みパスもここで確認できます。')
    with gr.Row():
        refresh_btn = gr.Button('最新の学習済み重みを再検索')
        open_infer_btn = gr.Button('1C 感情推論を開く')
    shared_outputs = [state, status, clip_table, session_json, list_path, wav_dir]

    load_btn.click(load_files, [uploads, save_name, speaker_name, language_code], shared_outputs)
    whisper_btn.click(prepare_whisper, outputs=[status])
    transcribe_btn.click(transcribe_all, [state], shared_outputs)
    export_btn.click(export_dataset, [state, save_name, speaker_name, language_code], shared_outputs)
    refresh_sessions_btn.click(refresh_session_list, [session_picker], [session_picker, status])
    load_session_btn.click(load_saved_session, [session_picker], [state, status, clip_table, session_json, list_path, wav_dir, save_name, speaker_name, language_code, session_picker])
    clone_session_btn.click(clone_current_session, [state, save_name, speaker_name, language_code], [state, status, clip_table, session_json, list_path, wav_dir, session_picker])
    delete_session_btn.click(delete_saved_session, [session_picker], [session_picker, status])
    version.change(version_weight_updates, [version, custom_pretrained], [pretrained_s1, pretrained_s2g, pretrained_s2d, model_guide])
    custom_pretrained.change(custom_pretrained_updates, [custom_pretrained, version], [pretrained_s1, pretrained_s2g, pretrained_s2d, model_guide])
    preprocess_btn.click(preprocess_action, [state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path], [state, status, session_json, train_log])
    sovits_btn.click(sovits_action, [state, version, gpu_number, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d], [state, status, session_json, train_log])
    gpt_btn.click(gpt_action, [state, version, gpu_number, gpt_batch_size, gpt_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1], [state, status, session_json, train_log])
    full_btn.click(full_pipeline_action, [state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2d, gpt_batch_size, gpt_epoch, if_dpo, pretrained_s1], [state, status, session_json, train_log])
    stop_btn.click(stop_current_process, outputs=[status])
    refresh_btn.click(refresh_latest_weights, [state, version], [state, session_json])
    open_infer_btn.click(open_inference_ui, outputs=[status])

app.queue().launch(server_name='0.0.0.0', server_port=DEFAULT_PORT, share=False, inbrowser=True)


























