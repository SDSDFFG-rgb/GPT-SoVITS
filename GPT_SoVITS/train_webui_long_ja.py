import ctypes
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
DEFAULT_VERSION = os.environ.get("version", "v4")
CURRENT_PROCESS = None
CURRENT_PROCESS_NAME = ""
WHISPER_MODEL = None
WHISPER_MODEL_NAME = None
SESSION_KIND = 'long'


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
        'session_dir': state.get('session_dir', ''),
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
            return 'sources' in data
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


def build_cloned_session(state, save_name, speaker_name, language_code):
    new_state = ensure_session(save_name, speaker_name, language_code, temporary=False)
    old_dir = Path(state['session_dir'])
    new_dir = Path(new_state['session_dir'])
    for folder in ['uploads', 'extracted', 'denoised', 'segments', 'edited', 'transcripts', 'dataset']:
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
    save_state(cloned)
    return cloned, str(old_dir), new_dir


def apply_editor_values_to_state(state, values, source_session_dir=None):
    expected = len(state['clips']) * 3
    if len(values) != expected:
        raise ValueError('編集欄の数が一致しません。画面を再読み込みしてもう一度試してください。')
    idx = 0
    transcript_dir = Path(state['session_dir']) / 'transcripts'
    transcript_dir.mkdir(parents=True, exist_ok=True)
    for clip in state['clips']:
        audio_path = values[idx]
        text_value = values[idx + 1]
        keep_value = values[idx + 2]
        resolved_audio_path = ''
        if audio_path:
            resolved_audio_path = str(audio_path)
            if source_session_dir and resolved_audio_path.startswith(source_session_dir):
                remapped = resolved_audio_path.replace(source_session_dir, state['session_dir'], 1)
                if Path(remapped).exists():
                    resolved_audio_path = remapped
        clip['trimmed_path'] = resolved_audio_path or clip.get('trimmed_path', '')
        clip['text'] = (text_value or '').strip()
        clip['keep'] = bool(keep_value)
        (transcript_dir / f"{clip['id']:03d}.txt").write_text((clip['text'] + '\n') if clip['text'] else '', encoding='utf-8')
        idx += 3
    save_state(state)
    kept = sum(1 for clip in state['clips'] if clip['keep'])
    filled = sum(1 for clip in state['clips'] if clip['text'])
    return kept, filled


def clone_current_session(state, save_name, speaker_name, language_code):
    if not state or not state.get('session_dir'):
        return None, '先に複製元のセッションを読み込んでください。', [], [], [], {}, '', '', maxine_status_text(detect_maxine_assets()), gr.update(choices=session_options(), value=None)
    cloned, old_dir, new_dir = build_cloned_session(state, save_name, speaker_name, language_code)
    choices = session_options()
    selected = next((choice for choice in choices if choice.endswith('| ' + new_dir.name)), None)
    message = f'保存済みの内容を新しいセッションとして保存しました: {new_dir.name}'
    ds = cloned.get('dataset', {})
    return cloned, message, source_rows(cloned), clip_rows(cloned), skipped_rows(cloned), state_summary(cloned), ds.get('list_path', ''), ds.get('wav_dir', ''), maxine_status_text(cloned.get('maxine_status')), gr.update(choices=choices, value=selected)


def clone_current_session_with_rows(state, save_name, speaker_name, language_code, *values):
    if not state or not state.get('session_dir'):
        return None, '先に複製元のセッションを読み込んでください。', [], [], [], {}, '', '', maxine_status_text(detect_maxine_assets()), gr.update(choices=session_options(), value=None)
    try:
        cloned, old_dir, new_dir = build_cloned_session(state, save_name, speaker_name, language_code)
        kept, filled = apply_editor_values_to_state(cloned, values, source_session_dir=old_dir)
        choices = session_options()
        selected = next((choice for choice in choices if choice.endswith('| ' + new_dir.name)), None)
        ds = cloned.get('dataset', {})
        message = f'編集内容を反映して新しいセッションとして保存しました: {new_dir.name} 採用 {kept} 件 / テキスト入力済み {filled} 件。'
        return cloned, message, source_rows(cloned), clip_rows(cloned), skipped_rows(cloned), state_summary(cloned), ds.get('list_path', ''), ds.get('wav_dir', ''), maxine_status_text(cloned.get('maxine_status')), gr.update(choices=choices, value=selected)
    except Exception as exc:
        ds = (state or {}).get('dataset', {})
        return state, f'別セッション保存に失敗しました: {exc}', source_rows(state), clip_rows(state), skipped_rows(state), state_summary(state), ds.get('list_path', ''), ds.get('wav_dir', ''), maxine_status_text((state or {}).get('maxine_status')), gr.update(choices=session_options(), value=None)

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


def get_whisper_model(model_name=None):
    global WHISPER_MODEL, WHISPER_MODEL_NAME
    model_name = model_name or os.environ.get('reference_whisper_model', 'turbo')
    if WHISPER_MODEL is None or WHISPER_MODEL_NAME != model_name:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
        WHISPER_MODEL_NAME = model_name
    return WHISPER_MODEL


def prepare_whisper(model_name=None):
    model_name = model_name or os.environ.get('reference_whisper_model', 'turbo')
    os.environ['reference_whisper_model'] = model_name
    get_whisper_model(model_name)
    return f'Whisper モデルを準備しました: {model_name}'


def prepare_anime_whisper():
    return prepare_whisper('quantumcookie/anime-whisper-ct2')
def prepare_whisper_selection(whisper_choice):
    if whisper_choice == 'anime-whisper':
        return prepare_anime_whisper()
    return prepare_whisper('turbo')


def run_command(args, *, cwd=None, env=None, process_name='command'):
    proc = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip()
        raise RuntimeError(f'[{process_name}] ' + tail[-2000:])
    return proc.stdout, proc.stderr


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
    if not state or not state.get('sources'):
        return editor_outputs(state, '先に素材を読み込んでください。')
    try:
        if not state.get('clips'):
            ok, message = ensure_source_audio_prepared(state)
            if not ok:
                return editor_outputs(state, message)
            rebuild_clips_from_sources(state)
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
        save_state(state)
        summary_text = '\n'.join(summaries[:12])
        if len(summaries) > 12:
            summary_text += f"\n- 他 {len(summaries) - 12} 件"
        message = f"Whisper 書き起こし完了: {success_count}/{len(state['clips'])} 件。空文字 {empty_count} 件。  \\n保存先: {transcript_dir}  \\n{summary_text}"
        return editor_outputs(state, message)
    except Exception as exc:
        traceback.print_exc()
        return editor_outputs(state, f'Whisper 書き起こしに失敗しました: {exc}')
def save_editor_rows(state, *values):
    if not state or not state.get('clips'):
        return state, '先に Whisper 書き起こしか無音分割を実行してください。', clip_rows(state), state_summary(state)
    old_session_dir = state.get('session_dir', '')
    state, materialized = materialize_session(state, state.get('save_name', 'long_train_ja'), state.get('speaker_name', ''), state.get('language_code', 'ja'))
    try:
        kept, filled = apply_editor_values_to_state(state, values, source_session_dir=(old_session_dir if materialized else None))
    except Exception as exc:
        return state, str(exc), clip_rows(state), state_summary(state)
    prefix = '新規保存しました。' if materialized else '一覧の編集を保存しました。'
    return state, f'{prefix} 採用 {kept} 件 / テキスト入力済み {filled} 件。保存先: {Path(state["session_dir"]) / "session_state.json"}', clip_rows(state), state_summary(state)

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
    tmp_config = TEMP_ROOT / 'tmp_s2_long.json'
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
    tmp_config = TEMP_ROOT / 'tmp_s1_long.yaml'
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


def latest_exp_weight(root, exp_name, suffix):
    root = Path(root)
    if not exp_name or not root.exists():
        return ''
    matches = sorted(root.glob(f'{exp_name}*{suffix}'), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else ''


def experiment_weight_map(version, exp_name):
    return {
        'sovits': latest_exp_weight(SoVITS_weight_version2root[version], exp_name, '.pth'),
        'gpt': latest_exp_weight(GPT_weight_version2root[version], exp_name, '.ckpt'),
    }


def cleanup_training_artifacts(state, version, keep_weights=None):
    exp_name = sanitize_name((state or {}).get('save_name', ''))
    if not exp_name:
        return []
    keep_weights = keep_weights or experiment_weight_map(version, exp_name)
    removed = []
    log_dir = Path(exp_root) / exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
        removed.append(str(log_dir))
    for key, root_dir, suffix in [('sovits', SoVITS_weight_version2root[version], '.pth'), ('gpt', GPT_weight_version2root[version], '.ckpt')]:
        keep_path = keep_weights.get(key, '')
        keep_resolved = str(Path(keep_path).resolve()) if keep_path else ''
        root = Path(root_dir)
        if not root.exists():
            continue
        for candidate in root.glob(f'{exp_name}*{suffix}'):
            candidate_resolved = str(candidate.resolve())
            if keep_resolved and candidate_resolved == keep_resolved:
                continue
            candidate.unlink(missing_ok=True)
            removed.append(str(candidate))
    return removed


def maybe_cleanup_after_training(state, version, cleanup_after_train):
    exp_name = sanitize_name((state or {}).get('save_name', ''))
    exp_weights = experiment_weight_map(version, exp_name)
    latest_weights = {
        'sovits': exp_weights['sovits'] or (state or {}).get('latest_weights', {}).get('sovits', ''),
        'gpt': exp_weights['gpt'] or (state or {}).get('latest_weights', {}).get('gpt', ''),
    }
    state['latest_weights'] = latest_weights
    if not cleanup_after_train:
        return state, ''
    if not latest_weights['sovits'] or not latest_weights['gpt']:
        return state, ' 自動整理はスキップしました。推論用の SoVITS/GPT 重みが両方そろった後にだけ削除します。'
    removed = cleanup_training_artifacts(state, version, latest_weights)
    return state, f' 自動整理を実行しました。削除対象: {len(removed)} 件。'


def refresh_latest_weights(state, version):
    state = state or ensure_session('long_train', '', 'ja')
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


def sovits_action(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d):
    try:
        log = run_sovits_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d)
        state, cleanup_note = maybe_cleanup_after_training(state, version, cleanup_after_train)
        return state, f'SoVITS 学習が完了しました。{cleanup_note}'.strip(), state_summary(state), log
    except Exception as exc:
        traceback.print_exc()
        return state, f'SoVITS 学習に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def gpt_action(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, if_dpo, pretrained_s1):
    try:
        log = run_gpt_training(state, version, gpu_number, batch_size, total_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1)
        state, cleanup_note = maybe_cleanup_after_training(state, version, cleanup_after_train)
        return state, f'GPT 学習が完了しました。{cleanup_note}'.strip(), state_summary(state), log
    except Exception as exc:
        traceback.print_exc()
        return state, f'GPT 学習に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


def full_pipeline_action(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2d, gpt_batch_size, gpt_epoch, if_dpo, pretrained_s1):
    try:
        parts = []
        prep_log, _ = run_preprocess(state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path)
        parts.append(prep_log)
        parts.append(run_sovits_training(state, version, gpu_number, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d))
        parts.append(run_gpt_training(state, version, gpu_number, gpt_batch_size, gpt_epoch, save_every_epoch, if_save_latest, if_save_every_weights, if_dpo, pretrained_s1))
        state, cleanup_note = maybe_cleanup_after_training(state, version, cleanup_after_train)
        return state, f'前処理から学習まで完了しました。{cleanup_note}'.strip(), state_summary(state), '\n\n'.join(parts)
    except Exception as exc:
        traceback.print_exc()
        return state, f'一括実行に失敗しました: {exc}', state_summary(state or {}), traceback.format_exc()


DEFAULT_BERT_DIR = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
DEFAULT_SSL_DIR = 'GPT_SoVITS/pretrained_models/chinese-hubert-base'
DEFAULT_SV_PATH = 'GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt'
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma'}
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}
SUPPORTED_UPLOAD_EXTS = sorted(AUDIO_EXTS | VIDEO_EXTS)
SPLIT_PROFILES = {
    '標準': {'noise': '-35dB', 'duration': 0.35},
    '保守的': {'noise': '-32dB', 'duration': 0.5},
    '積極的': {'noise': '-38dB', 'duration': 0.25},
}
DEFAULT_SILENCE_DURATION_SEC = SPLIT_PROFILES['標準']['duration']
DEFAULT_PORT = int(os.environ.get('long_train_webui', '9879'))
MAXINE_CACHE = None
MAXINE_RUNTIME_BIN = ROOT_DIR.parent / 'Maxine-AFX-Runtime' / 'nv' / 'bin'
MAXINE_MODELS_ROOT = MAXINE_RUNTIME_BIN / 'models'
MAXINE_EFFECTS_DEMO = ROOT_DIR.parent / 'Maxine-AFX-Runtime' / 'tools' / 'effects_demo.exe'


def source_kind(path):
    return 'video' if Path(path).suffix.lower() in VIDEO_EXTS else 'audio'


def save_state(state):
    if not state or not state.get('session_dir'):
        return
    (Path(state['session_dir']) / 'session_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_session(save_name, speaker_name, language_code, temporary=False):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    base_root = TEMP_ROOT if temporary else SESSION_ROOT
    base = base_root / f"{sid}_{SESSION_KIND}_{sanitize_name(save_name)}"
    for name in ['uploads', 'extracted', 'denoised', 'segments', 'edited', 'dataset', 'transcripts']:
        (base / name).mkdir(parents=True, exist_ok=True)
    return {
        'session_dir': str(base),
        'save_name': sanitize_name(save_name),
        'speaker_name': derive_speaker(save_name, speaker_name),
        'language_code': language_code or 'ja',
        'sources': [],
        'clips': [],
        'skipped_clips': [],
        'dataset': {},
        'latest_weights': {},
        'split_profile': '標準',
        'silence_duration_sec': DEFAULT_SILENCE_DURATION_SEC,
        'min_duration_sec': 3.0,
        'maxine_status': detect_maxine_assets(),
        'session_kind': SESSION_KIND,
        'is_temp_session': temporary,
    }


def source_rows(state):
    rows = []
    for source in (state or {}).get('sources', []):
        rows.append([source['id'], source.get('kind', ''), source['name'], f"{source.get('duration', 0.0):.2f}" if source.get('duration') else '', '済' if source.get('extracted_audio_path') else '未', '済' if source.get('maxine_applied') else '未'])
    return rows


def clip_rows(state):
    rows = []
    for clip in (state or {}).get('clips', []):
        rows.append([clip['id'], '採用' if clip['keep'] else '除外', clip['name'], clip.get('source_name', ''), f"{clip['duration']:.2f}", clip['text']])
    return rows


def skipped_rows(state):
    rows = []
    for clip in (state or {}).get('skipped_clips', []):
        rows.append([clip.get('source_name', ''), f"{clip.get('segment_start', 0.0):.2f} - {clip.get('segment_end', 0.0):.2f}", f"{clip.get('duration', 0.0):.2f}", clip.get('reason', '')])
    return rows


def state_summary(state):
    if not state:
        return {}
    ds = state.get('dataset', {})
    return {
        'session': Path(state['session_dir']).name,
        'save_name': state.get('save_name', ''),
        'speaker_name': state.get('speaker_name', ''),
        'language': state.get('language_code', ''),
        'source_count': len(state.get('sources', [])),
        'clip_count': len(state.get('clips', [])),
        'kept_count': sum(1 for c in state.get('clips', []) if c.get('keep')),
        'skipped_count': len(state.get('skipped_clips', [])),
        'split_profile': state.get('split_profile', '標準'),
        'silence_duration_sec': state.get('silence_duration_sec', DEFAULT_SILENCE_DURATION_SEC),
        'min_duration_sec': state.get('min_duration_sec', 3.0),
        'maxine': state.get('maxine_status', {}),
        'list_path': ds.get('list_path', ''),
        'wav_dir': ds.get('wav_dir', ''),
        'latest_weights': state.get('latest_weights', {}),
        'session_dir': state.get('session_dir', ''),
    }


def editor_outputs(state, message=''):
    ds = (state or {}).get('dataset', {})
    return (state, message, source_rows(state), clip_rows(state), skipped_rows(state), state_summary(state), ds.get('list_path', ''), ds.get('wav_dir', ''), maxine_status_text((state or {}).get('maxine_status')))


def normalize_loaded_state(state):
    if not state:
        return None, 0
    state.setdefault('save_name', 'long_train_ja')
    state.setdefault('speaker_name', state.get('save_name', 'long_train_ja'))
    state.setdefault('language_code', 'ja')
    state.setdefault('sources', [])
    state.setdefault('clips', [])
    state.setdefault('skipped_clips', [])
    state.setdefault('dataset', {})
    state.setdefault('latest_weights', {})
    state.setdefault('session_kind', SESSION_KIND)
    state.setdefault('split_profile', '標準')
    state.setdefault('silence_duration_sec', DEFAULT_SILENCE_DURATION_SEC)
    state.setdefault('min_duration_sec', 3.0)
    state.setdefault('maxine_status', detect_maxine_assets())
    session_dir = Path(state['session_dir'])
    dataset_dir = session_dir / 'dataset'
    ds = state['dataset']
    list_path = dataset_dir / 'annotations.list'
    wav_dir = dataset_dir / 'audio'
    session_json = dataset_dir / 'session.json'
    if list_path.exists(): ds['list_path'] = str(list_path)
    if wav_dir.exists(): ds['wav_dir'] = str(wav_dir)
    if session_json.exists(): ds['session_json'] = str(session_json)
    missing_trimmed = 0
    for clip in state['clips']:
        clip.setdefault('trimmed_path', '')
        clip.setdefault('text', '')
        clip.setdefault('keep', True)
        clip.setdefault('trim_start', 0.0)
        clip.setdefault('trim_end', clip.get('duration', 0.0))
        clip.setdefault('source_name', clip.get('name', ''))
        trimmed_path = clip.get('trimmed_path') or ''
        if trimmed_path and not Path(trimmed_path).exists():
            clip['trimmed_path'] = ''
            missing_trimmed += 1
    for source in state['sources']:
        source.setdefault('kind', source_kind(source.get('name', '')))
        source.setdefault('extracted_audio_path', '')
        source.setdefault('denoised_audio_path', '')
        source.setdefault('active_audio_path', source.get('denoised_audio_path') or source.get('extracted_audio_path') or source.get('upload_path', ''))
        source.setdefault('duration', 0.0)
        source.setdefault('maxine_applied', bool(source.get('denoised_audio_path')))
    return state, missing_trimmed


def maxine_dependency_dirs(dll_path):
    dll_dir = Path(dll_path).parent
    runtime_root = dll_dir.parent
    return [str(runtime_root), str(dll_dir), str(dll_dir / 'external' / 'cuda' / 'bin'), str(dll_dir / 'external' / 'nvtrt' / 'bin'), str(dll_dir / 'external' / 'openssl' / 'bin')]


def maxine_export_available(dll_path):
    if platform.system() != 'Windows':
        return False, 'Windows 以外では未対応です。'
    required_exports = ['NvAFX_GetEffectList', 'NvAFX_GetFloatList']
    old_path = os.environ.get('PATH', '')
    dep_dirs = [p for p in maxine_dependency_dirs(dll_path) if Path(p).exists()]
    handles = []
    if dep_dirs:
        os.environ['PATH'] = os.pathsep.join(dep_dirs + [old_path]) if old_path else os.pathsep.join(dep_dirs)
        add_dir = getattr(os, 'add_dll_directory', None)
        if add_dir is not None:
            for dep_dir in dep_dirs:
                try: handles.append(add_dir(dep_dir))
                except Exception: pass
    try:
        lib = ctypes.WinDLL(str(dll_path))
        missing = []
        for export_name in required_exports:
            try: getattr(lib, export_name)
            except Exception: missing.append(export_name)
        if missing:
            return False, '不足エクスポート: ' + ', '.join(missing)
        return True, ''
    except Exception as exc:
        return False, str(exc)
    finally:
        for handle in handles:
            try: handle.close()
            except Exception: pass
        os.environ['PATH'] = old_path


def detect_maxine_assets(force=False):
    global MAXINE_CACHE
    if MAXINE_CACHE is not None and not force:
        return dict(MAXINE_CACHE)
    status = {'available': False, 'reason': '', 'effects_demo': str(MAXINE_EFFECTS_DEMO), 'model_path': '', 'dll_dir': '', 'dll_path': '', 'version': ''}
    if not MAXINE_EFFECTS_DEMO.exists():
        status['reason'] = f'effects_demo.exe が見つかりません: {MAXINE_EFFECTS_DEMO}'
        MAXINE_CACHE = status
        return dict(status)
    if not MAXINE_RUNTIME_BIN.exists():
        status['reason'] = f'AFX ランタイムが見つかりません: {MAXINE_RUNTIME_BIN}'
        MAXINE_CACHE = status
        return dict(status)
    if not MAXINE_MODELS_ROOT.exists():
        status['reason'] = f'AFX モデルディレクトリが見つかりません: {MAXINE_MODELS_ROOT}'
        MAXINE_CACHE = status
        return dict(status)
    dll_path = MAXINE_RUNTIME_BIN / 'NVAudioEffects.dll'
    model_path = MAXINE_MODELS_ROOT / 'denoiser_48k.trtpkg'
    if not dll_path.exists():
        status['reason'] = f'AFX DLL が見つかりません: {dll_path}'
        MAXINE_CACHE = status
        return dict(status)
    if not model_path.exists():
        status['reason'] = f'AFX モデルが見つかりません: {model_path}'
        MAXINE_CACHE = status
        return dict(status)
    ok, reason = maxine_export_available(dll_path)
    if not ok:
        status['reason'] = f'AFX DLL を読み込めません: {reason}'
        MAXINE_CACHE = status
        return dict(status)
    status.update({'available': True, 'reason': '', 'model_path': str(model_path), 'dll_dir': str(MAXINE_RUNTIME_BIN), 'dll_path': str(dll_path), 'version': '1.6.1.2-GA-Ada'})
    MAXINE_CACHE = status
    return dict(status)


def maxine_status_text(status):
    if not status:
        return 'Maxine 状態: 未確認'
    if status.get('available'):
        return 'Maxine 状態: 利用可能  \\n- モデル: `' + status.get('model_path', '') + '`  \\n- DLL: `' + status.get('dll_dir', '') + '`  \\n- 実行ファイル: `' + status.get('effects_demo', '') + '`'
    return f'Maxine 状態: 利用不可  \\n- 理由: {status.get("reason", "不明")}'


def load_saved_session(session_name):
    session_name = session_name_from_choice(session_name)
    choices = session_options()
    if not session_name:
        return (None, '読み込むセッションを選んでください。', [], [], [], {}, '', '', maxine_status_text(detect_maxine_assets()), 'long_train_ja', 'long_train_ja', 'ja', '標準', DEFAULT_SILENCE_DURATION_SEC, 3.0, gr.update(choices=choices, value=None))
    session_dir = SESSION_ROOT / session_name
    state_path = session_dir / 'session_state.json'
    fallback_path = session_dir / 'dataset' / 'session.json'
    target = state_path if state_path.exists() else fallback_path
    if not target.exists():
        return (None, f'セッション情報が見つかりません: {session_name}', [], [], [], {}, '', '', maxine_status_text(detect_maxine_assets()), 'long_train_ja', 'long_train_ja', 'ja', '標準', DEFAULT_SILENCE_DURATION_SEC, 3.0, gr.update(choices=choices, value=session_name if session_name in choices else None))
    state = json.loads(target.read_text(encoding='utf-8'))
    state, missing_trimmed = normalize_loaded_state(state)
    message = f'セッションを読み込みました: {session_name}'
    if missing_trimmed:
        message += f'。一時トリム音声 {missing_trimmed} 件は元ファイル参照に戻しています。'
    ds = state.get('dataset', {})
    return (state, message, source_rows(state), clip_rows(state), skipped_rows(state), state_summary(state), ds.get('list_path', ''), ds.get('wav_dir', ''), maxine_status_text(state.get('maxine_status')), state.get('save_name', 'long_train_ja'), state.get('speaker_name', state.get('save_name', 'long_train_ja')), state.get('language_code', 'ja'), state.get('split_profile', '標準'), state.get('silence_duration_sec', DEFAULT_SILENCE_DURATION_SEC), state.get('min_duration_sec', 3.0), gr.update(choices=choices, value=session_name))


def prepare_anime_whisper():
    return prepare_whisper('quantumcookie/anime-whisper-ct2')

def load_sources(files, save_name, speaker_name, language_code):
    if not files:
        return editor_outputs(None, '音声または動画ファイルをドロップしてください。')
    state = ensure_session(save_name, speaker_name, language_code, temporary=True)
    upload_dir = Path(state['session_dir']) / 'uploads'
    sources = []
    for i, file_obj in enumerate(files):
        src = Path(file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
        ext = src.suffix or '.wav'
        dst = upload_dir / f'{i:03d}{ext}'
        shutil.copy2(src, dst)
        sources.append({'id': i, 'name': src.name, 'kind': source_kind(src), 'original_path': str(src), 'upload_path': str(dst), 'extracted_audio_path': '', 'denoised_audio_path': '', 'active_audio_path': '', 'duration': 0.0, 'maxine_applied': False})
    state['sources'] = sources
    state['clips'] = []
    state['skipped_clips'] = []
    save_state(state)
    return editor_outputs(state, f'{len(sources)} 個の素材を読み込みました。Whisper に直接渡すか、必要なら無音分割や Maxine を実行してください。')


def extract_source_audio(input_path, output_path):
    run_command(['ffmpeg', '-y', '-i', str(input_path), '-vn', '-ac', '1', '-ar', '48000', '-sample_fmt', 's16', str(output_path)], cwd=str(ROOT_DIR), process_name='音声抽出')


def ensure_source_audio_prepared(state):
    if not state or not state.get('sources'):
        return False, '先に素材を読み込んでください。'
    extracted_dir = Path(state['session_dir']) / 'extracted'
    extracted_dir.mkdir(parents=True, exist_ok=True)
    changed = False
    for source in state['sources']:
        active_path = source.get('active_audio_path')
        if active_path and Path(active_path).exists():
            if not source.get('duration'):
                source['duration'] = duration_of(str(active_path))
                changed = True
            continue
        output_path = extracted_dir / f"{source['id']:03d}.wav"
        extracted_path = source.get('extracted_audio_path')
        if not extracted_path or not Path(extracted_path).exists():
            extract_source_audio(source['upload_path'], output_path)
            source['extracted_audio_path'] = str(output_path)
            source['denoised_audio_path'] = ''
            source['maxine_applied'] = False
            extracted_path = str(output_path)
            changed = True
        active_path = source.get('denoised_audio_path') or extracted_path
        source['active_audio_path'] = str(active_path)
        source['duration'] = duration_of(str(active_path))
        changed = True
    state['maxine_status'] = detect_maxine_assets(force=True)
    if changed:
        save_state(state)
    return True, ''


def rebuild_clips_from_sources(state):
    state['clips'] = []
    state['skipped_clips'] = []
    clip_id = 0
    for source in state.get('sources', []):
        active_path = source.get('active_audio_path') or source.get('extracted_audio_path') or source.get('upload_path')
        if not active_path or not Path(active_path).exists():
            continue
        duration = round(source.get('duration') or duration_of(str(active_path)), 4)
        state['clips'].append({'id': clip_id, 'name': f"{Path(source['name']).stem}.wav", 'source_name': source['name'], 'source_path': str(active_path), 'trimmed_path': '', 'duration': duration, 'trim_start': 0.0, 'trim_end': duration, 'text': '', 'keep': True, 'segment_start': 0.0, 'segment_end': duration})
        clip_id += 1


def extract_sources(state):
    if not state or not state.get('sources'):
        return editor_outputs(state, '先に素材を読み込んでください。')
    ok, message = ensure_source_audio_prepared(state)
    if not ok:
        return editor_outputs(state, message)
    return editor_outputs(state, '素材から音声を抽出しました。Whisper に直接渡すか、必要なら Maxine や無音分割を実行してください。')


def run_maxine_denoiser(input_wav, output_wav, maxine_status):
    cfg_text = 'effect denoiser\n' + f'input_wav {input_wav}\n' + f'output_wav {output_wav}\n' + 'real_time 0\n' + 'intensity_ratio 1.0\n' + 'enable_vad 0\n' + f'model {maxine_status["model_path"]}\n'
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as fp:
        fp.write(cfg_text)
        cfg_path = fp.name
    try:
        env = os.environ.copy()
        path_parts = [p for p in maxine_dependency_dirs(maxine_status['dll_path']) if Path(p).exists()]
        env['PATH'] = os.pathsep.join(path_parts + [env.get('PATH', '')]) if env.get('PATH', '') else os.pathsep.join(path_parts)
        run_command([str(MAXINE_EFFECTS_DEMO), '-c', cfg_path], cwd=str(MAXINE_EFFECTS_DEMO.parent), env=env, process_name='Maxine denoise')
    finally:
        try: os.unlink(cfg_path)
        except OSError: pass


def apply_maxine(state):
    if not state or not state.get('sources'):
        return editor_outputs(state, '先に素材を読み込んでください。')
    ok, message = ensure_source_audio_prepared(state)
    if not ok:
        return editor_outputs(state, message)
    status = detect_maxine_assets(force=True)
    state['maxine_status'] = status
    if not status.get('available'):
        save_state(state)
        return editor_outputs(state, f'Maxine は使えません。元音声のまま継続します。理由: {status.get("reason", "不明")}')
    denoised_dir = Path(state['session_dir']) / 'denoised'
    warnings = []
    for source in state['sources']:
        input_wav = source['extracted_audio_path']
        output_wav = denoised_dir / f"{source['id']:03d}.wav"
        try:
            run_maxine_denoiser(input_wav, str(output_wav), status)
            source['denoised_audio_path'] = str(output_wav)
            source['active_audio_path'] = str(output_wav)
            source['maxine_applied'] = True
            source['duration'] = duration_of(str(output_wav))
        except Exception as exc:
            source['denoised_audio_path'] = ''
            source['active_audio_path'] = input_wav
            source['maxine_applied'] = False
            warnings.append(f"{source['name']}: {exc}")
    save_state(state)
    if warnings:
        joined = '\n'.join(warnings[:5])
        if len(warnings) > 5:
            joined += f'\n他 {len(warnings) - 5} 件'
        return editor_outputs(state, f'Maxine は一部失敗したため、失敗分は元音声のまま継続します。\\n{joined}')
    return editor_outputs(state, 'Maxine によるノイズ除去が完了しました。')


def detect_speech_segments(audio_path, split_profile, silence_duration_sec=None):
    profile = SPLIT_PROFILES.get(split_profile, SPLIT_PROFILES['標準'])
    duration = float(silence_duration_sec if silence_duration_sec is not None else profile['duration'])
    if duration <= 0:
        raise ValueError('無音検出時間は 0 より大きい値にしてください。')
    total = duration_of(audio_path)
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(audio_path), '-af', f"silencedetect=noise={profile['noise']}:d={duration}", '-f', 'null', '-'], cwd=str(ROOT_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if proc.returncode not in (0, 255):
        raise RuntimeError((proc.stderr or '').strip()[-2000:])
    silence_start_re = re.compile(r'silence_start:\s*([0-9.]+)')
    silence_end_re = re.compile(r'silence_end:\s*([0-9.]+)')
    silences = []
    current_start = None
    for line in proc.stderr.splitlines():
        m = silence_start_re.search(line)
        if m:
            current_start = float(m.group(1))
            continue
        m = silence_end_re.search(line)
        if m and current_start is not None:
            silences.append((current_start, float(m.group(1))))
            current_start = None
    if not silences:
        return [(0.0, total)] if total > 0 else []
    segments = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start > cursor:
            segments.append((round(cursor, 4), round(silence_start, 4)))
        cursor = max(cursor, silence_end)
    if total > cursor:
        segments.append((round(cursor, 4), round(total, 4)))
    return [(s, e) for s, e in segments if e - s > 0.05]


def cut_audio_segment(input_path, output_path, start_sec, end_sec):
    run_command(['ffmpeg', '-y', '-ss', f'{start_sec:.4f}', '-to', f'{end_sec:.4f}', '-i', str(input_path), '-vn', '-ac', '1', '-ar', '48000', '-sample_fmt', 's16', str(output_path)], cwd=str(ROOT_DIR), process_name='無音分割')


def split_sources(state, split_profile, silence_duration_sec, min_duration_sec):
    if not state or not state.get('sources'):
        return editor_outputs(state, '先に素材を読み込んでください。')
    ok, message = ensure_source_audio_prepared(state)
    if not ok:
        return editor_outputs(state, message)
    segments_dir = Path(state['session_dir']) / 'segments'
    if segments_dir.exists():
        shutil.rmtree(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)
    state['clips'] = []
    state['skipped_clips'] = []
    state['split_profile'] = split_profile
    state['silence_duration_sec'] = float(silence_duration_sec)
    state['min_duration_sec'] = float(min_duration_sec)
    clip_id = 0
    for source in state['sources']:
        active_path = source.get('active_audio_path') or source.get('extracted_audio_path')
        if not active_path:
            continue
        for seg_idx, (start, end) in enumerate(detect_speech_segments(active_path, split_profile, silence_duration_sec)):
            duration = round(end - start, 4)
            if duration < float(min_duration_sec):
                state['skipped_clips'].append({'source_name': source['name'], 'segment_index': seg_idx, 'segment_start': start, 'segment_end': end, 'duration': duration, 'reason': f'{float(min_duration_sec):.1f} 秒以下'})
                continue
            out_path = segments_dir / f"{clip_id:03d}.wav"
            cut_audio_segment(active_path, out_path, start, end)
            actual_duration = duration_of(str(out_path))
            state['clips'].append({'id': clip_id, 'name': f"{Path(source['name']).stem}_{seg_idx:03d}.wav", 'source_name': source['name'], 'source_path': str(out_path), 'trimmed_path': '', 'duration': actual_duration, 'trim_start': 0.0, 'trim_end': actual_duration, 'text': '', 'keep': True, 'segment_start': start, 'segment_end': end})
            clip_id += 1
    save_state(state)
    return editor_outputs(state, f'無音分割が完了しました。編集対象 {len(state["clips"])} 件 / 除外 {len(state["skipped_clips"])} 件。')


def export_dataset(state, save_name, speaker_name, language_code):
    if not state or not state.get('clips'):
        return editor_outputs(state, '先に素材を読み込んでください。')
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
def build_app():
    with gr.Blocks(title='GPT-SoVITS 長尺学習 UI', css=css, js=js) as app:
        gr.HTML(top_html)
        state = gr.State(None)
        gr.Markdown(
            '# 長尺音声・動画 学習 UI\n'
            '長い音声や動画から音声抽出、任意の Maxine ノイズ除去、無音分割、Whisper 書き起こし、手動修正、dataset 生成、前処理、学習までをまとめて行います。  \n'
            '学習済み重みの確認や推論は、最後に既存の **1C 感情推論 UI** を開いて行います。'
        )
        with gr.Row():
            save_name = gr.Textbox(label='保存名', value='long_train_ja', info='実験名と既定の話者名に使います。半角英数とアンダースコア推奨。')
            speaker_name = gr.Textbox(label='話者名', value='long_train_ja', info='空欄なら保存名を使います。')
            language_code = gr.Dropdown(label='言語', choices=['ja', 'all_ja', 'auto'], value='ja', info='通常は ja のままで構いません。')
            version = gr.Dropdown(label='学習モデル版', choices=SUPPORTED_VERSIONS, value=DEFAULT_VERSION, info='通常は v2Pro か v4 を選びます。')
        model_guide = gr.Markdown(version_guide(DEFAULT_VERSION))
        status = gr.Markdown('ここに進行状況が表示されます。')
        maxine_info = gr.Markdown(maxine_status_text(detect_maxine_assets()))

        with gr.Tabs():
            with gr.Tab('素材準備'):
                with gr.Row():
                    uploads = gr.File(label='長い音声 / 動画をまとめてドロップ', file_count='multiple', file_types=SUPPORTED_UPLOAD_EXTS)
                    with gr.Column():
                        gr.Markdown('1. 素材を読み込む  2. Whisper に直接渡すか、必要なら Maxine / 無音分割を挟む  3. 書き起こしを確認して学習データを書き出します。')
                        split_profile = gr.Dropdown(label='無音分割プリセット', choices=list(SPLIT_PROFILES.keys()), value='標準')
                        silence_duration_sec = gr.Number(label='無音検出時間(秒)', value=DEFAULT_SILENCE_DURATION_SEC, minimum=0.05, info='この秒数以上続いた無音だけを区切りとして使います。小さくすると細かく切れます。')
                        min_duration_sec = gr.Number(label='最低クリップ秒数', value=3.0, info='この秒数以下のクリップは一覧に出さず、スキップ一覧に記録します。')
                        with gr.Row():
                            load_btn = gr.Button('素材を読み込む', variant='primary')
                            extract_btn = gr.Button('音声を抽出')
                            maxine_btn = gr.Button('Maxineでノイズ除去')
                        whisper_choice = gr.Radio(label='Whisper モデル', choices=['turbo', 'anime-whisper'], value='turbo')
                        with gr.Row():
                            split_btn = gr.Button('無音分割', variant='primary')
                            whisper_btn = gr.Button('Whisper モデルを準備')
                            transcribe_btn = gr.Button('Whisperで書き起こし')
                        gr.Markdown('Whisperで書き起こし は、無音分割済みならそのクリップ群を、未分割なら素材ごとに 1 クリップとしてそのまま書き起こします。')
                        gr.Markdown('ラジオで選んだモデルが使われます。Whisperで書き起こし を押した時も、選択中モデルを自動で準備してから実行します。')
                        gr.Markdown('既に作業済みのセッションを再開したい場合は、下の一覧から読み込めます。')
                        with gr.Row():
                            session_picker = gr.Dropdown(label='保存済みセッション', choices=session_options(), value=None, allow_custom_value=False)
                            refresh_sessions_btn = gr.Button('一覧を更新')
                            load_session_btn = gr.Button('選択したセッションを読み込む')
                            clone_session_btn = gr.Button('保存済み内容で別セッション保存')
                            delete_session_btn = gr.Button('選択したセッションを削除')
                source_table = gr.Dataframe(headers=['ID', '種別', 'ファイル', '秒数', '抽出', 'Maxine'], datatype=['number', 'str', 'str', 'str', 'str', 'str'], interactive=False, wrap=True, label='素材一覧')
                skip_table = gr.Dataframe(headers=['素材', '区間', '秒数', '理由'], datatype=['str', 'str', 'str', 'str'], interactive=False, wrap=True, label='除外クリップ一覧')
                clip_table = gr.Dataframe(headers=['ID', '状態', 'ファイル', '元素材', '秒数', 'テキスト'], datatype=['number', 'str', 'str', 'str', 'str', 'str'], interactive=False, wrap=True, label='編集対象クリップ一覧')
                gr.Markdown('## クリップ編集\n左の波形でそのままトリムし、右で文字を直してください。')
                session_json = gr.JSON(label='セッション概要')

                @gr.render(inputs=state)
                def render_clip_editors(current_state):
                    if not current_state or not current_state.get('clips'):
                        gr.Markdown('まだ編集対象クリップがありません。Whisper 直接書き起こし、または無音分割を実行してください。')
                        return
                    row_inputs = []
                    for clip in current_state['clips']:
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=6, min_width=420):
                                audio_value = clip['trimmed_path'] or clip['source_path']
                                audio_editor = gr.Audio(value=audio_value, type='filepath', label=f"{clip['name']} / {clip.get('source_name', '')} ({clip['duration']:.2f}s)", waveform_options=gr.WaveformOptions(show_recording_waveform=False), interactive=True)
                            with gr.Column(scale=5, min_width=300):
                                text_editor = gr.Textbox(value=clip['text'], lines=4, label='書き起こし / 修正文', info='誤字、句読点、読み違いをここで直します。')
                                keep_editor = gr.Checkbox(value=clip['keep'], label='このクリップを学習に使う')
                            row_inputs.extend([audio_editor, text_editor, keep_editor])
                        gr.Markdown('---')
                    with gr.Row():
                        save_all_btn = gr.Button('現在のセッションに上書き保存', variant='primary')
                        clone_from_editor_btn = gr.Button('編集内容を反映して別セッションとして保存')
                    save_all_btn.click(save_editor_rows, [state] + row_inputs, [state, status, clip_table, session_json])
                    clone_from_editor_btn.click(clone_current_session_with_rows, [state, save_name, speaker_name, language_code] + row_inputs, [state, status, source_table, clip_table, skip_table, session_json, list_path, wav_dir, maxine_info, session_picker])
                    gr.Markdown('読み込み済みの既存セッションなら、そのまま同じ `session_state.json` と `transcripts/*.txt` に上書き保存されます。別名保存したい時は右のボタンを使うと、未保存の編集内容も一緒に複製されます。')

                gr.Markdown('## 学習データ書き出し\nここで `annotations.list` と学習用 WAV 一式を作ります。')
                with gr.Row():
                    export_btn = gr.Button('学習フォーマットを生成', variant='primary')
                    list_path = gr.Textbox(label='生成された annotations.list', interactive=False)
                    wav_dir = gr.Textbox(label='学習用 WAV フォルダ', interactive=False)

            with gr.Tab('学習'):
                gr.Markdown('既存の 1分学習 UI と同じ流れで前処理と学習を行います。迷ったら一括実行を使ってください。')
                with gr.Accordion('学習設定', open=False):
                    with gr.Row():
                        gpu_number = gr.Dropdown(label='GPU', choices=[str(i) for i in sorted(GPU_INDEX)], value=gpu_default(), info='学習に使う GPU を 1 枚選びます。')
                        custom_pretrained = gr.Checkbox(label='事前学習重みを手動指定する', value=False, info='通常はオフのままで構いません。')
                    with gr.Row():
                        pretrained_s1 = gr.Textbox(label='事前学習 GPT', value=default_weight(pretrained_gpt_name, DEFAULT_VERSION), interactive=False)
                        pretrained_s2g = gr.Textbox(label='事前学習 SoVITS-G', value=default_weight(pretrained_sovits_name, DEFAULT_VERSION), interactive=False)
                        pretrained_s2d = gr.Textbox(label='事前学習 SoVITS-D', value=default_sovits_d_weight(DEFAULT_VERSION), interactive=False, visible=DEFAULT_VERSION in {'v2Pro', 'v2ProPlus'})
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
                        cleanup_after_train = gr.Checkbox(label='学習完了後に不要物を削除', value=True, info='推論用の最新重みだけ残し、logs 配下と同実験の旧重みを削除します。セッションの音声・キャプションは残します。')
                        if_grad_ckpt = gr.Checkbox(label='gradient checkpoint', value=False)
                        if_dpo = gr.Checkbox(label='DPO を有効化', value=False)
                with gr.Row():
                    preprocess_btn = gr.Button('前処理を実行')
                    sovits_btn = gr.Button('SoVITS 学習を開始')
                    gpt_btn = gr.Button('GPT 学習を開始')
                    full_btn = gr.Button('前処理から学習まで一括実行', variant='primary')
                    stop_btn = gr.Button('実行中の処理を停止')
                train_log = gr.Textbox(label='ログ', lines=24, max_lines=32, interactive=False)
                gr.Markdown('## 推論\n学習後の確認は既存の 1C 感情推論 UI で行うのが扱いやすいです。')
                with gr.Row():
                    refresh_btn = gr.Button('最新の学習済み重みを再検索')
                    open_infer_btn = gr.Button('1C 感情推論を開く')

        shared_outputs = [state, status, source_table, clip_table, skip_table, session_json, list_path, wav_dir, maxine_info]
        load_btn.click(load_sources, [uploads, save_name, speaker_name, language_code], shared_outputs)
        extract_btn.click(extract_sources, [state], shared_outputs)
        maxine_btn.click(apply_maxine, [state], shared_outputs)
        split_btn.click(split_sources, [state, split_profile, silence_duration_sec, min_duration_sec], shared_outputs)
        whisper_btn.click(prepare_whisper_selection, [whisper_choice], [status])
        transcribe_btn.click(prepare_whisper_selection, [whisper_choice], [status]).then(transcribe_all, [state], shared_outputs)
        export_btn.click(export_dataset, [state, save_name, speaker_name, language_code], shared_outputs)
        refresh_sessions_btn.click(refresh_session_list, [session_picker], [session_picker, status])
        load_session_btn.click(load_saved_session, [session_picker], [state, status, source_table, clip_table, skip_table, session_json, list_path, wav_dir, maxine_info, save_name, speaker_name, language_code, split_profile, silence_duration_sec, min_duration_sec, session_picker])
        clone_session_btn.click(clone_current_session, [state, save_name, speaker_name, language_code], [state, status, source_table, clip_table, skip_table, session_json, list_path, wav_dir, maxine_info, session_picker])
        delete_session_btn.click(delete_saved_session, [session_picker], [session_picker, status])
        version.change(version_weight_updates, [version, custom_pretrained], [pretrained_s1, pretrained_s2g, pretrained_s2d, model_guide])
        custom_pretrained.change(custom_pretrained_updates, [custom_pretrained, version], [pretrained_s1, pretrained_s2g, pretrained_s2d, model_guide])
        preprocess_btn.click(preprocess_action, [state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path], [state, status, session_json, train_log])
        sovits_btn.click(sovits_action, [state, version, gpu_number, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2g, pretrained_s2d], [state, status, session_json, train_log])
        gpt_btn.click(gpt_action, [state, version, gpu_number, gpt_batch_size, gpt_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, if_dpo, pretrained_s1], [state, status, session_json, train_log])
        full_btn.click(full_pipeline_action, [state, version, gpu_number, bert_dir, ssl_dir, pretrained_s2g, sv_path, sovits_batch_size, sovits_epoch, save_every_epoch, if_save_latest, if_save_every_weights, cleanup_after_train, text_low_lr_rate, if_grad_ckpt, lora_rank, pretrained_s2d, gpt_batch_size, gpt_epoch, if_dpo, pretrained_s1], [state, status, session_json, train_log])
        stop_btn.click(stop_current_process, outputs=[status])
        refresh_btn.click(refresh_latest_weights, [state, version], [state, session_json])
        open_infer_btn.click(open_inference_ui, outputs=[status])
    return app


if __name__ == '__main__':
    build_app().queue().launch(server_name='0.0.0.0', server_port=DEFAULT_PORT, share=False, inbrowser=True)



































