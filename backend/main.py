"""
FX281 Podcast Processing Backend v12.0
- Local FunASR (SenseVoice) STT + Qwen analysis + Word/MP3 export
- Three-level deletion suggestion (keep/mild/strong)
- Task persistence to JSON file
- History API
"""

import os
import sys
import io
import json
import re
import uuid
import tempfile
import traceback
import logging
import asyncio
import httpx
from typing import List, Dict, Optional
from datetime import datetime
from http import HTTPStatus

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("fx281")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

import dashscope
dashscope.api_key = DASHSCOPE_API_KEY

QWEN_MODEL = "qwen-plus"
ASR_MODE = os.getenv("ASR_MODE", "local")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

from openai import OpenAI
qwen_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

_local_asr_model = None

def _get_local_asr():
    global _local_asr_model
    if _local_asr_model is not None:
        return _local_asr_model
    try:
        from funasr import AutoModel
        logger.info("Loading local SenseVoice model...")
        _local_asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 60000},
            device="cuda" if _check_cuda() else "cpu",
        )
        logger.info("Local SenseVoice model loaded successfully!")
        return _local_asr_model
    except Exception as e:
        logger.warning(f"Failed to load local ASR model: {e}")
        _local_asr_model = False
        return None

def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

tasks: Dict[str, dict] = {}

app = FastAPI(title="FX281 API", version="12.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

QWEN_BATCH_PROMPT = """你是一个专业的播客剪辑助手。以下是播客的部分转录文本（带时间戳）：

{transcript_text}

请仔细阅读以上文本，完成以下分析：

## 任务1：说话人识别
语音转文字工具无法区分说话人，所以所有段落都被标记为同一个说话人。你需要根据对话内容推断每段话的实际说话人：
- 仔细分析语气、用词、话题角色，判断哪些段落是同一个人说的
- 播客通常有2-4个说话人（主持人+1-3个嘉宾）
- 主持人(host)特征：提问、引导话题、总结、过渡
- 被访人(guest)特征：回答问题、分享经历、讲述观点
- 为每个说话人分配标识：Speaker_A, Speaker_B, Speaker_C 等
- 判断角色：host 或 guest

## 任务2：删减建议（三级体系）
请对每段话给出删减建议，分为三个等级：

**keep（保留）**：句子有实质语义价值，应当保留。
- 包含有价值的观点、经历、信息
- 回忆性叙述（即使有重复和口癖）
- 带有停顿但语义完整的句子
- 虽然啰嗦但传达了实质内容的句子
- 包含情感表达的句子

**mild（一般删减建议）**：句子有一定冗余，删不删都可以，由人决定。
- 带有大量口癖但仍有少量信息（去掉口癖后信息很少）
- 与前文高度重复的表述（不是回忆性重复，而是无意义的重复）
- 过长的过渡或铺垫，核心信息已在其他段落表达
- 语气词+少量附和内容

**strong（强烈删减建议）**：句子应当删除，没有保留价值。
- 纯语气词：整句只有"嗯"、"啊"、"哦"
- 纯附和：整句只有"嗯嗯"、"对对"、"好好好"，无附加观点
- 纯杂音：笑声、咳嗽等非语言内容

**判断原则**：
- 播客采访的对象可能是老人、教授，他们说话带有停顿、重复是正常的表达习惯
- 不要因为句子中包含口癖、停顿就标记删除，要判断去掉口癖后是否还有实质内容
- 有疑问时优先标 mild 而非 strong，让人类做最终决策

删减原因分类（对 mild 和 strong 使用）：
- filler：语气词/口癖
- echo：附和/重复
- noise：杂音/笑声
- redundant：冗余/过度铺垫

对每个非 keep 的句子，给出简短解读（15字以内）。

## 返回格式
返回 JSON 对象：
{{
  "segments": [
    {{
      "id": 1,
      "speaker": "Speaker_A",
      "speakerRole": "host",
      "text": "原文",
      "startTime": 0.0,
      "endTime": 1.0,
      "suggestion": "keep",
      "reason": null,
      "reasonDetail": null
    }},
    {{
      "id": 2,
      "speaker": "Speaker_B",
      "speakerRole": "guest",
      "text": "嗯嗯",
      "startTime": 1.0,
      "endTime": 1.5,
      "suggestion": "strong",
      "reason": "echo",
      "reasonDetail": "纯附和无观点"
    }},
    {{
      "id": 3,
      "speaker": "Speaker_B",
      "speakerRole": "guest",
      "text": "然后那个时候就是说我记得很清楚，就是七六年的时候",
      "startTime": 1.5,
      "endTime": 5.0,
      "suggestion": "keep",
      "reason": null,
      "reasonDetail": null
    }},
    {{
      "id": 4,
      "speaker": "Speaker_A",
      "speakerRole": "host",
      "text": "对对对，然后呢然后呢",
      "startTime": 5.0,
      "endTime": 6.0,
      "suggestion": "mild",
      "reason": "echo",
      "reasonDetail": "附和带少量催促"
    }}
  ]
}}

注意第3个示例：虽然句子带有"然后"、"就是说"等口癖，但整句包含有价值的回忆信息，应当 keep。
注意第4个示例：虽然"然后呢"有催促含义，但核心是附和，标 mild 让人决策。
必须为每一段文本都返回一个 segments 对象！所有文字使用简体中文。只输出 JSON。"""

QWEN_CHAPTER_PROMPT = """你是一个专业的播客编辑。以下是播客的完整转录文本（带时间戳和说话人），请通读全文后进行章节划分。

{transcript_text}

## 章节划分要求
1. 仔细通读全文，根据话题的自然变化划分章节
2. 章节数量：3-7个（根据内容丰富程度调整）
3. 每个章节标题要简洁精准地概括该段核心话题（5-10字）
4. 章节之间应当有明确的话题转换（如：从自我介绍转到回忆经历，从一段经历转到另一段，从讨论转到总结等）
5. startTime 取该章节第一句话的时间，endTime 取该章节最后一句话的时间
6. 章节必须覆盖全部时间范围，不能有遗漏

## 章节划分原则
- 不要按时间均匀切分，要按话题自然转折点切分
- 一个话题可能很长（占一半音频），也可能很短（只有几段），这很正常
- 标题要具体，不要用"第一部分"这种泛泛的标题
- 如果整段音频都在聊一个话题，至少也要分3个章节（开头、主体、收尾）

## 返回格式
返回 JSON 对象：
{{
  "chapters": [
    {{"title": "开场与自我介绍", "startTime": 0.0, "endTime": 120.0}},
    {{"title": "早年求学经历", "startTime": 120.0, "endTime": 350.0}},
    {{"title": "创作转型与突破", "startTime": 350.0, "endTime": 600.0}},
    {{"title": "对行业现状的看法", "startTime": 600.0, "endTime": 800.0}},
    {{"title": "未来展望与寄语", "startTime": 800.0, "endTime": 950.0}}
  ]
}}

只输出 JSON。所有文字使用简体中文。"""


def format_time(seconds: float) -> str:
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def merge_segments(raw: List[dict], min_chars: int = 15, max_gap: float = 2.0, max_chars: int = 200) -> List[dict]:
    if not raw:
        return []
    result = []
    current = {"start": raw[0]["start"], "end": raw[0]["end"], "text": raw[0]["text"]}
    for seg in raw[1:]:
        gap = seg["start"] - current["end"]
        combined_len = len(current["text"]) + len(seg["text"])
        if gap <= max_gap and combined_len <= max_chars and len(current["text"]) < min_chars:
            current["text"] += seg["text"]
            current["end"] = seg["end"]
        else:
            result.append(current)
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    result.append(current)
    final = []
    for seg in result:
        if final and len(seg["text"]) < min_chars:
            prev = final[-1]
            gap = seg["start"] - prev["end"]
            if gap <= max_gap and len(prev["text"]) + len(seg["text"]) <= max_chars:
                prev["text"] += seg["text"]
                prev["end"] = seg["end"]
                continue
        final.append(seg)
    return final


def _clean(text: str) -> str:
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'<\|/[^|]*\|>', '', text)
    return text.strip()


def _classify_reason(text: str) -> str:
    t = text.lower()
    for kw in ['冗余', '铺垫', '重复表述', 'redundant', '过渡']:
        if kw in t: return "redundant"
    for kw in ['附和', '嗯嗯', '对对', 'echo', '应和', '无观点', '催促']:
        if kw in t: return "echo"
    for kw in ['杂音', '笑声', '咳嗽', 'noise', '非语言']:
        if kw in t: return "noise"
    return "filler"


def _suggestion_to_isKept(suggestion: str) -> bool:
    return suggestion != "strong"


# ============================================
# Task persistence
# ============================================
def _save_task_to_disk(task_id: str):
    if task_id not in tasks:
        return
    task = tasks[task_id]
    if task.get("status") != "completed":
        return
    record = {
        "task_id": task_id,
        "filename": task.get("filename", ""),
        "created_at": task.get("created_at", ""),
        "segments": task.get("segments", []),
        "speakers": task.get("speakers", []),
        "chapters": task.get("chapters", []),
    }
    history = _load_history()
    found = False
    for i, h in enumerate(history):
        if h.get("task_id") == task_id:
            history[i] = record
            found = True
            break
    if not found:
        history.insert(0, record)
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def _load_history() -> List[dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return []


def _load_task_from_disk(task_id: str) -> Optional[dict]:
    history = _load_history()
    for h in history:
        if h.get("task_id") == task_id:
            return h
    return None


# ============================================
# Local ASR via FunASR
# ============================================
async def _local_transcribe(file_path: str, task_id: str) -> List[dict]:
    model = _get_local_asr()
    if model is None or model is False:
        raise Exception("Local ASR model not available")

    logger.info(f"[{task_id}] Running local SenseVoice transcription...")

    result = await asyncio.wait_for(
        asyncio.to_thread(
            model.generate,
            input=file_path,
            language="zh",
            use_itn=True,
            batch_size_s=300,
        ),
        timeout=600.0,
    )

    raw_segments = []
    if result and len(result) > 0:
        for res in result:
            if isinstance(res, dict):
                text = res.get("text", "")
                timestamp = res.get("timestamp", [])
                if timestamp and isinstance(timestamp, list):
                    for ts in timestamp:
                        if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                            start_ms = ts[0]
                            end_ms = ts[1]
                            seg_text = str(ts[2]).strip() if len(ts) > 2 else ""
                            seg_text = _clean(seg_text)
                            if seg_text:
                                raw_segments.append({
                                    "start": round(start_ms / 1000.0, 2),
                                    "end": round(end_ms / 1000.0, 2),
                                    "text": seg_text,
                                })
                elif text:
                    text = _clean(text)
                    if text:
                        raw_segments.append({"start": 0, "end": 0, "text": text})
            elif isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        timestamp = item.get("timestamp", [])
                        if timestamp and isinstance(timestamp, list):
                            for ts in timestamp:
                                if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                                    start_ms = ts[0]
                                    end_ms = ts[1]
                                    seg_text = str(ts[2]).strip() if len(ts) > 2 else ""
                                    seg_text = _clean(seg_text)
                                    if seg_text:
                                        raw_segments.append({
                                            "start": round(start_ms / 1000.0, 2),
                                            "end": round(end_ms / 1000.0, 2),
                                            "text": seg_text,
                                        })
                        elif text:
                            text = _clean(text)
                            if text:
                                raw_segments.append({"start": 0, "end": 0, "text": text})

    logger.info(f"[{task_id}] Local ASR raw segments: {len(raw_segments)}")
    return raw_segments


# ============================================
# DashScope ASR (fallback)
# ============================================
async def _upload_to_dashscope_oss(file_path: str, filename: str) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=120.0, write=300.0, pool=30.0)) as http_client:
        policy_resp = await http_client.get(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            params={"action": "getPolicy", "model": "sensevoice-v1"},
        )
        if policy_resp.status_code != 200:
            raise Exception(f"Failed to get upload policy: {policy_resp.status_code}")
        policy_data = policy_resp.json().get("data", {})
        upload_host = policy_data.get("upload_host", "")
        upload_dir = policy_data.get("upload_dir", "")
        if not upload_host or not upload_dir:
            raise Exception(f"Invalid upload policy: {policy_data}")
        key = f"{upload_dir}/{filename}"
        with open(file_path, "rb") as f:
            file_content = f.read()
        form_fields = {
            "OSSAccessKeyId": policy_data.get("oss_access_key_id", ""),
            "Signature": policy_data.get("signature", ""),
            "policy": policy_data.get("policy", ""),
            "x-oss-object-acl": policy_data.get("x_oss_object_acl", ""),
            "x-oss-forbid-overwrite": policy_data.get("x_oss_forbid_overwrite", ""),
            "key": key,
            "success_action_status": "200",
        }
        files = {"file": (filename, file_content)}
        upload_resp = await http_client.post(upload_host, data=form_fields, files=files)
        if upload_resp.status_code not in (200, 201, 204):
            raise Exception(f"OSS upload failed: {upload_resp.status_code}")
        return f"oss://{key}"


async def _dashscope_transcribe(file_path: str, filename: str, task_id: str) -> List[dict]:
    oss_url = await _upload_to_dashscope_oss(file_path, filename)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)) as http_client:
        submit_resp = await http_client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json={
                "model": "sensevoice-v1",
                "input": {"file_urls": [oss_url]},
                "parameters": {"language_hints": ["zh"]},
            },
        )
        if submit_resp.status_code not in (200, 201):
            raise Exception(f"Transcription submit failed: {submit_resp.status_code}")
        submit_data = submit_resp.json()
        ds_task_id = submit_data.get("output", {}).get("task_id")
        if not ds_task_id:
            raise Exception(f"No task_id: {submit_data}")
        while True:
            poll_resp = await http_client.get(
                f"https://dashscope.aliyuncs.com/api/v1/tasks/{ds_task_id}",
                headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            )
            poll_data = poll_resp.json()
            status = poll_data.get("output", {}).get("task_status", "")
            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise Exception(f"Transcription failed: {poll_data.get('output', {}).get('message', '')}")
            await asyncio.sleep(3)

    raw_segments = []
    results = poll_data.get("output", {}).get("results", [])
    try:
        for r in results:
            url = r.get("transcription_url", "")
            if url:
                data = await _fetch_transcription_url(url)
                raw_segments = _parse_transcription_data(data)
                if raw_segments:
                    break
        if not raw_segments:
            for r in results:
                for t in r.get("transcripts", []):
                    text = t.get("text", "")
                    if not text:
                        continue
                    for s in t.get("sentences", []):
                        s_text = s.get("text", "")
                        if not s_text:
                            continue
                        raw_segments.append({
                            "start": round(s.get("begin_time", 0) / 1000.0, 2),
                            "end": round(s.get("end_time", 0) / 1000.0, 2),
                            "text": s_text.strip(),
                        })
    except Exception as e:
        logger.error(f"[{task_id}] Parse error: {e}")

    return raw_segments


async def _fetch_transcription_url(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)) as c:
            r = await c.get(url)
            if r.status_code == 200:
                return r.json()
    except:
        pass
    return {}


def _parse_transcription_data(data: dict) -> List[dict]:
    segments = []
    try:
        for t in data.get("transcripts", []):
            text = _clean(t.get("text", ""))
            if not text:
                continue
            for s in t.get("sentences", []):
                s_text = _clean(s.get("text", ""))
                if not s_text:
                    continue
                segments.append({
                    "start": round(s.get("begin_time", 0) / 1000.0, 2),
                    "end": round(s.get("end_time", 0) / 1000.0, 2),
                    "text": s_text,
                })
    except:
        pass
    return segments


# ============================================
# Unified transcribe function
# ============================================
async def upload_and_transcribe(file_path: str, filename: str, task_id: str) -> List[dict]:
    raw_segments = []

    if ASR_MODE == "local":
        try:
            logger.info(f"[{task_id}] Using LOCAL SenseVoice ASR")
            raw_segments = await _local_transcribe(file_path, task_id)
            if raw_segments:
                logger.info(f"[{task_id}] Local ASR succeeded: {len(raw_segments)} segments")
            else:
                raise Exception("Local ASR returned empty results")
        except Exception as e:
            logger.warning(f"[{task_id}] Local ASR failed: {e}, falling back to DashScope...")
            raw_segments = await _dashscope_transcribe(file_path, filename, task_id)
    else:
        logger.info(f"[{task_id}] Using DashScope ASR")
        raw_segments = await _dashscope_transcribe(file_path, filename, task_id)

    if not raw_segments:
        return []

    merged = merge_segments(raw_segments)
    segments = []
    for i, seg in enumerate(merged):
        segments.append({
            "id": i + 1, "speaker": "Speaker_A", "text": seg["text"],
            "startTime": seg["start"], "endTime": seg["end"],
            "suggestion": "keep", "isKept": True,
            "reason": None, "reasonDetail": None,
        })
    logger.info(f"[{task_id}] After merging: {len(segments)} segments")
    return segments


# ============================================
# Qwen analysis
# ============================================
def parse_json_response(response_text: str):
    for prefix in ["```json", "```"]:
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    obj_start = response_text.find('{')
    arr_start = response_text.find('[')
    if arr_start >= 0 and (obj_start < 0 or arr_start < obj_start):
        start = arr_start
        end = response_text.rfind(']')
    elif obj_start >= 0:
        start = obj_start
        end = response_text.rfind('}')
    else:
        return None

    if end > start:
        chunk = response_text[start:end + 1]
        try:
            return json.loads(chunk)
        except:
            pass
        try:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', chunk))
        except:
            pass
    return None


async def _call_qwen(prompt: str, task_id: str, label: str, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    qwen_client.chat.completions.create,
                    model=QWEN_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=300.0,
            )
            return resp.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            if attempt < retries:
                await asyncio.sleep(5)
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ['503', '429', 'overload', 'rate_limit']) and attempt < retries:
                await asyncio.sleep(10 * attempt)
            else:
                break
    return None


def _extract_speakers(segments: List[dict]) -> List[dict]:
    smap = {}
    for seg in segments:
        spk = seg.get("speaker", "Speaker_A")
        if spk not in smap:
            smap[spk] = {"id": spk, "role": seg.get("speakerRole", "guest"), "name": spk.replace("_", " ")}
        elif seg.get("speakerRole") == "host":
            smap[spk]["role"] = "host"
    return list(smap.values())


def _auto_chapter(segments: List[dict], gap: float = 30.0) -> List[dict]:
    if not segments:
        return []
    chapters = []
    ch_start = segments[0]["startTime"]
    ch_end = segments[0]["endTime"]
    ch_idx = 0
    for i in range(1, len(segments)):
        if segments[i]["startTime"] - segments[i - 1]["endTime"] >= gap:
            chapters.append({"title": segments[ch_idx]["text"][:15] + "...", "startTime": round(ch_start, 2), "endTime": round(ch_end, 2)})
            ch_start = segments[i]["startTime"]
            ch_idx = i
        ch_end = segments[i]["endTime"]
    chapters.append({"title": segments[ch_idx]["text"][:15] + "...", "startTime": round(ch_start, 2), "endTime": round(ch_end, 2)})
    if len(chapters) > 7:
        step = len(chapters) // 5
        chapters = chapters[::step][:7]
    return chapters


def _normalize_suggestion(val) -> str:
    s = str(val).lower().strip()
    if s in ("strong", "delete", "remove", "del", "删除", "强烈"):
        return "strong"
    if s in ("mild", "maybe", "consider", "一般", "建议", "可选"):
        return "mild"
    return "keep"


async def analyze_with_qwen(segments: List[dict], task_id: str) -> dict:
    BATCH_SIZE = 50
    all_analyzed = []

    for batch_start in range(0, len(segments), BATCH_SIZE):
        batch = segments[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE

        lines = [f"[{format_time(s['startTime'])} - {format_time(s['endTime'])}] {s['speaker']}: {s['text']}" for s in batch]
        prompt = QWEN_BATCH_PROMPT.format(transcript_text="\n".join(lines))

        response_text = await _call_qwen(prompt, task_id, f"Batch {batch_num}/{total}")
        if not response_text:
            all_analyzed.extend(batch)
            continue

        parsed = parse_json_response(response_text)
        if parsed and isinstance(parsed, dict):
            bs = parsed.get("segments", [])
            if bs and isinstance(bs, list):
                all_analyzed.extend(bs)
            else:
                all_analyzed.extend(batch)
        elif parsed and isinstance(parsed, list):
            all_analyzed.extend(parsed)
        else:
            all_analyzed.extend(batch)

    result_segments = []
    for i, seg in enumerate(segments):
        if i < len(all_analyzed) and isinstance(all_analyzed[i], dict):
            a = all_analyzed[i]
            suggestion = _normalize_suggestion(a.get("suggestion", "keep"))
            if suggestion == "keep" and not a.get("isKept", True):
                suggestion = "strong"
            reason_val = a.get("reason") if suggestion != "keep" else None
            reason_detail = a.get("reasonDetail") if suggestion != "keep" else None
            if reason_val and reason_val not in ("filler", "echo", "noise", "redundant"):
                reason_val = _classify_reason(str(reason_val))
            if suggestion != "keep" and not reason_val:
                reason_val = _classify_reason(reason_detail or "")
            result_segments.append({
                "id": seg["id"],
                "speaker": a.get("speaker", seg["speaker"]),
                "speakerRole": a.get("speakerRole", "guest"),
                "text": seg["text"],
                "startTime": seg["startTime"],
                "endTime": seg["endTime"],
                "suggestion": suggestion,
                "isKept": _suggestion_to_isKept(suggestion),
                "reason": reason_val,
                "reasonDetail": reason_detail or reason_val,
            })
        else:
            result_segments.append({**seg, "speakerRole": "guest", "reasonDetail": None})

    speakers = _extract_speakers(result_segments)

    valid_chapters = await _generate_chapters(result_segments, task_id)
    if not valid_chapters:
        valid_chapters = _auto_chapter(result_segments)

    return {"segments": result_segments, "speakers": speakers, "chapters": valid_chapters}


async def _generate_chapters(segments: List[dict], task_id: str) -> List[dict]:
    if not segments:
        return []

    kept = [s for s in segments if s.get("isKept", True)]
    source = kept if len(kept) > 10 else segments

    lines = [f"[{format_time(s['startTime'])} - {format_time(s['endTime'])}] {s.get('speaker', 'Speaker_A')}: {s['text']}" for s in source]
    full_text = "\n".join(lines)

    if len(full_text) > 30000:
        step = max(1, len(source) // 200)
        sampled = source[::step]
        lines = [f"[{format_time(s['startTime'])} - {format_time(s['endTime'])}] {s.get('speaker', 'Speaker_A')}: {s['text']}" for s in sampled]
        full_text = "\n".join(lines)

    prompt = QWEN_CHAPTER_PROMPT.format(transcript_text=full_text)
    response_text = await _call_qwen(prompt, task_id, "Chapter generation")

    if not response_text:
        return []

    parsed = parse_json_response(response_text)
    if not parsed:
        return []

    chapters = parsed.get("chapters", []) if isinstance(parsed, dict) else []
    if not chapters or not isinstance(chapters, list):
        return []

    valid = []
    for ch in chapters:
        if not isinstance(ch, dict) or not ch.get("title"):
            continue
        title = str(ch["title"]).strip()
        st = float(ch.get("startTime", 0))
        et = float(ch.get("endTime", 0))
        if et <= st:
            et = st + 60
        valid.append({"title": title, "startTime": round(st, 2), "endTime": round(et, 2)})

    if len(valid) > 7:
        valid = valid[:7]

    if valid:
        valid[0]["startTime"] = segments[0]["startTime"]
        valid[-1]["endTime"] = segments[-1]["endTime"]

    return valid


# ============================================
# Background task processor
# ============================================
async def _process_audio_task(task_id: str, file_path: str, filename: str):
    try:
        tasks[task_id]["status"] = "transcribing"
        asr_label = "本地语音转文字" if ASR_MODE == "local" else "千问语音转文字"
        tasks[task_id]["progress"] = asr_label + "..."
        segments = await upload_and_transcribe(file_path, filename, task_id)
        if not segments:
            raise Exception("语音转文字返回空结果，音频可能为静音或已损坏")
        tasks[task_id]["status"] = "analyzing"
        tasks[task_id]["progress"] = f"千问文本分析... (共 {len(segments)} 段)"
        analysis = await analyze_with_qwen(segments, task_id)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = "完成!"
        tasks[task_id]["segments"] = analysis["segments"]
        tasks[task_id]["speakers"] = analysis["speakers"]
        tasks[task_id]["chapters"] = analysis["chapters"]
        tasks[task_id]["message"] = f"处理完成，共 {len(analysis['segments'])} 段"
        _save_task_to_disk(task_id)
    except Exception as e:
        err_msg = str(e) or f"{type(e).__name__}: 未知错误"
        logger.error(f"[{task_id}] Failed: {err_msg}")
        logger.error(traceback.format_exc())
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = err_msg
        tasks[task_id]["progress"] = f"处理失败: {err_msg}"


# ============================================
# Export helpers
# ============================================
def _generate_word(task_data: dict, speaker_names: dict) -> str:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(11)

    title = doc.add_heading('FX281 Studio 播客粗剪文稿', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    segments = task_data.get("segments", [])
    chapters = task_data.get("chapters", [])
    speakers = task_data.get("speakers", [])

    kept = [s for s in segments if s.get("isKept", True)]
    removed = [s for s in segments if not s.get("isKept", True)]
    mild_count = len([s for s in segments if s.get("suggestion") == "mild"])
    strong_count = len([s for s in segments if s.get("suggestion") == "strong"])

    p = doc.add_paragraph()
    summary = f'原始段数: {len(segments)} | 保留: {len(kept)} | 建议删减: {len(removed)}'
    if mild_count or strong_count:
        summary += f' (一般建议: {mild_count}, 强烈建议: {strong_count})'
    p.add_run(summary).font.size = Pt(9)
    speaker_lines = []
    for s in speakers:
        name = speaker_names.get(s["id"], s["id"])
        role = "主持人" if s["role"] == "host" else "被访人"
        speaker_lines.append(f"{name}({role})")
    p.add_run(f'\n说话人: {", ".join(speaker_lines)}').font.size = Pt(9)

    doc.add_paragraph()

    suggestion_labels = {"mild": "一般建议删减", "strong": "强烈建议删减"}

    if chapters:
        for i, ch in enumerate(chapters):
            doc.add_heading(f'{i+1}. {ch["title"]}', level=2)
            ch_segs = [s for s in segments if (s.get("startTime",0) >= ch.get("startTime",0) - 0.5) and (s.get("startTime",0) < ch.get("endTime",0) + 0.5)]
            for seg in ch_segs:
                p = doc.add_paragraph()
                name = speaker_names.get(seg.get("speaker",""), seg.get("speaker",""))
                time_str = format_time(seg.get("startTime", 0))
                run = p.add_run(f'[{time_str}] {name}: ')
                run.font.size = Pt(10)
                run.font.bold = True
                suggestion = seg.get("suggestion", "keep")
                if seg.get("isKept", True) and suggestion == "keep":
                    p.add_run(seg["text"]).font.size = Pt(10)
                elif suggestion == "mild":
                    run_text = p.add_run(seg["text"])
                    run_text.font.size = Pt(10)
                    run_text.font.color.rgb = RGBColor(128, 128, 128)
                    reason = seg.get("reason", "")
                    detail = seg.get("reasonDetail", "")
                    tag = suggestion_labels.get(suggestion, suggestion)
                    if reason or detail:
                        run_tag = p.add_run(f'  [{tag}] {reason}: {detail}')
                        run_tag.font.size = Pt(8)
                        run_tag.font.color.rgb = RGBColor(150, 150, 150)
                else:
                    run_del = p.add_run(seg["text"])
                    run_del.font.size = Pt(10)
                    run_del.font.strike = True
                    run_del.font.color.rgb = RGBColor(200, 50, 50)
                    reason = seg.get("reason", "")
                    detail = seg.get("reasonDetail", "")
                    tag = suggestion_labels.get(suggestion, suggestion)
                    if reason or detail:
                        run_tag = p.add_run(f'  [{tag}] {reason}: {detail}')
                        run_tag.font.size = Pt(8)
                        run_tag.font.color.rgb = RGBColor(150, 150, 150)
    else:
        for seg in segments:
            p = doc.add_paragraph()
            name = speaker_names.get(seg.get("speaker",""), seg.get("speaker",""))
            time_str = format_time(seg.get("startTime", 0))
            p.add_run(f'[{time_str}] {name}: ').font.size = Pt(10)
            suggestion = seg.get("suggestion", "keep")
            if seg.get("isKept", True) and suggestion == "keep":
                p.add_run(seg["text"]).font.size = Pt(10)
            elif suggestion == "mild":
                run_text = p.add_run(seg["text"])
                run_text.font.color.rgb = RGBColor(128, 128, 128)
            else:
                run_del = p.add_run(seg["text"])
                run_del.font.strike = True
                run_del.font.color.rgb = RGBColor(200, 50, 50)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(tmp.name)
    tmp.close()
    return tmp.name


def _generate_mp3(task_data: dict, original_file_path: str) -> str:
    from pydub import AudioSegment
    audio = AudioSegment.from_file(original_file_path)
    segments = task_data.get("segments", [])
    kept_segments = [s for s in segments if s.get("isKept", True)]
    if not kept_segments:
        return original_file_path

    kept_segments.sort(key=lambda s: s.get("startTime", 0))
    result = AudioSegment.empty()
    for seg in kept_segments:
        start_ms = int(seg.get("startTime", 0) * 1000)
        end_ms = int(seg.get("endTime", 0) * 1000)
        if end_ms > start_ms:
            chunk = audio[start_ms:end_ms]
            result += chunk

    if len(result) == 0:
        return original_file_path

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    result.export(tmp.name, format="mp3", bitrate="192k")
    tmp.close()
    return tmp.name


# ============================================
# API Routes
# ============================================
@app.get("/")
async def root():
    return {
        "status": "ok", "service": "FX281 API", "version": "12.0.0",
        "asr_mode": ASR_MODE, "qwen_model": QWEN_MODEL,
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "asr_mode": ASR_MODE}


@app.get("/api/history")
async def get_history():
    history = _load_history()
    summaries = []
    for h in history:
        summaries.append({
            "task_id": h.get("task_id"),
            "filename": h.get("filename", ""),
            "created_at": h.get("created_at", ""),
            "segment_count": len(h.get("segments", [])),
            "kept_count": len([s for s in h.get("segments", []) if s.get("isKept", True)]),
            "mild_count": len([s for s in h.get("segments", []) if s.get("suggestion") == "mild"]),
            "strong_count": len([s for s in h.get("segments", []) if s.get("suggestion") == "strong"]),
        })
    return JSONResponse(content=summaries)


@app.get("/api/history/{task_id}")
async def get_history_detail(task_id: str):
    record = _load_task_from_disk(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found in history")
    return JSONResponse(content=record)


@app.delete("/api/history/{task_id}")
async def delete_history(task_id: str):
    history = _load_history()
    new_history = [h for h in history if h.get("task_id") != task_id]
    if len(new_history) == len(history):
        raise HTTPException(status_code=404, detail="Task not found in history")
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")
    if task_id in tasks:
        del tasks[task_id]
    return JSONResponse(content={"status": "ok", "deleted": task_id})


@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed = ['.mp3', '.m4a', '.wav', '.flac', '.mp4', '.aac', '.ogg']
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            content = await file.read()
            f.write(content)
            tmp = f.name
    except Exception as e:
        if tmp and os.path.exists(tmp): os.remove(tmp)
        raise HTTPException(status_code=500, detail=str(e))

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "uploaded", "progress": "文件已上传...",
        "segments": None, "speakers": None, "chapters": None,
        "error": None, "message": None, "filename": file.filename,
        "original_file": tmp,
        "created_at": datetime.now().isoformat(),
    }
    asyncio.create_task(_process_audio_task(task_id, tmp, file.filename))
    return JSONResponse(content={"task_id": task_id, "status": "uploaded"})


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        record = _load_task_from_disk(task_id)
        if record:
            return JSONResponse(content={
                "task_id": task_id, "status": "completed",
                "progress": "完成!", "message": f"处理完成，共 {len(record.get('segments', []))} 段",
                "error": None,
                "segments": record.get("segments", []),
                "speakers": record.get("speakers", []),
                "chapters": record.get("chapters", []),
            })
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    resp = {
        "task_id": task_id, "status": task["status"],
        "progress": task["progress"], "message": task.get("message"),
        "error": task.get("error"),
    }
    if task["status"] == "completed":
        resp["segments"] = task.get("segments", [])
        resp["speakers"] = task.get("speakers", [])
        resp["chapters"] = task.get("chapters", [])
    return JSONResponse(content=resp)


@app.post("/api/export/word/{task_id}")
async def export_word(task_id: str):
    task = tasks.get(task_id)
    if not task or task.get("status") != "completed":
        record = _load_task_from_disk(task_id)
        if record:
            task = record
        else:
            raise HTTPException(status_code=404, detail="Task not found or not completed")
    try:
        speaker_names = {s["id"]: s.get("name", s["id"]) for s in task.get("speakers", [])}
        path = _generate_word(task, speaker_names)
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename="FX281_粗剪文稿.docx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/mp3/{task_id}")
async def export_mp3(task_id: str):
    task = tasks.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Task not found or not completed")
    original = task.get("original_file")
    if not original or not os.path.exists(original):
        raise HTTPException(status_code=400, detail="Original audio file not available")
    try:
        path = _generate_mp3(task, original)
        return FileResponse(path, media_type="audio/mpeg", filename="FX281_粗剪版.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist")

if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("=" * 50)
    logger.info("FX281 Podcast Processing API v12.0")
    logger.info(f"ASR: {'Local SenseVoice (FunASR)' if ASR_MODE == 'local' else 'DashScope sensevoice-v1'}")
    logger.info(f"Analysis: {QWEN_MODEL} (DashScope)")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info(f"Port: {port}")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
