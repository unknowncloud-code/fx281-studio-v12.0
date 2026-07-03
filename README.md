# FX281 Studio

播客访谈音频的 AI 粗剪原型：本地完成 VAD 与语音转写，云端大模型只接收文本并生成逐句编辑建议、章节和概述。编辑在界面中逐条确认后，系统才按人工结果导出 Word 与 MP3。

## 能力边界

- App 当前提供本地 ASR、文本级建议、章节、概述、人工勾选和 Word/MP3 导出。
- `strong / mild / keep` 都是建议，初始不会自动删除；只有用户取消勾选后才进入裁剪结果。
- LearnBuddy Skill 生成的 `final-decision.json` 尚未接入 App，`partial_trim` 也未自动执行；相关 Figma 页面属于交互原型。
- 句子时间戳由 VAD 段内按文本长度近似分配，适合粗剪定位，不等同于逐词强制对齐。
- 说话人角色由文本上下文推断，不是声纹级说话人识别。

## 数据与隐私

- 原始音频在本机完成 ASR，不上传给 DeepSeek。
- 转写文本会发送至 DeepSeek 做内容分析。
- 完成任务会把转写结果写入本机 `backend/data/history.json`；用户可在历史记录中删除。
- 不要把 API Key 写进源码。使用环境变量 `DEEPSEEK_API_KEY`。

## 环境要求

- Node.js 20+
- Python 3.10+
- FFmpeg
- PyTorch / torchaudio
- DeepSeek API Key

首次运行可能需要下载 SenseVoiceSmall 与 FSMN-VAD 模型，因此不能视为完全离线或零配置安装。

## 启动

### 1. 后端

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio
pip install -r requirements.txt
export DEEPSEEK_API_KEY="你的本机密钥"
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. 前端

```bash
npm install
npm run dev
```

打开 Vite 输出的本地地址。生产构建：

```bash
npm run build
```

## 当前架构

```text
音频文件
  -> 本地 FSMN-VAD
  -> 本地 SenseVoiceSmall
  -> 近似句段时间戳
  -> DeepSeek 文本分析
  -> AI 建议（不执行删除）
  -> 人工勾选
  -> 同步人工决策
  -> Word / MP3 导出
```

## 比赛演示说明

LearnBuddy Skill 与 App 当前为松耦合验证：Skill 负责结构诊断、风险扫描、目标时长适配和结构化决策；App 展示既有的音频转写与人工粗剪能力。演示中不得宣称 App 已导入或执行 `final-decision.json`，除非后续完成真实接口验证。
