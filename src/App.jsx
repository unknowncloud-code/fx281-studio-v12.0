import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Play, Pause, Square, CheckSquare, Upload, Download, Pencil, SkipForward, X, FileText, Music, Menu, Clock } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const SPEAKER_COLORS = ['#1a1a1a', '#4a4a4a', '#7a7a7a', '#aaa', '#333', '#666'];

const CLOUD_BACKEND = 'https://fx281-studio-v12-0-backend.onrender.com';
const getApiBase = () => {
  if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    return CLOUD_BACKEND;
  }
  return '';
};
const API_BASE = typeof window !== 'undefined' ? getApiBase() : '';

const REASON_LABELS = {
  filler: '口癖', echo: '附和', noise: '杂音', redundant: '冗余',
};

const REASON_COLORS = {
  filler: 'bg-amber-50 text-amber-600 border-amber-200',
  echo: 'bg-purple-50 text-purple-600 border-purple-200',
  noise: 'bg-red-50 text-red-600 border-red-200',
  redundant: 'bg-blue-50 text-blue-600 border-blue-200',
};

const SUGGESTION_LABELS = {
  keep: '保留', mild: '一般建议删减', strong: '强烈建议删减',
};

function formatTime(seconds) {
  if (!seconds || seconds < 0) return '0:00';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function App() {
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWsId, setActiveWsId] = useState(null);
  const [showExport, setShowExport] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [historyList, setHistoryList] = useState([]);

  const audioRef = useRef(null);
  const progressRef = useRef(null);
  const prevAudioUrlRef = useRef(null);
  const segmentRefs = useRef({});
  const previewTimeoutRef = useRef(null);
  const draggingRef = useRef(false);

  const activeWs = workspaces.find(w => w.id === activeWsId) || null;

  const updateWs = useCallback((id, updates) => {
    setWorkspaces(prev => prev.map(w => w.id === id ? { ...w, ...updates } : w));
  }, []);

  const transcripts = activeWs?.transcripts || [];
  const speakers = activeWs?.speakers || [];
  const chapters = activeWs?.chapters || [];
  const speakerNames = activeWs?.speakerNames || {};
  const editingSpeaker = activeWs?.editingSpeaker || null;
  const activeTab = activeWs?.activeTab || '编辑模式';
  const isPlaying = activeWs?.isPlaying || false;
  const audioDuration = activeWs?.audioDuration || 0;
  const currentTime = activeWs?.currentTime || 0;
  const activeSegmentId = activeWs?.activeSegmentId || null;
  const isUploading = activeWs?.isUploading || false;
  const uploadStep = activeWs?.uploadStep || 0;
  const uploadProgress = activeWs?.uploadProgress || '';
  const error = activeWs?.error || null;
  const audioUrl = activeWs?.audioUrl || null;
  const taskId = activeWs?.taskId || null;
  const hasData = transcripts.length > 0;

  const displayedTranscripts = activeTab === '粗剪预览' ? transcripts.filter(t => t.isKept) : transcripts;
  const keptTranscripts = transcripts.filter(t => t.isKept);
  const removedTranscripts = transcripts.filter(t => !t.isKept);
  const mildTranscripts = transcripts.filter(t => t.suggestion === 'mild');
  const strongTranscripts = transcripts.filter(t => t.suggestion === 'strong');
  const originalDuration = audioDuration || (transcripts.length > 0 ? Math.max(...transcripts.map(t => t.endTime || 0)) : 0);
  const keptDuration = keptTranscripts.reduce((sum, t) => sum + ((t.endTime || 0) - (t.startTime || 0)), 0);
  const removedDuration = originalDuration - keptDuration;

  const speakerMap = {};
  keptTranscripts.forEach(t => {
    const dur = (t.endTime || 0) - (t.startTime || 0);
    if (!speakerMap[t.speaker]) speakerMap[t.speaker] = 0;
    speakerMap[t.speaker] += dur;
  });
  const totalSpeakerDuration = Object.values(speakerMap).reduce((a, b) => a + b, 0) || 1;
  const pieData = Object.entries(speakerMap).map(([name, value], i) => ({
    name, value: Math.round((value / totalSpeakerDuration) * 100), duration: value,
    color: SPEAKER_COLORS[i % SPEAKER_COLORS.length],
  }));

  const getSpeakerName = (id) => speakerNames[id] || id?.replace(/_/g, ' ') || 'Unknown';
  const getSpeakerRole = (id) => speakers.find(s => s.id === id)?.role || 'guest';

  const reasonStats = {};
  removedTranscripts.forEach(t => { const r = t.reason || 'other'; reasonStats[r] = (reasonStats[r] || 0) + 1; });

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTimeUpdate = () => {
      if (activeWsId) updateWs(activeWsId, { currentTime: audio.currentTime });
    };
    const onLoadedMetadata = () => {
      if (activeWsId) updateWs(activeWsId, { audioDuration: audio.duration });
    };
    const onPlay = () => {
      if (activeWsId) updateWs(activeWsId, { isPlaying: true });
    };
    const onPause = () => {
      if (activeWsId) updateWs(activeWsId, { isPlaying: false });
    };
    const onEnded = () => {
      if (activeWsId) updateWs(activeWsId, { isPlaying: false, currentTime: 0 });
    };
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onEnded);
    };
  }, [activeWsId, updateWs]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    if (audioUrl) {
      audio.src = audioUrl;
      audio.load();
    } else {
      audio.removeAttribute('src');
      audio.load();
    }
  }, [audioUrl]);

  useEffect(() => {
    if (transcripts.length === 0 || !activeWsId) return;
    const active = transcripts.find(t => currentTime >= (t.startTime || 0) && currentTime < (t.endTime || 0));
    updateWs(activeWsId, { activeSegmentId: active ? active.id : null });
  }, [currentTime, transcripts, activeWsId, updateWs]);

  const toggleKeep = (id) => {
    if (!activeWsId) return;
    updateWs(activeWsId, { transcripts: transcripts.map(t => t.id === id ? { ...t, isKept: !t.isKept } : t) });
  };

  const jumpToSegment = (segment) => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;
    audio.currentTime = segment.startTime || 0;
    audio.play();
  };

  const playSegment = (segment) => {
    const audio = audioRef.current;
    if (!audio || !audioUrl || !activeWsId) return;
    if (previewTimeoutRef.current) clearTimeout(previewTimeoutRef.current);
    audio.currentTime = segment.startTime || 0;
    audio.play();
    const duration = (segment.endTime || 0) - (segment.startTime || 0);
    previewTimeoutRef.current = setTimeout(() => { audio.pause(); updateWs(activeWsId, { isPlaying: false }); }, duration * 1000 + 200);
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioUrl || activeTab !== '粗剪预览' || !isPlaying) return;

    const onTimeUpdate = () => {
      const t = audio.currentTime;
      const cur = transcripts.find(s => t >= (s.startTime || 0) && t < (s.endTime || 0));
      if (cur && !cur.isKept) {
        const next = transcripts.find(s => s.isKept && (s.startTime || 0) >= (cur.endTime || 0));
        if (next) {
          audio.currentTime = next.startTime || 0;
        } else {
          audio.pause();
        }
      }
    };

    audio.addEventListener('timeupdate', onTimeUpdate);
    return () => audio.removeEventListener('timeupdate', onTimeUpdate);
  }, [isPlaying, activeTab, transcripts, audioUrl]);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;
    if (isPlaying) {
      audio.pause();
    } else {
      if (activeTab === '粗剪预览') {
        const kept = transcripts.filter(t => t.isKept);
        if (kept.length === 0) return;
        let startSeg = kept.find(t => (t.endTime || 0) > currentTime) || kept[0];
        audio.currentTime = startSeg.startTime || 0;
      }
      audio.play();
    }
  };

  const seekTo = (ratio) => {
    const audio = audioRef.current;
    if (!audio || !audioUrl || !audioDuration) return;
    audio.currentTime = ratio * audioDuration;
  };

  const handleProgressMouseDown = (e) => {
    if (!progressRef.current || !audioDuration) return;
    draggingRef.current = true;
    const rect = progressRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    seekTo(Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)));
    const onMove = (ev) => {
      if (!draggingRef.current) return;
      const moveX = ev.touches ? ev.touches[0].clientX : ev.clientX;
      seekTo(Math.max(0, Math.min(1, (moveX - rect.left) / rect.width)));
    };
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove, { passive: true });
    window.addEventListener('touchend', onUp);
  };

  const loadHistory = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/history`);
      if (resp.ok) {
        const data = await resp.json();
        setHistoryList(data);
        setShowHistory(true);
      }
    } catch (e) {
      console.warn('Failed to load history:', e);
    }
  };

  const loadHistoryDetail = async (tid) => {
    try {
      const resp = await fetch(`${API_BASE}/api/history/${tid}`);
      if (!resp.ok) throw new Error('加载失败');
      const data = await resp.json();
      const names = {};
      (data.speakers || []).forEach(s => { names[s.id] = s.name || s.id.replace(/_/g, ' '); });
      const wsId = Date.now().toString();
      const ws = {
        id: wsId, filename: data.filename || '历史记录', taskId: tid,
        audioUrl: null,
        transcripts: data.segments || [], speakers: data.speakers || [], chapters: data.chapters || [],
        speakerNames: names, editingSpeaker: null,
        activeTab: '编辑模式', isPlaying: false,
        audioDuration: 0, currentTime: 0, activeSegmentId: null,
        isUploading: false, uploadStep: 0, uploadProgress: '',
        error: null,
      };
      setWorkspaces(prev => [...prev, ws]);
      setActiveWsId(wsId);
      setShowHistory(false);
    } catch (e) {
      console.warn('Failed to load history detail:', e);
    }
  };

  const deleteHistory = async (tid) => {
    try {
      const resp = await fetch(`${API_BASE}/api/history/${tid}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error('删除失败');
      setHistoryList(prev => prev.filter(h => h.task_id !== tid));
    } catch (e) {
      console.warn('Failed to delete history:', e);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const allowedExtensions = ['.mp3', '.m4a', '.wav', '.flac'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExtensions.includes(ext)) { return; }

    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);

    const localUrl = URL.createObjectURL(file);
    if (audioRef.current) { audioRef.current.pause(); }
    prevAudioUrlRef.current = localUrl;

    const wsId = Date.now().toString();
    const ws = {
      id: wsId, filename: file.name, taskId: null,
      audioUrl: localUrl,
      transcripts: [], speakers: [], chapters: [],
      speakerNames: {}, editingSpeaker: null,
      activeTab: '编辑模式', isPlaying: false,
      audioDuration: 0, currentTime: 0, activeSegmentId: null,
      isUploading: true, uploadStep: 1, uploadProgress: `上传中 (${fileSizeMB}MB)...`,
      error: null,
    };
    setWorkspaces(prev => [...prev, ws]);
    setActiveWsId(wsId);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const controller = new AbortController();
      const uploadTimeout = setTimeout(() => controller.abort(), 600000);
      const uploadResp = await fetch(`${API_BASE}/api/process-audio`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(uploadTimeout);
      if (!uploadResp.ok) throw new Error(`上传失败 (${uploadResp.status})`);
      const uploadData = await uploadResp.json();
      const tid = uploadData.task_id;
      if (!tid) throw new Error('未返回任务ID');

      updateWs(wsId, { taskId: tid, uploadStep: 2, uploadProgress: '本地语音转文字...' });

      const pollInterval = 3000;
      const maxTime = 3600000;
      const startTime = Date.now();

      while (Date.now() - startTime < maxTime) {
        try {
          const resp = await fetch(`${API_BASE}/api/task/${tid}`);
          if (!resp.ok) throw new Error('查询失败');
          const data = await resp.json();
          if (data.status === 'completed') {
            const names = {};
            (data.speakers || []).forEach(s => { names[s.id] = s.name || s.id.replace(/_/g, ' '); });
            updateWs(wsId, {
              transcripts: data.segments || [], speakers: data.speakers || [], chapters: data.chapters || [],
              speakerNames: names, isUploading: false, uploadStep: 4, uploadProgress: '完成!',
            });
            setTimeout(() => updateWs(wsId, { uploadStep: 0, uploadProgress: '' }), 1500);
            break;
          }
          if (data.status === 'failed') {
            const errMsg = data.error || '处理失败';
            updateWs(wsId, { error: `处理失败: ${errMsg}`, isUploading: false });
            break;
          }
          if (data.status === 'transcribing') updateWs(wsId, { uploadStep: 2, uploadProgress: '本地语音转文字...' });
          else if (data.status === 'analyzing') updateWs(wsId, { uploadStep: 3, uploadProgress: data.progress || '千问文本分析...' });
        } catch (err) {
          console.warn('[Poll] Error:', err.message);
          await new Promise(r => setTimeout(r, pollInterval));
        }
        await new Promise(r => setTimeout(r, pollInterval));
      }
    } catch (err) {
      updateWs(wsId, { error: `处理失败: ${err.message}`, isUploading: false });
    }
    event.target.value = '';
  };

  const removeWorkspace = (id) => {
    setWorkspaces(prev => prev.filter(w => w.id !== id));
    if (activeWsId === id) {
      setActiveWsId(prev => {
        const remaining = workspaces.filter(w => w.id !== id);
        return remaining.length > 0 ? remaining[0].id : null;
      });
    }
  };

  const jumpToChapter = (chapter) => {
    const audio = audioRef.current;
    if (audio && audioUrl) {
      audio.currentTime = chapter.startTime || 0;
      audio.play();
    }
    const chStart = chapter.startTime || 0;
    const firstSeg = transcripts.find(t => (t.startTime || 0) >= chStart - 1.0 && (t.startTime || 0) <= chStart + 30.0);
    if (firstSeg && segmentRefs.current[firstSeg.id]) {
      segmentRefs.current[firstSeg.id].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const updateSpeakerName = (id, newName) => {
    if (!activeWsId) return;
    updateWs(activeWsId, { speakerNames: { ...speakerNames, [id]: newName }, editingSpeaker: null });
  };

  const handleExportWord = async () => {
    if (!taskId) return;
    try {
      const resp = await fetch(`${API_BASE}/api/export/word/${taskId}`, { method: 'POST' });
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `导出失败 (${resp.status})`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'FX281_粗剪文稿.docx'; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { if (activeWsId) updateWs(activeWsId, { error: `导出Word失败: ${e.message}` }); }
  };

  const handleExportMp3 = async () => {
    if (!taskId) return;
    try {
      const resp = await fetch(`${API_BASE}/api/export/mp3/${taskId}`, { method: 'POST' });
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `导出失败 (${resp.status})`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'FX281_粗剪版.mp3'; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { if (activeWsId) updateWs(activeWsId, { error: `导出MP3失败: ${e.message}` }); }
  };

  const progressPercent = audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0;

  const sortedChapters = useMemo(() => {
    if (!chapters.length) return [];
    return [...chapters].sort((a, b) => (a.startTime || 0) - (b.startTime || 0));
  }, [chapters]);

  const getChapterForSegment = (seg) => {
    if (!sortedChapters.length) return null;
    const t = seg.startTime || 0;
    let matched = null;
    for (const ch of sortedChapters) {
      if ((ch.startTime || 0) <= t + 0.5) {
        matched = ch;
      } else {
        break;
      }
    }
    return matched;
  };

  const getSegmentStyle = (item) => {
    const sug = item.suggestion || 'keep';
    if (!item.isKept) {
      return 'line-through text-red-400';
    }
    if (sug === 'mild') {
      return 'text-gray-400';
    }
    if (sug === 'strong') {
      return 'line-through text-red-400';
    }
    return 'text-gray-800';
  };

  const getSegmentBg = (item) => {
    const sug = item.suggestion || 'keep';
    if (!item.isKept) return 'bg-red-50/40';
    if (sug === 'mild') return 'bg-amber-50/30';
    return '';
  };

  let lastChapterTitle = null;

  const renderSidebar = (isMobile) => (
    <>
      <div className="p-5 pb-3">
        <p className="text-[10px] font-medium text-gray-400 uppercase tracking-widest mb-2">时长</p>
        <div className="flex items-baseline gap-2 mb-0.5">
          <span className="text-base line-through text-gray-300 font-light">{hasData ? formatTime(originalDuration) : '--:--'}</span>
          <span className="text-gray-300 text-xs">→</span>
          <span className="text-2xl font-light tracking-tighter text-black">{hasData ? formatTime(keptDuration) : '--:--'}</span>
        </div>
        <p className="text-[10px] text-gray-400">{hasData ? `删减 ${formatTime(removedDuration)}，${removedTranscripts.length} 句` : '等待上传'}</p>
      </div>
      <hr className="border-gray-100 mx-5" />

      {hasData && (mildTranscripts.length > 0 || strongTranscripts.length > 0) && (
        <>
          <div className="p-5 pb-3">
            <p className="text-[10px] font-medium text-gray-400 uppercase tracking-widest mb-2">AI 建议</p>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-sm bg-gray-800"></div>
                <span className="text-xs text-gray-700">保留</span>
                <span className="text-[10px] text-gray-400 ml-auto">{transcripts.length - mildTranscripts.length - strongTranscripts.length} 句</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-sm bg-gray-400"></div>
                <span className="text-xs text-gray-400">一般建议删减</span>
                <span className="text-[10px] text-gray-400 ml-auto">{mildTranscripts.length} 句</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-sm bg-red-400"></div>
                <span className="text-xs text-red-400">强烈建议删减</span>
                <span className="text-[10px] text-gray-400 ml-auto">{strongTranscripts.length} 句</span>
              </div>
            </div>
          </div>
          <hr className="border-gray-100 mx-5" />
        </>
      )}

      <div className="p-5 pb-3">
        <p className="text-[10px] font-medium text-gray-400 uppercase tracking-widest mb-2">说话人</p>
        {speakers.length > 0 ? (
          <div className="space-y-1.5">
            {speakers.map((spk, i) => (
              <div key={spk.id} className="flex items-center gap-1.5 group">
                <div className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: SPEAKER_COLORS[i % SPEAKER_COLORS.length] }}></div>
                {editingSpeaker === spk.id ? (
                  <input className="flex-1 text-xs bg-gray-100 px-1.5 py-0.5 rounded outline-none focus:ring-1 ring-black"
                    defaultValue={getSpeakerName(spk.id)} autoFocus
                    onBlur={(e) => updateSpeakerName(spk.id, e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') updateSpeakerName(spk.id, e.target.value); }} />
                ) : (
                  <span className="text-xs text-gray-700 cursor-pointer hover:text-black flex-1 truncate" onClick={() => { if (activeWsId) updateWs(activeWsId, { editingSpeaker: spk.id }); }}>
                    {getSpeakerName(spk.id)}
                  </span>
                )}
                <span className={`text-[9px] px-1 py-0.5 rounded font-medium ${spk.role === 'host' ? 'bg-black text-white' : 'bg-gray-100 text-gray-500'}`}>
                  {spk.role === 'host' ? '主持人' : '被访人'}
                </span>
                <button onClick={() => { if (activeWsId) updateWs(activeWsId, { editingSpeaker: editingSpeaker === spk.id ? null : spk.id }); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"><Pencil className="w-2.5 h-2.5 text-gray-400" /></button>
              </div>
            ))}
            <div className="flex items-center gap-3 mt-2">
              <div className="w-14 h-14 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart><Pie data={pieData} cx="50%" cy="50%" innerRadius={0} outerRadius={26} dataKey="value" stroke="none">
                    {pieData.map((entry, index) => (<Cell key={index} fill={entry.color} />))}</Pie></PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-col gap-0.5 text-[10px]">
                {pieData.map(s => (<div key={s.name} className="flex justify-between gap-2"><span className="text-gray-500">{getSpeakerName(s.name)}</span><span className="font-mono text-gray-400">{s.value}%</span></div>))}
              </div>
            </div>
          </div>
        ) : <p className="text-[10px] text-gray-400">等待数据</p>}
      </div>
      <hr className="border-gray-100 mx-5" />

      <div className="p-5 pb-3">
        <p className="text-[10px] font-medium text-gray-400 uppercase tracking-widest mb-2">章节导览</p>
        {sortedChapters.length > 0 ? (
          <div className="space-y-0.5">
            {sortedChapters.map((ch, i) => (
              <button key={i} onClick={() => { jumpToChapter(ch); if (isMobile) setShowSidebar(false); }}
                className="w-full text-left px-1.5 py-1 rounded hover:bg-gray-50 transition-colors group">
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] font-mono text-gray-400 shrink-0">{formatTime(ch.startTime)}</span>
                  <span className="text-[11px] text-gray-600 group-hover:text-black truncate">{ch.title}</span>
                </div>
              </button>
            ))}
          </div>
        ) : <p className="text-[10px] text-gray-400">等待数据</p>}
      </div>

      {hasData && removedTranscripts.length > 0 && (
        <>
          <hr className="border-gray-100 mx-5" />
          <div className="p-5">
            <p className="text-[10px] font-medium text-gray-400 uppercase tracking-widest mb-2">删减原因</p>
            <div className="flex flex-wrap gap-1">
              {Object.entries(reasonStats).map(([reason, count]) => {
                const color = REASON_COLORS[reason] || 'bg-gray-50 text-gray-600 border-gray-200';
                return (<span key={reason} className={`text-[9px] px-1.5 py-0.5 rounded border font-medium ${color}`}>{REASON_LABELS[reason] || reason} ×{count}</span>);
              })}
            </div>
          </div>
        </>
      )}
    </>
  );

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 font-sans overflow-hidden">
      <audio ref={audioRef} preload="auto" onError={() => {}} />

      <header className="h-11 bg-black text-white flex items-center justify-between px-3 md:px-4 shrink-0 z-10 border-b border-gray-800">
        <div className="flex items-center gap-2 md:gap-3">
          {hasData && (
            <button onClick={() => setShowSidebar(true)} className="md:hidden p-1 -ml-1 hover:bg-gray-800 rounded">
              <Menu className="w-4 h-4" />
            </button>
          )}
          <span className="font-bold text-sm tracking-tight">FX281 Studio</span>
          {hasData && <span className="text-gray-500 text-[10px] hidden sm:inline">{transcripts.length}段 · {removedTranscripts.length}删 · {formatTime(keptDuration)}</span>}
        </div>
        <div className="flex items-center gap-2 md:gap-3">
          <button onClick={loadHistory}
            className="flex items-center gap-1.5 hover:bg-gray-800 text-gray-400 hover:text-white px-2 py-1 rounded-full transition-colors text-xs">
            <Clock className="w-3 h-3" /><span className="hidden sm:inline">历史</span>
          </button>
          <label className="flex items-center gap-1.5 bg-white hover:bg-gray-100 text-black px-2.5 md:px-3 py-1 rounded-full cursor-pointer transition-colors font-medium text-xs">
            <Upload className="w-3 h-3" /><span className="hidden sm:inline">上传录音</span>
            <input type="file" accept="audio/*,.mp3,.m4a,.wav,.flac" onChange={handleFileUpload} disabled={isUploading} className="hidden" />
          </label>
          <button onClick={() => setShowExport(true)} disabled={!hasData}
            className="flex items-center gap-1.5 bg-white hover:bg-gray-100 text-black px-2.5 md:px-3 py-1 rounded-full transition-colors font-medium text-xs disabled:opacity-30 disabled:cursor-not-allowed">
            <Download className="w-3 h-3" /><span className="hidden sm:inline">导出</span>
          </button>
        </div>
      </header>

      {workspaces.length > 0 && (
        <div className="flex items-center bg-gray-100 border-b border-gray-200 px-2 overflow-x-auto shrink-0">
          {workspaces.map(ws => (
            <div key={ws.id}
              onClick={() => setActiveWsId(ws.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs cursor-pointer border-b-2 transition-colors shrink-0 ${
                ws.id === activeWsId ? 'border-black text-black bg-white' : 'border-transparent text-gray-500 hover:text-black'
              }`}>
              <span className="truncate max-w-[120px]">{ws.filename}</span>
              {workspaces.length > 1 && (
                <button onClick={(e) => { e.stopPropagation(); removeWorkspace(ws.id); }}
                  className="hover:bg-gray-200 rounded p-0.5"><X className="w-3 h-3" /></button>
              )}
            </div>
          ))}
        </div>
      )}

      {!hasData && !isUploading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4 mx-auto">
              <Upload className="w-6 h-6 text-gray-400" />
            </div>
            <h3 className="text-base font-medium text-gray-700 mb-1">上传音频开始编辑</h3>
            <p className="text-gray-400 text-xs mb-4">支持 mp3, m4a, wav, flac 格式</p>
            <label className="inline-flex items-center gap-2 bg-black hover:bg-gray-800 text-white px-6 py-2 rounded-full cursor-pointer transition-colors font-medium text-sm">
              <Upload className="w-4 h-4" />
              <span>上传录音</span>
              <input type="file" accept="audio/*,.mp3,.m4a,.wav,.flac" onChange={handleFileUpload} className="hidden" />
            </label>
            <button onClick={loadHistory} className="block mx-auto mt-3 text-xs text-gray-400 hover:text-black transition-colors">
              或查看历史记录
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-1 overflow-hidden">
          {showSidebar && (
            <div className="fixed inset-0 z-40 md:hidden" onClick={() => setShowSidebar(false)}>
              <div className="absolute inset-0 bg-black/40"></div>
              <div className="absolute left-0 top-0 bottom-0 w-72 bg-white shadow-xl overflow-y-auto" onClick={e => e.stopPropagation()}>
                <div className="flex items-center justify-between p-4 border-b border-gray-100">
                  <span className="font-bold text-sm">信息面板</span>
                  <button onClick={() => setShowSidebar(false)} className="p-1 hover:bg-gray-100 rounded"><X className="w-4 h-4" /></button>
                </div>
                {renderSidebar(true)}
              </div>
            </div>
          )}

          <aside className="hidden md:flex w-64 flex-col border-r border-gray-200 overflow-y-auto bg-white shrink-0">
            {renderSidebar(false)}
          </aside>

          <main className="flex-1 flex flex-col relative bg-white overflow-hidden">
            <div className="flex-1 overflow-y-auto px-4 md:px-10 pt-4 md:pt-6 pb-28">
              <div className="max-w-3xl mb-4">
                <h1 className="text-lg font-semibold mb-0.5 text-black">粗剪逐字稿</h1>
                <p className="text-[10px] text-gray-400">
                  {hasData ? '审阅并编辑播客内容。勾选/取消勾选覆盖 AI 建议。' : '上传音频文件开始 AI 分析。'}
                </p>
              </div>

              {hasData && (
                <div className="flex gap-1 mb-6 bg-gray-100 p-0.5 rounded-full inline-flex">
                  {['编辑模式', '粗剪预览'].map(tab => (
                    <button key={tab} onClick={() => { if (activeWsId) updateWs(activeWsId, { activeTab: tab }); }}
                      className={`px-4 py-1 rounded-full text-[11px] font-medium transition-all ${activeTab === tab ? 'bg-white text-black shadow-sm' : 'bg-transparent text-gray-500 hover:text-black'}`}>
                      {tab}
                    </button>
                  ))}
                </div>
              )}

              {hasData && (
                <div className="max-w-3xl flex flex-col gap-0.5">
                  {displayedTranscripts.map((item) => {
                    const chapter = getChapterForSegment(item);
                    const showChapterDivider = chapter && chapter.title !== lastChapterTitle;
                    if (showChapterDivider) lastChapterTitle = chapter.title;
                    const suggestion = item.suggestion || 'keep';

                    return (
                      <React.Fragment key={item.id}>
                        {showChapterDivider && (
                          <div
                            className="flex items-center gap-3 py-2 mt-3 mb-1 cursor-pointer hover:bg-gray-50 rounded px-2 -mx-2"
                            onClick={() => jumpToChapter(chapter)}
                          >
                            <div className="flex-1 h-px bg-gray-200"></div>
                            <span className="text-[10px] font-medium text-gray-400 whitespace-nowrap">{chapter.title}</span>
                            <span className="text-[9px] font-mono text-gray-300">{formatTime(chapter.startTime)}</span>
                            <div className="flex-1 h-px bg-gray-200"></div>
                          </div>
                        )}
                        <div
                          ref={el => segmentRefs.current[item.id] = el}
                          onClick={() => jumpToSegment(item)}
                          className={`group flex items-start gap-2 px-2 py-1.5 rounded-lg transition-all cursor-pointer ${
                            activeSegmentId === item.id ? 'bg-gray-100 ring-1 ring-gray-200' : ''
                          } ${getSegmentBg(item)} ${item.isKept && suggestion === 'keep' ? 'hover:bg-gray-50' : ''}`}
                        >
                          <button onClick={(e) => { e.stopPropagation(); toggleKeep(item.id); }}
                            className={`mt-0.5 flex-shrink-0 transition-colors ${
                              item.isKept ? (suggestion === 'mild' ? 'text-gray-400' : 'text-black') : 'text-red-400'
                            }`}
                            disabled={activeTab === '粗剪预览'}>
                            {item.isKept ? <CheckSquare className="w-3.5 h-3.5" /> : <Square className="w-3.5 h-3.5" />}
                          </button>
                          <button onClick={(e) => { e.stopPropagation(); playSegment(item); }}
                            className="mt-0.5 flex-shrink-0 text-gray-300 hover:text-black transition-colors" title="播放此段">
                            <Play className="w-3 h-3" />
                          </button>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className={`text-[10px] font-medium ${getSpeakerRole(item.speaker) === 'host' ? 'font-semibold text-black' : 'text-gray-500'}`}>
                                {getSpeakerName(item.speaker)}
                              </span>
                              <span className="text-[9px] font-mono text-gray-400">{formatTime(item.startTime)}</span>
                              {suggestion !== 'keep' && activeTab === '编辑模式' && (
                                <span className={`text-[9px] px-1 py-0.5 rounded border font-medium ${
                                  suggestion === 'mild' ? 'bg-amber-50 text-amber-500 border-amber-200' : 'bg-red-50 text-red-500 border-red-200'
                                }`}>
                                  {SUGGESTION_LABELS[suggestion]}
                                </span>
                              )}
                            </div>
                            <p className={`text-xs leading-relaxed ${getSegmentStyle(item)}`}>{item.text}</p>
                            {suggestion !== 'keep' && activeTab === '编辑模式' && (
                              <div className="flex items-center gap-1.5 mt-0.5">
                                {item.reason && (
                                  <span className={`text-[9px] px-1.5 py-0.5 rounded border font-medium ${REASON_COLORS[item.reason] || 'bg-gray-50 text-gray-600 border-gray-200'}`}>
                                    {REASON_LABELS[item.reason] || item.reason}
                                  </span>
                                )}
                                {item.reasonDetail && (
                                  <span className="text-[9px] text-gray-400">{item.reasonDetail}</span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </React.Fragment>
                    );
                  })}
                </div>
              )}
            </div>
          </main>
        </div>
      )}

      <footer className="h-12 md:h-12 bg-black flex items-center px-3 md:px-4 shrink-0 z-20 border-t border-gray-800">
        <div className="w-10 md:w-12 font-mono text-[10px] text-gray-500">{formatTime(currentTime)}</div>
        <div ref={progressRef}
          className="flex-1 mx-2 md:mx-3 relative h-2 md:h-1.5 bg-gray-700 rounded cursor-pointer group"
          onMouseDown={handleProgressMouseDown}
          onTouchStart={handleProgressMouseDown}>
          <div className="absolute left-0 top-0 bottom-0 bg-white/80 rounded" style={{ width: `${progressPercent}%` }}></div>
          <div className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 md:w-3 md:h-3 bg-white rounded-full shadow md:opacity-0 md:group-hover:opacity-100 transition-opacity"
            style={{ left: `calc(${progressPercent}% - 7px)` }}></div>
          {hasData && transcripts.map(t => (
            <div key={t.id} className="absolute top-0 bottom-0"
              style={{ left: `${(t.startTime / audioDuration) * 100}%`, width: `${((t.endTime - t.startTime) / audioDuration) * 100}%`,
                backgroundColor: t.isKept ? (t.suggestion === 'mild' ? 'rgba(255,191,0,0.2)' : 'rgba(255,255,255,0.1)') : 'rgba(255,0,0,0.25)' }} />
          ))}
        </div>
        <div className="flex items-center gap-2 md:gap-3">
          <div className="w-10 md:w-auto font-mono text-[10px] text-gray-500">{formatTime(audioDuration)}</div>
          <button className="bg-white hover:bg-gray-100 text-black px-3 md:px-4 py-1 rounded-full transition-colors text-[11px] font-medium disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1"
            onClick={togglePlayPause} disabled={!audioUrl}>
            {isPlaying ? <><Pause className="w-3 h-3" /><span className="hidden sm:inline"> 暂停</span></> : <><Play className="w-3 h-3" /><span className="hidden sm:inline"> {activeTab === '粗剪预览' ? '预览' : '播放'}</span></>}
          </button>
          {activeTab === '粗剪预览' && <span className="text-[9px] text-gray-500 items-center gap-0.5 hidden md:flex"><SkipForward className="w-2.5 h-2.5" />跳过删减</span>}
        </div>
      </footer>

      {isUploading && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-8 max-w-xs w-full mx-4 shadow-2xl">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 mb-6 relative">
                <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-black rounded-full border-t-transparent animate-spin"></div>
              </div>
              <h3 className="text-base font-semibold text-gray-900 mb-1">AI 正在处理</h3>
              <p className="text-gray-500 text-[10px] mb-4">{uploadProgress || '准备中...'}</p>
              <div className="w-full h-0.5 bg-gray-200 rounded-full overflow-hidden mb-4">
                <div className="h-full bg-gray-900 rounded-full transition-all duration-1000" style={{ width: `${uploadStep * 25}%` }}></div>
              </div>
              <div className="w-full space-y-1.5">
                {[{ step: 1, label: '上传音频文件' }, { step: 2, label: '本地语音转文字' }, { step: 3, label: '千问文本分析' }, { step: 4, label: '生成剪辑建议' }].map(s => (
                  <div key={s.step} className="flex items-center gap-1.5">
                    <div className={`w-3.5 h-3.5 rounded-full flex items-center justify-center ${uploadStep >= s.step ? 'bg-gray-900' : 'bg-gray-200'}`}>
                      <span className={`text-[8px] ${uploadStep >= s.step ? 'text-white' : 'text-gray-400'}`}>{s.step}</span>
                    </div>
                    <span className={`text-[10px] ${uploadStep >= s.step ? 'text-gray-600' : 'text-gray-300'}`}>{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {showHistory && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50" onClick={() => setShowHistory(false)}>
          <div className="bg-white rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl max-h-[70vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">历史记录</h3>
              <button onClick={() => setShowHistory(false)} className="p-1 hover:bg-gray-100 rounded"><X className="w-4 h-4" /></button>
            </div>
            {historyList.length === 0 ? (
              <p className="text-gray-400 text-sm text-center py-8">暂无历史记录</p>
            ) : (
              <div className="flex-1 overflow-y-auto space-y-2">
                {historyList.map(h => (
                  <div key={h.task_id}
                    className="flex items-center gap-2 p-3 rounded-xl border border-gray-200 hover:border-black hover:bg-gray-50 transition-all group">
                    <button onClick={() => loadHistoryDetail(h.task_id)}
                      className="flex-1 text-left min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate flex-1">{h.filename || '未命名'}</span>
                        <span className="text-[10px] text-gray-400 ml-2 shrink-0">{h.created_at ? new Date(h.created_at).toLocaleDateString() : ''}</span>
                      </div>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-[10px] text-gray-500">{h.segment_count} 段</span>
                        <span className="text-[10px] text-gray-400">保留 {h.kept_count}</span>
                        {h.mild_count > 0 && <span className="text-[10px] text-amber-500">一般建议 {h.mild_count}</span>}
                        {h.strong_count > 0 && <span className="text-[10px] text-red-500">强烈建议 {h.strong_count}</span>}
                      </div>
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); deleteHistory(h.task_id); }}
                      className="p-1.5 rounded-lg text-gray-300 hover:text-red-500 hover:bg-red-50 transition-colors shrink-0 opacity-0 group-hover:opacity-100"
                      title="删除此记录">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {showExport && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50" onClick={() => setShowExport(false)}>
          <div className="bg-white rounded-2xl p-8 max-w-sm w-full mx-4 shadow-2xl" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">导出</h3>
            <div className="space-y-3">
              <button onClick={handleExportWord}
                className="w-full flex items-center gap-3 p-4 rounded-xl border border-gray-200 hover:border-black hover:bg-gray-50 transition-all text-left">
                <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center"><FileText className="w-5 h-5 text-blue-600" /></div>
                <div><p className="text-sm font-medium">Word 文稿</p><p className="text-[10px] text-gray-400">含删减标注的 .docx 文件</p></div>
              </button>
              <button onClick={handleExportMp3}
                className="w-full flex items-center gap-3 p-4 rounded-xl border border-gray-200 hover:border-black hover:bg-gray-50 transition-all text-left">
                <div className="w-10 h-10 bg-green-50 rounded-lg flex items-center justify-center"><Music className="w-5 h-5 text-green-600" /></div>
                <div><p className="text-sm font-medium">MP3 音频</p><p className="text-[10px] text-gray-400">删减后的 .mp3 剪辑版</p></div>
              </button>
            </div>
            <button onClick={() => setShowExport(false)} className="mt-4 w-full py-2 text-xs text-gray-500 hover:text-black transition-colors">关闭</button>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed bottom-16 left-1/2 -translate-x-1/2 bg-gray-900 text-white px-4 py-2 rounded-full shadow-2xl z-50 flex items-center gap-2">
          <span className="text-[11px]">{error}</span>
          <button onClick={() => { if (activeWsId) updateWs(activeWsId, { error: null }); }} className="ml-1 hover:bg-gray-700 w-4 h-4 flex items-center justify-center rounded-full text-xs">×</button>
        </div>
      )}
    </div>
  );
}
