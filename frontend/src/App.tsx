import {
  Download,
  Eye,
  EyeOff,
  Gauge,
  ImagePlus,
  Layers,
  Loader2,
  MousePointer2,
  Move,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from "lucide-react";
import { PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Candidate,
  ManualResult,
  Placement,
  RecommendResult,
  requestManualEvaluation,
  requestRecommendations,
} from "./api";

type Mode = "auto" | "manual";

const initialPlacement: Placement = { x: 0.36, y: 0.52, scale: 0.22 };

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string>("");
  useEffect(() => {
    if (!file) {
      setUrl("");
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

function scorePercent(score?: number) {
  return `${Math.round((score ?? 0) * 100)}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

async function sampleFile(path: string, name: string) {
  const response = await fetch(path);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

export default function App() {
  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [foregroundFile, setForegroundFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>("auto");
  const [topK, setTopK] = useState(3);
  const [placement, setPlacement] = useState<Placement>(initialPlacement);
  const [harmonize, setHarmonize] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [result, setResult] = useState<RecommendResult | ManualResult | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [backgroundSize, setBackgroundSize] = useState<{ width: number; height: number } | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const foregroundRef = useRef<HTMLImageElement | null>(null);
  const dragOffset = useRef({ x: 0, y: 0 });

  const backgroundUrl = useObjectUrl(backgroundFile);
  const foregroundUrl = useObjectUrl(foregroundFile);

  const bestCandidate = useMemo(() => {
    if (!result) return null;
    return result.mode === "auto" ? result.top[0] : result.candidate;
  }, [result]);

  const activeCandidate = selected ?? bestCandidate;
  const activeScore = activeCandidate?.score ?? 0;
  const activeLabel = activeCandidate?.label ?? "待评估";

  const ready = Boolean(backgroundFile && foregroundFile);

  const runAuto = useCallback(
    async (explain = true) => {
      if (!backgroundFile || !foregroundFile) return;
      setLoading(true);
      setError("");
      try {
        const next = await requestRecommendations({
          background: backgroundFile,
          foreground: foregroundFile,
          topK,
          scale: placement.scale,
          harmonize,
          explain,
        });
        setResult(next);
        setSelected(next.top[0] ?? null);
        if (next.top[0]) {
          setPlacement({ x: next.top[0].nx, y: next.top[0].ny, scale: next.top[0].nw });
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "推荐失败");
      } finally {
        setLoading(false);
      }
    },
    [backgroundFile, foregroundFile, harmonize, placement.scale, topK],
  );

  const runManual = useCallback(
    async (explain = false, customPlacement = placement) => {
      if (!backgroundFile || !foregroundFile) return;
      setLoading(true);
      setError("");
      try {
        const next = await requestManualEvaluation({
          background: backgroundFile,
          foreground: foregroundFile,
          placement: customPlacement,
          scale: customPlacement.scale,
          harmonize,
          explain,
        });
        setResult(next);
        setSelected(next.candidate);
        setPlacement({
          x: next.candidate.nx,
          y: next.candidate.ny,
          scale: next.candidate.nw,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "评估失败");
      } finally {
        setLoading(false);
      }
    },
    [backgroundFile, foregroundFile, harmonize, placement],
  );

  useEffect(() => {
    if (mode !== "manual" || !ready || dragging) return;
    const timer = window.setTimeout(() => {
      void runManual(false);
    }, 420);
    return () => window.clearTimeout(timer);
  }, [dragging, mode, placement.x, placement.y, placement.scale, ready, runManual]);

  async function loadSamples() {
    setError("");
    const [background, foreground] = await Promise.all([
      sampleFile("/samples/room_background.png", "room_background.png"),
      sampleFile("/samples/plant_foreground.png", "plant_foreground.png"),
    ]);
    setBackgroundFile(background);
    setForegroundFile(foreground);
    setResult(null);
    setSelected(null);
    setPlacement(initialPlacement);
  }

  function handleForegroundPointerDown(event: PointerEvent<HTMLImageElement>) {
    if (mode !== "manual" || !stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect();
    const fgRect = foregroundRef.current?.getBoundingClientRect();
    dragOffset.current = {
      x: event.clientX - (fgRect?.left ?? rect.left),
      y: event.clientY - (fgRect?.top ?? rect.top),
    };
    setDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleForegroundPointerMove(event: PointerEvent<HTMLImageElement>) {
    if (!dragging || !stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect();
    const foregroundRect = foregroundRef.current?.getBoundingClientRect();
    const foregroundWidth = foregroundRect?.width ?? rect.width * placement.scale;
    const foregroundHeight = foregroundRect?.height ?? foregroundWidth;
    const x = (event.clientX - rect.left - dragOffset.current.x) / rect.width;
    const y = (event.clientY - rect.top - dragOffset.current.y) / rect.height;
    setPlacement((current) => ({
      ...current,
      x: clamp(x, 0, Math.max(0.01, 1 - foregroundWidth / rect.width)),
      y: clamp(y, 0, Math.max(0.01, 1 - foregroundHeight / rect.height)),
    }));
  }

  function handleForegroundPointerUp(event: PointerEvent<HTMLImageElement>) {
    if (!dragging) return;
    setDragging(false);
    event.currentTarget.releasePointerCapture(event.pointerId);
    void runManual(false);
  }

  function chooseCandidate(candidate: Candidate) {
    const nextPlacement = { x: candidate.nx, y: candidate.ny, scale: candidate.nw };
    setSelected(candidate);
    setPlacement(nextPlacement);
    setMode("manual");
    void runManual(false, nextPlacement);
  }

  function exportComposite() {
    if (!result?.composite) return;
    const anchor = document.createElement("a");
    anchor.href = result.composite;
    anchor.download = "object-placement-composite.png";
    anchor.click();
  }

  const candidateList = result?.mode === "auto" ? result.top : result ? [result.candidate] : [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">方向 A</p>
          <h1>智能物体放置助手</h1>
        </div>
        <div className="status-pill">
          <Gauge size={18} />
          <span>{activeLabel}</span>
          <strong>{scorePercent(activeScore)}</strong>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel controls">
          <div className="panel-heading">
            <ImagePlus size={19} />
            <h2>输入</h2>
          </div>

          <label className="file-control">
            <span>背景图</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                setBackgroundFile(event.target.files?.[0] ?? null);
                setResult(null);
              }}
            />
            <Upload size={18} />
          </label>

          <label className="file-control">
            <span>前景图</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                setForegroundFile(event.target.files?.[0] ?? null);
                setResult(null);
              }}
            />
            <Upload size={18} />
          </label>

          <button className="secondary-button" type="button" onClick={loadSamples}>
            <Sparkles size={18} />
            加载示例
          </button>

          <div className="segmented">
            <button
              className={mode === "auto" ? "active" : ""}
              type="button"
              onClick={() => setMode("auto")}
            >
              <Layers size={17} />
              自动推荐
            </button>
            <button
              className={mode === "manual" ? "active" : ""}
              type="button"
              onClick={() => setMode("manual")}
            >
              <MousePointer2 size={17} />
              手动评估
            </button>
          </div>

          <div className="control-block">
            <div className="inline-label">
              <SlidersHorizontal size={17} />
              <span>物体大小</span>
              <strong>{Math.round(placement.scale * 100)}%</strong>
            </div>
            <input
              type="range"
              min="0.08"
              max="0.42"
              step="0.01"
              value={placement.scale}
              onChange={(event) => {
                const scale = Number(event.target.value);
                setPlacement((current) => ({ ...current, scale }));
                setSelected(null);
              }}
            />
          </div>

          <div className="control-row">
            <label>
              Top K
              <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                <option value={1}>Top-1</option>
                <option value={3}>Top-3</option>
                <option value={5}>Top-5</option>
              </select>
            </label>
            <label>
              协调
              <input
                type="checkbox"
                checked={harmonize}
                onChange={(event) => setHarmonize(event.target.checked)}
              />
            </label>
          </div>

          <button className="primary-button" type="button" disabled={!ready || loading} onClick={() => runAuto(true)}>
            {loading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
            生成推荐
          </button>

          <button
            className="secondary-button"
            type="button"
            disabled={!ready || loading}
            onClick={() => runManual(true)}
          >
            <Eye size={18} />
            生成解释
          </button>

          <button className="ghost-button" type="button" onClick={() => setPlacement(initialPlacement)}>
            <RotateCcw size={18} />
            重置位置
          </button>
        </aside>

        <section className="stage-panel">
          <div className="stage-toolbar">
            <div className="score-meter">
              <span style={{ width: `${scorePercent(activeScore)}%` }} />
            </div>
            <button className="icon-button" type="button" onClick={() => setShowHeatmap((value) => !value)}>
              {showHeatmap ? <EyeOff size={18} /> : <Eye size={18} />}
              热力图
            </button>
            <button className="icon-button" type="button" disabled={!result?.composite} onClick={exportComposite}>
              <Download size={18} />
              导出
            </button>
          </div>

          <div
            className={`stage ${backgroundUrl ? "has-image" : ""}`}
            ref={stageRef}
            style={backgroundSize ? { aspectRatio: `${backgroundSize.width} / ${backgroundSize.height}` } : undefined}
          >
            {backgroundUrl ? (
              <img
                className="background-image"
                src={backgroundUrl}
                alt="背景"
                onLoad={(event) =>
                  setBackgroundSize({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  })
                }
              />
            ) : (
              <div className="empty-stage">选择图片</div>
            )}
            {foregroundUrl && backgroundUrl ? (
              <img
                ref={foregroundRef}
                className={`foreground-object ${mode === "manual" ? "draggable" : ""}`}
                src={foregroundUrl}
                alt="前景"
                style={{
                  left: `${placement.x * 100}%`,
                  top: `${placement.y * 100}%`,
                  width: `${placement.scale * 100}%`,
                }}
                onPointerDown={handleForegroundPointerDown}
                onPointerMove={handleForegroundPointerMove}
                onPointerUp={handleForegroundPointerUp}
              />
            ) : null}
            {showHeatmap && result?.heatmap ? (
              <img className="heatmap-layer" src={result.heatmap} alt="解释热力图" />
            ) : null}
            {mode === "manual" && foregroundUrl ? (
              <div className="drag-hint">
                <Move size={16} />
                拖动
              </div>
            ) : null}
          </div>

          {error ? <div className="error-box">{error}</div> : null}
        </section>

        <aside className="panel results">
          <div className="panel-heading">
            <Layers size={19} />
            <h2>结果</h2>
          </div>

          <div className="score-card">
            <span>{activeLabel}</span>
            <strong>{scorePercent(activeScore)}</strong>
            <p>{activeCandidate?.reason ?? "等待模型输出"}</p>
          </div>

          <div className="meta-grid">
            <div>
              <span>mask</span>
              <strong>
                {result?.foreground.mask_source ?? "-"}{" "}
                {result?.foreground.mask_quality
                  ? Math.round(result.foreground.mask_quality * 100)
                  : ""}
              </strong>
            </div>
            <div>
              <span>位置</span>
              <strong>
                {activeCandidate ? `${activeCandidate.x}, ${activeCandidate.y}` : "-"}
              </strong>
            </div>
          </div>

          <div className="candidate-list">
            {candidateList.map((candidate, index) => (
              <button
                className={`candidate-card ${activeCandidate?.id === candidate.id ? "selected" : ""}`}
                key={candidate.id}
                type="button"
                onClick={() => chooseCandidate(candidate)}
              >
                <div className="candidate-rank">#{index + 1}</div>
                {candidate.preview ? <img src={candidate.preview} alt={`候选 ${index + 1}`} /> : null}
                <div>
                  <strong>{scorePercent(candidate.score)}</strong>
                  <span>{candidate.label}</span>
                  <p>{candidate.reason}</p>
                </div>
              </button>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}
