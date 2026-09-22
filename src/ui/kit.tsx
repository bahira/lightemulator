import { useEffect, useRef, useState, useCallback, type ReactNode } from 'react';
import { cn } from '../utils/cn';

// ---------------------------------------------------------------------------
//  Layout primitives
// ---------------------------------------------------------------------------

export function Panel({
  title, subtitle, tag, right, children, className, flush,
}: {
  title?: string; subtitle?: string; tag?: string; right?: ReactNode;
  children: ReactNode; className?: string; flush?: boolean;
}) {
  return (
    <section className={cn(
      'relative rounded-xl border border-white/8 bg-[#080b14]/90',
      'shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset,0_20px_50px_-30px_rgba(0,0,0,1)]',
      className,
    )}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-white/6 px-3.5 py-2.5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              {tag && (
                <span className="rounded bg-cyan-400/10 px-1.5 py-px font-mono text-[9px] font-bold tracking-widest text-cyan-300 ring-1 ring-cyan-400/20">
                  {tag}
                </span>
              )}
              <h3 className="truncate text-[12px] font-semibold tracking-tight text-slate-100">{title}</h3>
            </div>
            {subtitle && <p className="mt-0.5 text-[10px] leading-snug text-slate-500">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className={flush ? '' : 'p-3.5'}>{children}</div>
    </section>
  );
}

export function Stat({
  label, value, unit, hint, tone = 'cyan', mono = true,
}: {
  label: string; value: ReactNode; unit?: string; hint?: string;
  tone?: 'cyan' | 'violet' | 'amber' | 'green' | 'rose' | 'slate'; mono?: boolean;
}) {
  const tones: Record<string, string> = {
    cyan: 'text-cyan-300', violet: 'text-violet-300', amber: 'text-amber-300',
    green: 'text-emerald-300', rose: 'text-rose-300', slate: 'text-slate-300',
  };
  return (
    <div className="rounded-lg border border-white/6 bg-white/[0.015] px-2.5 py-2">
      <div className="text-[9px] uppercase tracking-[0.12em] text-slate-500">{label}</div>
      <div className={cn('mt-0.5 flex items-baseline gap-1', mono && 'font-mono')}>
        <span className={cn('text-[15px] font-semibold tabular-nums', tones[tone])}>{value}</span>
        {unit && <span className="text-[9px] text-slate-500">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 truncate text-[9px] text-slate-600">{hint}</div>}
    </div>
  );
}

export function Btn({
  children, onClick, active, tone = 'default', disabled, className, title,
}: {
  children: ReactNode; onClick?: () => void; active?: boolean; disabled?: boolean;
  tone?: 'default' | 'primary' | 'danger'; className?: string; title?: string;
}) {
  const base = 'rounded-md px-2.5 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider transition-all disabled:opacity-35 disabled:cursor-not-allowed';
  const tones = {
    default: active
      ? 'bg-white/10 text-slate-100 ring-1 ring-white/20'
      : 'bg-white/[0.03] text-slate-400 ring-1 ring-white/8 hover:bg-white/6 hover:text-slate-200',
    primary: 'bg-cyan-400/12 text-cyan-200 ring-1 ring-cyan-400/30 hover:bg-cyan-400/20',
    danger: 'bg-rose-500/12 text-rose-300 ring-1 ring-rose-500/30 hover:bg-rose-500/20',
  };
  return (
    <button title={title} disabled={disabled} onClick={onClick} className={cn(base, tones[tone], className)}>
      {children}
    </button>
  );
}

export function Slider({
  label, value, min, max, step, onChange, format, unit,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; format?: (v: number) => string; unit?: string;
}) {
  return (
    <label className="block select-none">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] text-slate-400">{label}</span>
        <span className="font-mono text-[10px] tabular-nums text-cyan-300">
          {format ? format(value) : value}{unit && <span className="text-slate-600">{unit}</span>}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded-full bg-white/10
          [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-300
          [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(103,232,249,0.7)]
          [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-cyan-300"
      />
    </label>
  );
}

export function Segmented<T extends string>({
  options, value, onChange, className,
}: {
  options: { id: T; label: string; title?: string }[];
  value: T; onChange: (v: T) => void; className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap gap-1 rounded-lg bg-white/[0.03] p-1 ring-1 ring-white/6', className)}>
      {options.map((o) => (
        <button
          key={o.id} title={o.title} onClick={() => onChange(o.id)}
          className={cn(
            'rounded px-2 py-1 font-mono text-[10px] tracking-wide transition-all',
            value === o.id
              ? 'bg-cyan-400/15 text-cyan-200 ring-1 ring-cyan-400/25'
              : 'text-slate-500 hover:text-slate-300',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Tag({ children, tone = 'slate' }: { children: ReactNode; tone?: 'slate' | 'green' | 'rose' | 'amber' | 'cyan' | 'violet' }) {
  const tones = {
    slate: 'bg-white/5 text-slate-400 ring-white/10',
    green: 'bg-emerald-400/10 text-emerald-300 ring-emerald-400/25',
    rose: 'bg-rose-500/10 text-rose-300 ring-rose-500/25',
    amber: 'bg-amber-400/10 text-amber-300 ring-amber-400/25',
    cyan: 'bg-cyan-400/10 text-cyan-300 ring-cyan-400/25',
    violet: 'bg-violet-400/10 text-violet-300 ring-violet-400/25',
  };
  return (
    <span className={cn('rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold tracking-wide ring-1', tones[tone])}>
      {children}
    </span>
  );
}

export function Formula({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-black/40 px-1.5 py-0.5 font-mono text-[10px] text-cyan-200/80 ring-1 ring-white/6">
      {children}
    </code>
  );
}

// ---------------------------------------------------------------------------
//  Export helpers — every lab can emit its results
// ---------------------------------------------------------------------------

export function copyText(s: string): Promise<void> {
  return navigator.clipboard.writeText(s);
}

export function downloadFile(name: string, content: string, mime = 'text/plain'): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Bouton copier avec accusé visuel. `text` peut être évalué paresseusement. */
export function CopyBtn({ text, label = '⧉ Copier' }: { text: string | (() => string); label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <Btn
      tone={done ? 'primary' : 'default'}
      onClick={() => {
        void copyText(typeof text === 'function' ? text() : text);
        setDone(true);
        setTimeout(() => setDone(false), 1200);
      }}
    >
      {done ? '✓ copié' : label}
    </Btn>
  );
}

// ---------------------------------------------------------------------------
//  URL share — paramètre persisté dans le hash, partageable par lien
// ---------------------------------------------------------------------------

export function useHashParam(key: string, initial: string): [string, (v: string) => void] {
  const [val, setVal] = useState(
    () => new URLSearchParams(window.location.hash.slice(1)).get(key) ?? initial,
  );
  useEffect(() => {
    const p = new URLSearchParams(window.location.hash.slice(1));
    if (val === initial) p.delete(key);
    else p.set(key, val);
    const h = p.toString();
    window.history.replaceState(null, '', h ? `#${h}` : window.location.pathname);
  }, [key, val, initial]);
  return [val, setVal];
}

// ---------------------------------------------------------------------------
//  Canvas hook — handles devicePixelRatio + ResizeObserver
// ---------------------------------------------------------------------------

export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[],
  aspect = 0.5,
) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const parent = el.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver(() => {
      const w = parent.clientWidth;
      setSize({ w, h: Math.round(w * aspect) });
    });
    ro.observe(parent);
    const w = parent.clientWidth;
    setSize({ w, h: Math.round(w * aspect) });
    return () => ro.disconnect();
  }, [aspect]);

  const render = useCallback(() => {
    const el = ref.current;
    if (!el || size.w === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (el.width !== Math.round(size.w * dpr) || el.height !== Math.round(size.h * dpr)) {
      el.width = Math.round(size.w * dpr);
      el.height = Math.round(size.h * dpr);
    }
    const ctx = el.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    drawRef.current(ctx, size.w, size.h);
  }, [size]);

  useEffect(() => {
    const id = requestAnimationFrame(render);
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [render, ...deps]);

  return { ref, size };
}

// ---------------------------------------------------------------------------
//  Scientific colormaps (LUT based)
// ---------------------------------------------------------------------------

type Stop = [number, [number, number, number]];

function buildLUT(stops: Stop[], size = 256): Uint8Array {
  const lut = new Uint8Array(size * 3);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let a = stops[0], b = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; }
    }
    const span = b[0] - a[0] || 1;
    const u = (t - a[0]) / span;
    lut[i * 3] = a[1][0] + (b[1][0] - a[1][0]) * u;
    lut[i * 3 + 1] = a[1][1] + (b[1][1] - a[1][1]) * u;
    lut[i * 3 + 2] = a[1][2] + (b[1][2] - a[1][2]) * u;
  }
  return lut;
}

export const LUT = {
  inferno: buildLUT([
    [0.0, [0, 0, 4]], [0.15, [22, 11, 57]], [0.3, [66, 10, 104]], [0.45, [114, 31, 129]],
    [0.6, [168, 45, 96]], [0.75, [221, 81, 58]], [0.88, [250, 152, 23]], [1.0, [252, 255, 164]],
  ]),
  viridis: buildLUT([
    [0.0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]], [1.0, [253, 231, 37]],
  ]),
  ice: buildLUT([
    [0.0, [3, 6, 18]], [0.25, [8, 47, 73]], [0.5, [14, 116, 144]],
    [0.72, [34, 211, 238]], [0.88, [165, 243, 252]], [1.0, [240, 253, 255]],
  ]),
  twilight: buildLUT([
    [0.0, [226, 217, 226]], [0.13, [148, 152, 202]], [0.25, [86, 108, 191]],
    [0.38, [52, 74, 138]], [0.5, [26, 30, 60]], [0.62, [86, 42, 78]],
    [0.75, [163, 66, 92]], [0.87, [214, 122, 116]], [1.0, [226, 217, 226]],
  ]),
  diverging: buildLUT([
    [0.0, [49, 54, 149]], [0.25, [69, 152, 202]], [0.5, [12, 16, 28]],
    [0.75, [244, 145, 76]], [1.0, [165, 15, 21]],
  ]),
};

export type LutName = keyof typeof LUT;

/**
 * Paints a scalar field into an ImageData buffer. `stride` maps a source row
 * to the destination; caller decides orientation.
 */
export function paintField(
  img: ImageData, data: Float32Array | Float64Array,
  cols: number, rows: number, lut: Uint8Array,
  min: number, max: number, gamma = 1, transpose = false,
): void {
  const d = img.data;
  const span = max - min || 1;
  const W = transpose ? rows : cols;
  const H = transpose ? cols : rows;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const src = transpose ? x * cols + y : y * cols + x;
      let t = (data[src] - min) / span;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      if (gamma !== 1) t = Math.pow(t, gamma);
      const li = (t * 255) | 0;
      const o = (y * W + x) * 4;
      d[o] = lut[li * 3];
      d[o + 1] = lut[li * 3 + 1];
      d[o + 2] = lut[li * 3 + 2];
      d[o + 3] = 255;
    }
  }
}

export function drawImageBuffer(
  ctx: CanvasRenderingContext2D, img: ImageData,
  dx: number, dy: number, dw: number, dh: number, smooth = true,
): void {
  const off = document.createElement('canvas');
  off.width = img.width; off.height = img.height;
  const octx = off.getContext('2d');
  if (!octx) return;
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(off, dx, dy, dw, dh);
}

// ---------------------------------------------------------------------------
//  Small chart helpers
// ---------------------------------------------------------------------------

export function drawGrid(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, nx = 6, ny = 4) {
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= nx; i++) {
    const px = Math.round(x + (w * i) / nx) + 0.5;
    ctx.moveTo(px, y); ctx.lineTo(px, y + h);
  }
  for (let i = 0; i <= ny; i++) {
    const py = Math.round(y + (h * i) / ny) + 0.5;
    ctx.moveTo(x, py); ctx.lineTo(x + w, py);
  }
  ctx.stroke();
  ctx.restore();
}

export function drawSeries(
  ctx: CanvasRenderingContext2D,
  data: ArrayLike<number>,
  x: number, y: number, w: number, h: number,
  min: number, max: number,
  color: string, fill?: string, lineWidth = 1.5,
) {
  const n = data.length;
  if (n < 2) return;
  const span = max - min || 1;
  const px = (i: number) => x + (i / (n - 1)) * w;
  const py = (v: number) => y + h - ((v - min) / span) * h;

  if (fill) {
    ctx.beginPath();
    ctx.moveTo(px(0), y + h);
    for (let i = 0; i < n; i++) ctx.lineTo(px(i), py(data[i]));
    ctx.lineTo(px(n - 1), y + h);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const X = px(i), Y = py(data[i]);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

export function axisLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, align: CanvasTextAlign = 'left', color = 'rgba(148,163,184,0.75)') {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = '9px ui-monospace, monospace';
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
  ctx.restore();
}
