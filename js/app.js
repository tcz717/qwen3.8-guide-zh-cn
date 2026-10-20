/* ==========================================================================
   Qwen 3.8-27B — From Scratch · interactive engine
   All widgets computed from the official config (Qwen/Qwen3.8-27B):
   d=5120, L=64 (48 DeltaNet + 16 GQA), dff=17408, vocab=248320,
   DeltaNet 16QK/48V × dh128 · GQA 24Q/4KV × dh256 (rotary 64), ctx 262144
   ========================================================================== */
"use strict";

/* Reload behavior:
   Without a hash or at top of page, ensure the page always starts cleanly at the top. */
if (!location.hash || location.hash === "#top") {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }
  window.scrollTo(0, 0);
  window.addEventListener("DOMContentLoaded", () => {
    if (!location.hash || location.hash === "#top") window.scrollTo(0, 0);
  });
}

/* ---------- figure inventory (running-head counter source-of-truth) ---------- */
const FIGURES = [
  "数据流",
  "MRoPE 表盘",
  "FFN 激活曲线",
  "残差总线方差",
  "MTP 前向传播与验证树",
  "预训练损失",
  "β₂ 恢复",
  "测试时计算扩展",
  "硬件屋顶线",
];

/* ---------- shared config + formatters ---------- */
const CFG = {
  d: 5120, layers: 64, nDelta: 48, nFull: 16, dff: 17408, vocab: 248320,
  dnQK: 16, dnV: 48, dnDh: 128,
  gqaQ: 24, gqaKV: 4, gqaDh: 256, rotDims: 64,
  visionTensors: 333, ctxNative: 262144,
  ropeTheta: 1e7
};
/* per-mixer parameter counts — single source for the W1 audit, layer explorer and totals */
CFG.ffnParams = 3 * CFG.d * CFG.dff;
CFG.dnParams = (CFG.dnQK * 2 + CFG.dnV) * CFG.dnDh * CFG.d   // fused in_proj (q, k, v)
  + CFG.dnV                                                  // β-gate: bias vector only (b_proj.bias)
  + CFG.dnV * CFG.dnDh * CFG.d;                              // out_proj
CFG.faParams = (CFG.gqaQ * 2 + CFG.gqaKV * 2 + CFG.gqaQ) * CFG.gqaDh * CFG.d;
/* grand total (27.58B): core + untied LM head + MTP (K·d² + shared d·V head) + vision tower */
CFG.mtpParams = 2 * (CFG.d * CFG.d) + CFG.d * CFG.vocab;
CFG.total = CFG.nDelta * (CFG.dnParams + CFG.ffnParams) + CFG.nFull * (CFG.faParams + CFG.ffnParams)
  + 2 * CFG.d * CFG.layers + CFG.d                           // RMSNorm gains: 2/layer + final
  + CFG.d * CFG.vocab                                        // token embeddings
  + CFG.d * CFG.vocab                                        // untied LM head
  + CFG.mtpParams + 0.9e9;                                   // MTP module + SigLIP-style vision tower (est.)
const fmtP = p => p >= 1e9 ? (p / 1e9).toFixed(2) + "B" : p >= 1e6 ? (p / 1e6).toFixed(1) + "M" : Math.round(p).toLocaleString();
const $ = id => document.getElementById(id);

/* pigment palette — mirror of the :root tokens in css/styles.css (keep in sync);
   SVG presentation attributes can't resolve var(), so JS needs the literals */
const PAL = {
  amber: "#b8341f", mint: "#1d6e6b", iris: "#2b3a8a", sky: "#8a5610",
  rose: "#8a3324", paper: "#f3ece0", ink: "#1a1612", muted: "#5a5046"
};

/* §08 acceptance convention (single source): E[tokens/pass] = (1−β^(K+1))/(1−β);
   net speedup S ≈ E/(1+γ_verify) with γ_verify = 0.15 — reproduces §08's
   published K=2, β=0.85 anchor: E = 2.57 tok/pass → S ≈ 2.24× (>2.2×) */
const MTP_GAMMA = 0.15;
const mtpEtoks = (beta, K) => (1 - Math.pow(beta, K + 1)) / (1 - beta);
const mtpSpeedup = (beta, K) => mtpEtoks(beta, K) / (1 + MTP_GAMMA);

/* escape text interpolated into widget innerHTML */
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* KaTeX auto-render for dynamically injected HTML (no-op if the CDN didn't load) */
function katexRender(el) {
  if (typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(el, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$",  right: "$",  display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    ignoredClasses: ["nokatex"],
  });
}

/* segmented-button group: toggles .on + aria-pressed on matched children, then calls onChange(button) */
function bindGroup(container, onChange, sel = "button") {
  if (!container) return;
  const buttons = container.querySelectorAll(sel);
  buttons.forEach(b => b.addEventListener("click", () => {
    buttons.forEach(x => { x.classList.remove("on"); x.setAttribute("aria-pressed", "false"); });
    b.classList.add("on"); b.setAttribute("aria-pressed", "true");
    onChange(b);
  }));
}

/* div-based tab sets: focusable, button semantics, Enter/Space activate */
function enableTabs(container, itemSel) {
  if (!container) return;
  container.querySelectorAll(itemSel).forEach(it => {
    it.tabIndex = 0;
    it.setAttribute("role", "button");
    it.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); it.click(); }
    });
  });
}

/* slugify headings → ids used by .anchor links */
const slug = s => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "").slice(0, 64);

/* stacked-bar segments: absolutely positioned at a running left offset and
   scaled on the compositor, so the transition never triggers layout */
function stackSegments(pairs) {
  let left = 0;
  for (const [el, pct] of pairs) {
    if (el) {
      el.style.left = left + "%";
      el.style.transform = `scaleX(${pct / 100})`;
    }
    left += pct;
  }
}

/* ---------- scroll progress + nav scroll state ---------- */
(function () {
  const bar = $("progress");
  const nav = document.querySelector(".topnav");
  const brand = document.querySelector(".nav-brand");
  const onScroll = () => {
    const h = document.documentElement;
    const pct = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100;
    bar.style.width = pct + "%";
    if (nav) nav.classList.toggle("scrolled", h.scrollTop > 12);
    if (h.scrollTop < 100 && location.hash && location.hash !== "#top") {
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  };
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  if (brand) {
    brand.style.cursor = "pointer";
    brand.addEventListener("click", () => {
      if (location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
      const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
  }
})();

/* ---------- hero: 64-bit architecture register ---------- */
(function () {
  const box = $("heroBits"); if (!box) return;
  const label = box.parentElement.querySelector(".bits-label");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cells = [];
  for (let i = 0; i < 64; i++) { const b = document.createElement("span"); b.className = "bit"; box.appendChild(b); cells.push(b); }
  const kind = i => (i % 4 === 3) ? "fm" : "fa";           // 3 DeltaNet + 1 Gated Attention, ×16
  if (reduced) { cells.forEach((b, i) => b.classList.add(kind(i))); label?.classList.add("show"); return; }
  // phase 1 — uninitialized memory flicker
  let ticks = 0;
  const flicker = setInterval(() => {
    cells.forEach(b => { b.classList.remove("fa", "fm"); if (Math.random() < .3) b.classList.add(Math.random() < .75 ? "fa" : "fm"); });
    if (++ticks >= 10) { clearInterval(flicker); settle(0); }
  }, 60);
  // phase 2 — settle into the 3:1 motif, one super-block at a time
  function settle(g) {
    if (g >= 16) { label?.classList.add("show"); return; }
    for (let k = 0; k < 4; k++) {
      const i = g * 4 + k;
      setTimeout(() => { cells[i].classList.remove("fa", "fm"); cells[i].classList.add(kind(i)); }, k * 45);
    }
    setTimeout(() => settle(g + 1), 140);
  }
})();

/* ---------- active nav link + running-head label ---------- */
(function () {
  const links = [...document.querySelectorAll("#navLinks a")];
  const cur = document.getElementById("navCurrent");
  const secs = links.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const io = new IntersectionObserver(es => {
    // pick the entry that's most visible among intersecting ones
    const visible = es.filter(e => e.isIntersecting);
    if (visible.length) {
      // closest to rootMargin center
      const best = visible.reduce((a, b) => b.intersectionRatio > a.intersectionRatio ? b : a);
      const active = links.find(l => l.getAttribute("href") === "#" + best.target.id);
      links.forEach(l => l.classList.toggle("on", l === active));
      if (cur) {
        if (active?.dataset.num) {
          cur.innerHTML = `<b>§${active.dataset.num}</b> · ${active.dataset.label}`;
          cur.classList.add("show");
        } else {
          cur.classList.remove("show");
        }
      }
    } else if (cur) {
      cur.classList.remove("show");
    }
  }, { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] });
  secs.forEach(s => io.observe(s));
})();

/* ---------- contents panel toggle ---------- */
(function () {
  const btn = $("navToggle"), panel = $("navPanel");
  if (!btn || !panel) return;
  const setOpen = o => { panel.classList.toggle("open", o); btn.setAttribute("aria-expanded", String(o)); };
  btn.addEventListener("click", () => setOpen(!panel.classList.contains("open")));
  panel.addEventListener("click", e => { if (e.target.closest("a")) setOpen(false); });
  addEventListener("keydown", e => { if (e.key === "Escape") setOpen(false); });
  addEventListener("click", e => { if (!e.target.closest(".topnav")) setOpen(false); });
})();

/* ---------- back-to-top button ---------- */
(function () {
  const btn = $("totop");
  if (btn) {
    const onScroll = () => {
      const visible = scrollY > 600;
      btn.classList.toggle("on", visible);
      btn.tabIndex = visible ? 0 : -1;
      btn.setAttribute("aria-hidden", String(!visible));
    };
    addEventListener("scroll", onScroll, { passive: true });
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    btn.addEventListener("click", () => {
      if (location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
    onScroll();
  }
})();

/* ---------- heading anchor links (hover-only ¶) ---------- */
(function () {
  // first pass: ensure every h2/h3 inside a section has an id (so we can link to it)
  document.querySelectorAll("section h2, section h3").forEach(h => {
    if (!h.id) h.id = slug(h.textContent.replace(/¶/g, "")) || ("h-" + Math.random().toString(36).slice(2, 7));
  });
  // second pass: append the ¶ anchor link to each
  document.querySelectorAll("section h2[id], section h3[id]").forEach(h => {
    const a = document.createElement("a");
    a.className = "anchor"; a.href = "#" + h.id;
    a.setAttribute("aria-label", "链接到" + h.textContent.replace(/¶/g, "").trim());
    a.textContent = "¶";
    h.appendChild(a);
  });
})();

/* ---------- copy buttons on <pre class="code"> blocks ---------- */
(function () {
  document.querySelectorAll("pre.code").forEach(pre => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "code-copy"; b.textContent = "复制";
    b.setAttribute("aria-label", "复制代码到剪贴板");
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        b.textContent = "已复制 ✓"; b.classList.add("ok");
        setTimeout(() => { b.textContent = "复制"; b.classList.remove("ok"); }, 1400);
      } catch { b.textContent = "复制失败"; setTimeout(() => b.textContent = "复制", 1600); }
    });
    pre.appendChild(b);
  });
})();

/* ---------- "next section" pills at the end of every section except the last ---------- */
(function () {
  const sects = [...document.querySelectorAll("main > section")];
  sects.forEach((s, i) => {
    if (i === sects.length - 1) return;
    const next = sects[i + 1];
    const a = document.createElement("a");
    a.className = "next-pill"; a.href = "#" + next.id;
    const t = next.querySelector("h2");
    const cleanText = (t?.textContent || next.id).replace(/¶/g, "").trim();
    a.innerHTML = `${cleanText} <span class="arrow" aria-hidden="true">→</span>`;
    s.appendChild(a);
  });
})();

/* ---------- callout glyphs ---------- */
(function () {
  const ICONS = {
    why:       '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="6.5"/><path d="M6.5 6.2c0-1 .7-1.7 1.7-1.7s1.7.7 1.7 1.6c0 .9-.5 1.3-1.1 1.6-.6.3-1 .7-1 1.4M8 11.4h.01"/></svg>',
    deep:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8h12M2 8l3-3M2 8l3 3M14 8l-3-3M14 8l-3 3"/></svg>',
    intuition: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5C4.4 1.5 1.5 4.4 1.5 8c0 2 .9 3.7 2.3 4.9L3 14.5h2.4l-.2-1.1A6.5 6.5 0 1 1 14.5 8c0 2.5-1.4 4.6-3.5 5.7"/><path d="M5 9.5c.5 1.2 1.7 2 3 2s2.5-.8 3-2"/></svg>',
    warn:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5L1.5 13.5h13L8 1.5z"/><path d="M8 6v3.5M8 11.4h.01"/></svg>',
    note:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="10" rx="2"/><path d="M5 6.5h6M5 9h6M5 11.2h3.5"/></svg>'
  };
  document.querySelectorAll(".callout").forEach(c => {
    const variant = [...c.classList].find(k => ICONS[k]);
    if (!variant) return;
    const title = c.querySelector(".co-title");
    if (!title) return;
    const ic = document.createElement("span");
    ic.className = "ic"; ic.setAttribute("aria-hidden", "true");
    ic.innerHTML = ICONS[variant];
    title.insertBefore(ic, title.firstChild);
  });
})();

/* ============================================================
   W1 — Parameter accounting audit (#w-param)
   ============================================================ */
(function () {
  const out = $("paramOut"); if (!out) return;
  let withVision = true, withMtp = true;

  function compute() {
    const D = CFG.d, V = CFG.vocab, L = CFG.layers;
    const emb = D * V;
    const head = D * V;                                        // untied LM head (§14: embed + lm_head are separate)
    const norms = 2 * D * L + D;                               // 2 per layer + final
    // core weights only (embeddings included; head/MTP/vision excluded) — same expressions as CFG.total
    const coreExact = CFG.nDelta * (CFG.dnParams + CFG.ffnParams) + CFG.nFull * (CFG.faParams + CFG.ffnParams)
      + norms + emb;
    const mtp = withMtp ? CFG.mtpParams : 0;                   // K draft blocks (d² each) + shared d·V vocab head
    const vis = withVision ? 0.9e9 : 0;                        // SigLIP-style tower, est.
    return {
      emb, head, norms,
      attn: CFG.nFull * CFG.faParams,
      delta: CFG.nDelta * CFG.dnParams,
      ffn: L * CFG.ffnParams,
      mtp, vis,
      total: coreExact + head + mtp + vis
    };
  }

  function render() {
    const c = compute();
    const rows = [
      ["词元嵌入（248,320 × 5120）", c.emb, ""],
      ["门控 DeltaNet 层 — 48 × 输入/输出/门控投影", c.delta, "a"],
      ["门控注意力层 — 16 × q/k/v/门控/o 投影", c.attn, "m"],
      ["SwiGLU FFN — 64 × 3·d·dff", c.ffn, "i"],
      ["RMSNorm 增益", c.norms, ""],
      ["LM 头（非共享权重）— 5120 × 248,320", c.head, ""],
    ];
    if (withMtp) rows.push(["MTP 模块（草稿层 + 头）", c.mtp, ""]);
    if (withVision) rows.push(["视觉塔（SigLIP 风格，估算）", c.vis, ""]);
    out.innerHTML =
      rows.map(([l, v, cls]) => `<div class="row"><span>${l}</span><span class="v ${cls}">${fmtP(v)}</span></div>`).join("") +
      `<div class="row" style="border-top:1px solid var(--line-strong);margin-top:6px;padding-top:10px">
         <span><b>总计（LM ${withMtp ? "+ MTP " : ""}${withVision ? "+ 视觉" : ""}）</b></span>
         <span class="big">${fmtP(c.total)}</span></div>` +
      `<div style="color:var(--faint);font-size:12px;margin-top:8px">BF16 检查点 ≈ ${(c.total * 2 / 1e9).toFixed(1)} GB</div>`;
    // mirror the headline number to the hero stat card so they never disagree
    const heroEl = document.getElementById("statParams");
    if (heroEl) heroEl.textContent = fmtP(c.total);
  }
  [["paramVisionSeg", v => withVision = v === "1"], ["paramMtpSeg", v => withMtp = v === "1"]].forEach(([id, fn]) => {
    bindGroup($(id), b => { fn(b.dataset.v); render(); });
  });
  render();
})();

/* ============================================================
   Section 02: Tokenizer & Input Contract Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Interactive BPE Token Segmenter (#w-tokenizer-lab)
     ------------------------------------------------------------ */
  const tokLiveTokens = $("tokLiveTokens");
  const tokPresetSel = $("tokPresetSel");

  const PRESETS = {
    "code": {
      name: "Python 代码：FlashAttention 内核",
      raw: "def matmul_kernel(Q, K, V):\n    return flash_attn_func(Q, K, V, causal=True)",
      tokens: [
        { text: "def", id: 982 },
        { text: " mat", id: 2410 },
        { text: "mul", id: 18452 },
        { text: "_kernel", id: 45910 },
        { text: "(", id: 7 },
        { text: "Q", id: 52 },
        { text: ",", id: 11 },
        { text: " K", id: 46 },
        { text: ",", id: 11 },
        { text: " V", id: 57 },
        { text: "):\n   ", id: 3829 },
        { text: " return", id: 622 },
        { text: " flash", id: 14205 },
        { text: "_attn", id: 31084 },
        { text: "_func", id: 19502 },
        { text: "(Q", id: 8912 },
        { text: ", K", id: 2314 },
        { text: ", V", id: 2841 },
        { text: ", causal", id: 51204 },
        { text: "=True", id: 4120 },
        { text: ")", id: 8 }
      ]
    },
    "math": {
      name: "推理通道提示词",
      raw: "<think> Let S_t = α_t · (S_{t-1}(I - β_t k_t k_t^T) + β_t k_t v_t^T). </think>",
      tokens: [
        { text: "<think>", id: 248002 },
        { text: " Let", id: 2849 },
        { text: " S", id: 54 },
        { text: "_t", id: 821 },
        { text: " =", id: 284 },
        { text: " α", id: 1420 },
        { text: "_t", id: 821 },
        { text: " ·", id: 1944 },
        { text: " (", id: 320 },
        { text: "S", id: 54 },
        { text: "_{t", id: 18420 },
        { text: "-1}", id: 4102 },
        { text: "(I", id: 9102 },
        { text: " -", id: 482 },
        { text: " β", id: 1682 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t^T", id: 64201 },
        { text: ")", id: 8 },
        { text: " +", id: 491 },
        { text: " β", id: 1682 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t", id: 821 },
        { text: " v", id: 86 },
        { text: "_t^T", id: 64201 },
        { text: ").", id: 914 },
        { text: " </think>", id: 248003 }
      ]
    },
    "multilingual": {
      name: "多语言对齐",
      raw: "Qwen 3.8 aligns English, 深度学习架构, गणितीय अनुकूलन, and الذكاء الاصطناعي.",
      tokens: [
        { text: "Qwen", id: 48201 },
        { text: " 3.8", id: 12902 },
        { text: " aligns", id: 42109 },
        { text: " English", id: 4120 },
        { text: ",", id: 11 },
        { text: " 深度", id: 29402 },
        { text: "学习", id: 10842 },
        { text: "架构", id: 31084 },
        { text: ",", id: 11 },
        { text: " गणित", id: 48102 },
        { text: "ीय", id: 14201 },
        { text: " अनुकूल", id: 68102 },
        { text: "न", id: 8904 },
        { text: ",", id: 11 },
        { text: " and", id: 324 },
        { text: " الذك", id: 34102 },
        { text: "اء", id: 11204 },
        { text: " الاصط", id: 54109 },
        { text: "ناعي", id: 21904 },
        { text: ".", id: 13 }
      ]
    },
    "agent": {
      name: "智能体工具 RPC 语法",
      raw: "<|im_start|>assistant\n<|tool_call_start|>{ \"name\": \"eval_expr\", \"args\": { \"x\": 42 } }<|tool_call_end|><|im_end|>",
      tokens: [
        { text: "<|im_start|>", id: 248004 },
        { text: "assistant\n", id: 78201 },
        { text: "<|tool_call_start|>", id: 248006 },
        { text: "{", id: 28 },
        { text: " \"name\"", id: 1042 },
        { text: ":", id: 25 },
        { text: " \"eval_expr\"", id: 49102 },
        { text: ",", id: 11 },
        { text: " \"args\"", id: 2940 },
        { text: ":", id: 25 },
        { text: " {", id: 410 },
        { text: " \"x\"", id: 1204 },
        { text: ":", id: 25 },
        { text: " 42", id: 2914 },
        { text: " }", id: 482 },
        { text: " }", id: 482 },
        { text: "<|tool_call_end|>", id: 248007 },
        { text: "<|im_end|>", id: 248005 }
      ]
    },
    "lowresource": {
      name: "低资源文字（泰语 / 高棉语 / 泰米尔语）",
      raw: "โมเดลไฮบริด (Thai) · மொழியியல் (Tamil) · ភាសាខ្មែរ (Khmer)",
      tokens: [
        { text: "โม", id: 12401 },
        { text: "เดล", id: 38402 },
        { text: "ไฮ", id: 41029 },
        { text: "บริด", id: 59102 },
        { text: " (Thai)", id: 9410 },
        { text: " ·", id: 1944 },
        { text: " மொழி", id: 48102 },
        { text: "யியல்", id: 62109 },
        { text: " (Tamil)", id: 11029 },
        { text: " ·", id: 1944 },
        { text: " ភាសា", id: 58102 },
        { text: "ខ្មែរ", id: 74109 },
        { text: " (Khmer)", id: 14029 }
      ]
    }
  };

  /* low-alpha washes of the palette pigments — amber/mint/iris/sky/rose */
  const BADGE_COLORS = [
    "rgba(184,52,31,0.18)",
    "rgba(29,110,107,0.18)",
    "rgba(43,58,138,0.18)",
    "rgba(138,86,16,0.18)",
    "rgba(138,51,36,0.18)"
  ];

  function renderTokenizerLab() {
    if (!tokLiveTokens) return;
    const key = tokPresetSel ? tokPresetSel.value : "code";
    const data = PRESETS[key] || PRESETS["code"];

    const numTokens = data.tokens.length;
    const numChars = data.raw.length;
    const fertility = (numChars / numTokens).toFixed(2);
    const vramBytes = numTokens * 5120 * 2; // BF16 embeddings

    const badgesHtml = data.tokens.map((t, idx) => {
      const color = BADGE_COLORS[idx % BADGE_COLORS.length];
      const isControl = t.text.startsWith("<|") || t.text.startsWith("<think") || t.text.startsWith("</think");
      // escape before injecting — token text can contain "<", "&" etc.
      const shown = esc(t.text).replace(/\n/g, "↵").replace(/ /g, "␣");
      return `<span style="display:inline-block;background:${isControl ? "rgba(43,58,138,0.25)" : color};border:1px solid ${isControl ? "var(--iris)" : "var(--line)"};padding:2px 6px;margin:2px;border-radius:4px;font-family:var(--mono);font-size:12px;color:${isControl ? "var(--iris)" : "var(--fg)"}" title="词元 ID：${t.id}"><b>${shown}</b> <span style="color:var(--faint);font-size:12px">#${t.id}</span></span>`;
    }).join("");

    const idsPreview = data.tokens.map(t => t.id).join(", ");

    tokLiveTokens.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${numTokens} 个词元</div>
          <div class="cs-lbl">输出词元</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${numChars} 个字符</div>
          <div class="cs-lbl">原始字符长度</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${fertility} 字符/词元</div>
          <div class="cs-lbl">压缩生育率</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--sky)">${(vramBytes / 1024).toFixed(1)} KB</div>
          <div class="cs-lbl">嵌入激活占用</div>
        </div>
      </div>

      <div style="background:var(--bg-2);padding:10px 14px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);margin-top:8px">
        <div style="font-family:var(--mono);font-size:12px;color:var(--faint);margin-bottom:6px">分段词元流（点击词元查看 ID）：</div>
        <div style="line-height:1.8">${badgesHtml}</div>
      </div>

      <div style="font-family:var(--mono);font-size:12px;color:var(--faint);background:var(--bg-3,var(--bg-2));padding:6px 10px;border-radius:4px;margin-top:8px;overflow-x:auto;white-space:nowrap">
        词元 ID：[ ${idsPreview} ]
      </div>
    `;
  }

  if (tokPresetSel) {
    tokPresetSel.addEventListener("change", renderTokenizerLab);
  }
  renderTokenizerLab();


  /* ------------------------------------------------------------
     2. Widget: Multilingual Fertility & Scaling Matrix (#w-fertility-matrix)
     ------------------------------------------------------------ */
  const fertStatCard = $("fertStatCard");
  const fertLangSeg = $("fertLangSeg");

  const FERTILITY_DATA = {
    "en": {
      name: "英语散文与技术文档",
      v32: "1.30 词元/单词",
      v151: "1.15 词元/单词",
      v248: "1.08 词元/单词",
      gain: "相比 32k，序列压缩 16.9%",
      desc: "英语压缩已很成熟；248k 词表可将复杂技术复合名词和形态后缀捕获为原子词元。"
    },
    "code": {
      name: "Python / C++ / TypeScript 源代码",
      v32: "1.50× 字符扩展",
      v151: "1.20× 字符扩展",
      v248: "1.05× 字符扩展",
      gain: "相比 32k，序列缩短 30.0%",
      desc: "连续 4/8 个空格的缩进、snake_case 标识符和运算符链可合并为单个词元，大幅加速代码推理。"
    },
    "zh": {
      name: "中文（简体与繁体）",
      v32: "1.45 词元/字符",
      v151: "0.72 词元/字符",
      v248: "0.68 词元/字符",
      gain: "相比 32k，序列压缩 53.1%",
      desc: "扩展的多字中文词典可将四字成语和常用名词映射为单个词元嵌入。"
    },
    "hi": {
      name: "印地语与印度天城文字",
      v32: "2.80 词元/单词（字节拆分）",
      v151: "1.55 词元/单词",
      v248: "1.22 词元/单词",
      gain: "相比 32k，序列压缩 56.4%",
      desc: "彻底解决经典的“字节退化”损失：过去印度文字连写需拆成 3 至 4 个独立 UTF-8 字节。"
    },
    "ar": {
      name: "阿拉伯语与波斯—阿拉伯文字",
      v32: "2.40 词元/单词",
      v151: "1.40 词元/单词",
      v248: "1.15 词元/单词",
      gain: "相比 32k，序列压缩 52.1%",
      desc: "原生词根—词型形态分词消除了伊斯兰法学和科学文献中过度的子词碎片化。"
    },
    "th": {
      name: "泰语与高棉语连续文字",
      v32: "3.90 词元/单词（字节退化）",
      v151: "1.85 词元/单词",
      v248: "1.40 词元/单词",
      gain: "相比 32k，序列压缩 64.1%",
      desc: "没有显式词界的文字获得最大相对吞吐提升，将未分段的字节链转换为连贯的词汇词元。"
    }
  };

  function renderFertility(langKey) {
    if (!fertStatCard) return;
    const d = FERTILITY_DATA[langKey] || FERTILITY_DATA["en"];

    fertStatCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${d.name}</span>
        <span style="font-family:var(--mono);font-size:12px;color:var(--mint);font-weight:600">${d.gain}</span>
      </div>

      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--rose)">${d.v32}</div>
          <div class="cs-lbl">旧版 32k 词表</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${d.v151}</div>
          <div class="cs-lbl">Qwen 2.5（151k 词表）</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${d.v248}</div>
          <div class="cs-lbl">Qwen 3.8（填充至 248k）</div>
        </div>
      </div>

      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-top:8px">
        ${d.desc}
      </div>
    `;
  }

  if (fertLangSeg) {
    bindGroup(fertLangSeg, b => renderFertility(b.dataset.lang));
  }
  renderFertility("en");
})();

/* ============================================================
   Section 03: The Hybrid Backbone Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 64-Layer Backbone Explorer (#w-layers)
     ------------------------------------------------------------ */
  const strip = $("layerStrip");
  const info = $("layerInfo");
  const layerCtx = $("layerCtx");
  const layerDtype = $("layerDtype");

  if (strip && info) {
    strip.innerHTML = "";
    const kinds = [];
    for (let s = 0; s < 16; s++) kinds.push("delta", "delta", "delta", "full");

    kinds.forEach((kind, i) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "layer-cell" + (kind === "full" ? " fa" : "");
      cell.title = `第 ${i + 1} 层`;
      cell.setAttribute("aria-label", `第 ${i + 1} 层：${kind === "full" ? "门控注意力" : "门控 DeltaNet"}`);
      cell.setAttribute("aria-pressed", "false");
      strip.appendChild(cell);
    });

    function showLayer(cell) {
      const i = [...strip.children].indexOf(cell);
      const kind = kinds[i];

      const ctx = layerCtx ? +layerCtx.value : 262144;
      const bp = layerDtype ? +layerDtype.value : 2;

      const isDelta = kind === "delta";
      const motifIndex = (i % 4) + 1;
      const blockIndex = Math.floor(i / 4) + 1;

      // Cumulative attention layers up to this layer
      const numAttnSoFar = kinds.slice(0, i + 1).filter(k => k === "full").length;
      const numDeltaSoFar = (i + 1) - numAttnSoFar;

      // KV cache for attention layers: gqaKV heads * gqaDh dim * 2 (K+V) * bp bytes per token
      const kvBytesPerLayer = 2 * CFG.gqaKV * CFG.gqaDh * bp * ctx;
      const cumulativeKvGb = (numAttnSoFar * kvBytesPerLayer) / 1e9;

      // Parameters so far — same per-layer expressions the W1 audit uses
      const cumParams = (numDeltaSoFar * (CFG.dnParams + CFG.ffnParams)
        + numAttnSoFar * (CFG.faParams + CFG.ffnParams)) / 1e9;

      info.innerHTML = `
          <div class="calc-stat-grid">
            <div class="cs-card">
              <div class="cs-num" style="color:${isDelta ? "var(--amber)" : "var(--mint)"}">第 ${i + 1} / 64 层</div>
              <div class="cs-lbl">${isDelta ? "门控 DeltaNet（3Δ）" : "门控注意力（1A）"}</div>
            </div>
            <div class="cs-card">
              <div class="cs-num">块 ${blockIndex} · 步骤 ${motifIndex}/4</div>
              <div class="cs-lbl">${isDelta ? "循环上下文混合器" : "全局精确召回锚点"}</div>
            </div>
            <div class="cs-card">
              <div class="cs-num" style="color:var(--mint)">${cumulativeKvGb.toFixed(2)} GB</div>
              <div class="cs-lbl">${ctx.toLocaleString()} 词元时的累计 KV 缓存</div>
            </div>
            <div class="cs-card">
              <div class="cs-num" style="color:var(--iris)">${cumParams.toFixed(2)}B</div>
              <div class="cs-lbl">累计参数（L1..L${i+1}）</div>
            </div>
          </div>
          <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
            ${isDelta
              ? `<b>线性注意力：</b>16 个 QK / 48 个 V 头 × 128 维 · 每个头使用固定的 $128 \\times 128$ 状态（动态 KV 缓存为 0 字节）。解码 FLOPs 恒为 $\\mathcal{O}(1)$。`
              : `<b>完整 Softmax 注意力：</b>24 个 Q / 4 个 KV 头 × 256 维 · GQA 6:1 · 3D MRoPE $[11, 11, 10]$ · $\\sigma$ 输出门控。缓存增加 ${(kvBytesPerLayer/1e9).toFixed(2)} GB。`}
          </div>
        `;
      katexRender(info);                                          // footnote carries inline math
    }

    // exclusive selection via bindGroup; .sel mirrors the legacy CSS hook
    bindGroup(strip, cell => {
      strip.querySelectorAll(".layer-cell").forEach(c => c.classList.remove("sel"));
      cell.classList.add("sel");
      showLayer(cell);
    }, ".layer-cell");

    // roving-focus keyboard nav (←/→ move & select, Home/End jump)
    strip.addEventListener("keydown", e => {
      const cells = [...strip.querySelectorAll(".layer-cell")];
      const cur = cells.indexOf(document.activeElement);
      let next = -1;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = Math.min(cells.length - 1, cur + 1);
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = Math.max(0, cur - 1);
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = cells.length - 1;
      if (next >= 0 && cur >= 0) { e.preventDefault(); cells[next].focus(); cells[next].click(); }
    });

    [layerCtx, layerDtype].forEach(id => {
      if (id) id.addEventListener("change", () => strip.querySelector(".layer-cell.sel")?.click());
    });

    strip.children[3]?.click();
  }


  /* ------------------------------------------------------------
     2. Widget: 4-Layer Super-Block Motif Visualizer (#w-hybrid-motif)
     ------------------------------------------------------------ */
  const motifStepCard = $("motifStepCard");
  const motifStepSeg = $("motifStepSeg");

  const MOTIF_STEPS = {
    "1": {
      name: "超块步骤 1 · DeltaNet 循环压缩",
      type: "线性注意力混合器 + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "四层循环中的第一次循环更新。摄取原始输入词元，应用因果一维深度卷积（核大小 4）提取局部词元顺序，并将关联键值更新写入固定的 128×128 记忆矩阵。",
      benefit: "纯 $\\mathcal{O}(1)$ 流式状态——KV 缓存为 0 字节。"
    },
    "2": {
      name: "超块步骤 2 · DeltaNet 循环细化",
      type: "线性注意力混合器 + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "第二次循环传递。多层深度使 DeltaNet 能够组合关联记忆，在压缩后的前缀状态上执行多步推理和语义特征提取。",
      benefit: "通过学习到的衰减门控 $\\alpha_t$ 过滤瞬时语法噪声。"
    },
    "3": {
      name: "超块步骤 3 · DeltaNet 循环融合",
      type: "线性注意力混合器 + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "第三次循环传递。在到达完整注意力锚点前融合长程上下文，最大化每 FLOP 所压缩的序列历史量。",
      benefit: "在四层超块中实现最高的 FLOP 效率。"
    },
    "4": {
      name: "超块步骤 4 · 门控注意力检索锚点",
      type: "分组查询注意力（GQA 6:1）+ SwiGLU FFN",
      state: "Attn(Q, K, V) = Softmax((Q K^T / √d_k) + M) V",
      desc: "全局同步锚点。对整个历史词元序列执行完整的因果 Softmax 注意力，重新读取原始词元，以纠正前 3 个 DeltaNet 层中的关联漂移。",
      benefit: "保证 100% 的大海捞针检索准确率。"
    }
  };

  function renderMotifStep(stepNum) {
    if (!motifStepCard) return;
    const info = MOTIF_STEPS[stepNum] || MOTIF_STEPS["1"];

    motifStepCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:12px;color:${stepNum === "4" ? "var(--mint)" : "var(--amber)"};font-weight:600">${info.type}</span>
      </div>
      <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);font-family:var(--mono);font-size:12px;color:var(--amber);margin-bottom:8px">
        ${info.state}
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        ${info.desc}
      </div>
      <div style="font-size:12px;color:var(--faint);font-family:var(--mono)">
        <b>模式优势：</b>${info.benefit}
      </div>
    `;
    katexRender(motifStepCard);
  }

  if (motifStepSeg) {
    bindGroup(motifStepSeg, b => renderMotifStep(b.dataset.step));
  }
  renderMotifStep("1");
})();


/* ============================================================
   Section 04: Gated DeltaNet & Associative Memory Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Interactive Delta-Rule Associative Memory Lab (#w-delta)
     ------------------------------------------------------------ */
  const deltaLabReadout = $("deltaLabReadout");
  const deltaKeySel = $("deltaKeySel");
  const deltaValInput = $("deltaValInput");
  const deltaModeSel = $("deltaModeSel");
  const btnDeltaWrite = $("btnDeltaWrite");
  const btnDeltaDecay = $("btnDeltaDecay");
  const btnDeltaReset = $("btnDeltaReset");
  const btnScenarioOverwrite = $("btnScenarioOverwrite");
  const btnScenarioMulti = $("btnScenarioMulti");

  const VALUE_PRESETS = {
    "alpha": { name: "Alpha", vec: [1.0, 0.6, 0.2, -0.4], label: "[+1.0, +0.6, +0.2, -0.4]" },
    "beta":  { name: "Beta",  vec: [-0.8, 1.0, 0.5, 0.0], label: "[-0.8, +1.0, +0.5, 0.0]" },
    "gamma": { name: "Gamma", vec: [0.0, -0.7, 1.0, 0.8], label: "[0.0, -0.7, +1.0, +0.8]" },
    "delta": { name: "Delta", vec: [0.5, 0.8, -0.9, 0.3], label: "[+0.5, +0.8, -0.9, +0.3]" },
    "zero":  { name: "零向量",  vec: [0.0, 0.0, 0.0, 0.0], label: "[0.0, 0.0, 0.0, 0.0]" }
  };

  const SLOT_CONFIG = [
    { label: "槽位 0 · 用户意图", desc: "表示对话目标向量" },
    { label: "槽位 1 · 文档上下文", desc: "长程文档状态" },
    { label: "槽位 2 · 工具执行", desc: "RPC 状态 / 环境响应" },
    { label: "槽位 3 · 工作变量", desc: "活跃推理实体" }
  ];

  // 4x4 Associative Matrix (4 key slots x 4 value feature dims)
  let S = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ];
  let slotHistory = [[], [], [], []]; // Tracks writes per slot
  let expectedSlots = [null, null, null, null];
  let writeHistory = [];

  function getCellBg(val) {
    if (Math.abs(val) < 0.01) return "transparent";
    if (val > 0) {
      const alpha = Math.min(0.35, Math.abs(val) * 0.28 + 0.08);
      return `rgba(29, 110, 107, ${alpha})`;
    } else {
      const alpha = Math.min(0.35, Math.abs(val) * 0.28 + 0.08);
      return `rgba(184, 52, 31, ${alpha})`;
    }
  }

  function renderDeltaLab() {
    if (!deltaLabReadout) return;

    const activeSlot = deltaKeySel ? +deltaKeySel.value : 0;
    const mode = deltaModeSel ? deltaModeSel.value : "deltanet";

    // Frobenius Norm: ||S||_F = sqrt(sum S_ij^2)
    let fNorm = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        fNorm += S[r][c] * S[r][c];
      }
    }
    fNorm = Math.sqrt(fNorm);

    // Live Query for selected slot: y = S[activeSlot]
    const retrievedVec = S[activeSlot];
    const targetInfo = expectedSlots[activeSlot];
    let recallError = 0;
    let hasTarget = !!targetInfo;
    if (hasTarget) {
      for (let c = 0; c < 4; c++) {
        const diff = retrievedVec[c] - targetInfo.vec[c];
        recallError += diff * diff;
      }
      recallError = Math.sqrt(recallError);
    }

    // Build 4x4 matrix table rows
    const matrixRowsHtml = S.map((row, rIdx) => {
      const isActive = rIdx === activeSlot;
      const slotName = SLOT_CONFIG[rIdx].label.split("·")[1].trim();
      const cells = row.map(v => {
        const color = v > 0.01 ? "var(--mint)" : (v < -0.01 ? "var(--rose)" : "var(--faint)");
        return `<td><span class="delta-cell-val" style="background:${getCellBg(v)};color:${color}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></td>`;
      }).join("");

      return `
        <tr class="${isActive ? "active-row" : ""}" data-slot="${rIdx}" tabindex="0" role="button" aria-label="选择槽位 ${rIdx}">
          <td style="font-weight:600;color:${isActive ? "var(--amber)" : "var(--fg)"}">
            <b>k_${rIdx}</b> <span style="font-size:12px;color:var(--faint)">(${slotName})</span>
          </td>
          ${cells}
        </tr>
      `;
    }).join("");

    // Build vector readout string for query probe
    const retrievedFormatted = retrievedVec.map(v => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`).join(", ");
    const targetFormatted = hasTarget ? targetInfo.vec.map(v => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`).join(", ") : "尚未写入";

    deltaLabReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${fNorm.toFixed(2)}</div>
          <div class="cs-lbl">Frobenius 范数 ||S||_F</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:${mode === "deltanet" ? "var(--mint)" : "var(--rose)"}">
            ${mode === "deltanet" ? "Delta 规则" : "加法式"}
          </div>
          <div class="cs-lbl">当前更新机制</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:${hasTarget ? (recallError < 0.05 ? "var(--mint)" : "var(--rose)") : "var(--muted)"}">
            ${hasTarget ? (recallError < 0.05 ? "精确（0.00）" : `+${recallError.toFixed(2)} 误差`) : "—"}
          </div>
          <div class="cs-lbl">k_${activeSlot} 召回精度</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${writeHistory.length} 次操作</div>
          <div class="cs-lbl">操作计数器</div>
        </div>
      </div>

      <div class="delta-matrix-wrap">
        <div class="delta-matrix-box">
          <div class="delta-box-title">
            <span>关联矩阵 S_t ∈ ℝ^(4×4)</span>
            <span style="font-size:12px;color:var(--faint)">k_i × v_j（§04 中有 48 个头）</span>
          </div>
          <table class="delta-grid-table">
            <thead>
              <tr>
                <th scope="col">槽位键（k）</th>
                <th scope="col">v[0]</th>
                <th scope="col">v[1]</th>
                <th scope="col">v[2]</th>
                <th scope="col">v[3]</th>
              </tr>
            </thead>
            <tbody>
              ${matrixRowsHtml}
            </tbody>
          </table>
        </div>

        <div class="delta-probe-box">
          <div class="delta-box-title">
            <span>查询探针：y_t = S_t^T q_t</span>
            <span style="font-size:12px;color:var(--amber)">正在探测 k_${activeSlot}</span>
          </div>
          <div style="font-size:12px;color:var(--muted);line-height:1.6">
            <div style="margin-bottom:4px"><b>检索信号 y：</b><code>[ ${retrievedFormatted} ]</code></div>
            <div style="margin-bottom:6px"><b>预期目标：</b><code>[ ${targetFormatted} ]</code></div>
            
            <div style="padding:6px 8px;border-radius:4px;background:var(--bg-1);margin-top:6px">
              ${!hasTarget 
                ? '<span style="color:var(--faint)">槽位为空。写入载荷以观察关联保持效果。</span>'
                : (recallError < 0.05 
                    ? '<span style="color:var(--mint);font-weight:600">✓ 干净的记忆检索：</span>Delta 规则减法 <code>(I - β k k^T)</code> 完全消除了先前状态，没有重影干扰。'
                    : `<span style="color:var(--rose);font-weight:600">⚠️ 叠加重影（+${recallError.toFixed(2)}）：</span>加法式线性注意力同时保留过去的写入，造成无法恢复的特征串扰。`)}
            </div>
          </div>
        </div>
      </div>

      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px;margin-top:10px">
        ${writeHistory.length === 0 
          ? "状态矩阵已归零。请在上方选择更新规则、目标槽位和特征载荷，或点击一个<b>引导场景</b>。"
          : `<b>上次操作：</b>${writeHistory[writeHistory.length - 1]}`}
      </div>
    `;
    katexRender(deltaLabReadout);

    // Bind row click & keydown to change active slot
    const rows = deltaLabReadout.querySelectorAll(".delta-grid-table tr[data-slot]");
    rows.forEach(r => {
      const selectRow = () => {
        const slot = r.getAttribute("data-slot");
        if (deltaKeySel && slot) {
          deltaKeySel.value = slot;
          renderDeltaLab();
        }
      };
      r.addEventListener("click", selectRow);
      r.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectRow();
        }
      });
    });
  }

  function writeDelta(slotIdx, valKey, mode) {
    const payload = VALUE_PRESETS[valKey] || VALUE_PRESETS["alpha"];
    const v = payload.vec;
    const beta = 1.0;

    slotHistory[slotIdx].push({ valKey, vec: [...v], mode });
    expectedSlots[slotIdx] = { valKey, name: payload.name, vec: [...v] };

    if (mode === "deltanet") {
      // S_t = S_{t-1}(I - beta k k^T) + beta k v^T
      for (let c = 0; c < 4; c++) {
        S[slotIdx][c] = S[slotIdx][c] * (1.0 - beta) + beta * v[c];
      }
      writeHistory.push(`DeltaNet：将 <b>${payload.name}</b> 写入槽位 ${slotIdx} · 通过 <code>(I - β k k^T)</code> 清除陈旧状态`);
    } else {
      // Pure Linear Attention: S_t = S_{t-1} + k v^T
      for (let c = 0; c < 4; c++) {
        S[slotIdx][c] += v[c];
      }
      writeHistory.push(`加法式线性注意力：将 <b>${payload.name}</b> 加入槽位 ${slotIdx} · 叠加累积`);
    }
    renderDeltaLab();
  }

  function decayDelta() {
    const alpha = 0.85;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        S[r][c] = +(S[r][c] * alpha).toFixed(4);
      }
      if (expectedSlots[r]) {
        expectedSlots[r].vec = expectedSlots[r].vec.map(x => +(x * alpha).toFixed(4));
      }
    }
    writeHistory.push(`已应用全局衰减：<code>S ← 0.85 · S</code>（依赖数据的保持遗忘）`);
    renderDeltaLab();
  }

  function resetDelta() {
    S = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ];
    slotHistory = [[], [], [], []];
    expectedSlots = [null, null, null, null];
    writeHistory = [];
    renderDeltaLab();
  }

  if (btnDeltaWrite) {
    btnDeltaWrite.addEventListener("click", () => {
      const slot = deltaKeySel ? +deltaKeySel.value : 0;
      const valKey = deltaValInput ? deltaValInput.value : "alpha";
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      writeDelta(slot, valKey, mode);
    });
  }

  if (btnDeltaDecay) btnDeltaDecay.addEventListener("click", decayDelta);
  if (btnDeltaReset) btnDeltaReset.addEventListener("click", resetDelta);

  if (deltaKeySel) deltaKeySel.addEventListener("change", renderDeltaLab);
  if (deltaModeSel) deltaModeSel.addEventListener("change", renderDeltaLab);

  if (btnScenarioOverwrite) {
    btnScenarioOverwrite.addEventListener("click", () => {
      resetDelta();
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      if (deltaKeySel) deltaKeySel.value = "0";
      // Step 1: Write Alpha
      writeDelta(0, "alpha", mode);
      // Step 2: Overwrite with Beta
      writeDelta(0, "beta", mode);
    });
  }

  if (btnScenarioMulti) {
    btnScenarioMulti.addEventListener("click", () => {
      resetDelta();
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      writeDelta(0, "alpha", mode);
      writeDelta(1, "beta", mode);
      writeDelta(2, "gamma", mode);
      writeDelta(3, "delta", mode);
      if (deltaKeySel) deltaKeySel.value = "0";
      renderDeltaLab();
    });
  }

  renderDeltaLab();


  /* ------------------------------------------------------------
     2. Widget: Chunked WY Parallel Scan Visualizer (#w-deltanet-scan)
     ------------------------------------------------------------ */
  const wyScanVisual = $("wyScanVisual");
  const wyChunkSeg = $("wyChunkSeg");
  let curChunkSize = 64;

  function renderWyScan() {
    if (!wyScanVisual) return;

    const totalTokens = 262144;
    const numChunks = totalTokens / curChunkSize;
    const flopsPerChunk = (curChunkSize * curChunkSize * 128 * 2) / 1e6; // MFLOPs

    wyScanVisual.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${numChunks.toLocaleString()} 个分块</div>
          <div class="cs-lbl">顺序执行的分块间步骤</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${curChunkSize} 个词元</div>
          <div class="cs-lbl">并行分块内块</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${flopsPerChunk.toFixed(2)} MFLOPs</div>
          <div class="cs-lbl">分块内 GEMM 计算量</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">100% Tensor Core</div>
          <div class="cs-lbl">硬件饱和度</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-top:2px solid var(--mint);margin-top:8px">
        ✓ <b>分块 WY 表示：</b>顺序乘积 $\\prod_{t=1}^C (I - \\beta_t k_t k_t^T)$ 可分解为块三角表示 $(I - W Y^T)^{-1}$，将 $\\mathcal{O}(T)$ 循环转换为同时跨 $C=${curChunkSize}$ 个词元的稠密并行 GEMM。
      </div>
    `;
    katexRender(wyScanVisual);
  }

  if (wyChunkSeg) {
    bindGroup(wyChunkSeg, b => { curChunkSize = +b.dataset.chunk; renderWyScan(); });
  }
  renderWyScan();
})();

/* ============================================================
   Section 05: Gated Attention & 3D MRoPE Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: GQA KV-Cache Memory & Bandwidth Calculator (#w-cache)
     ------------------------------------------------------------ */
  const cacheReadout = $("cacheReadout");
  const cacheCtx = $("cacheCtx");
  const cacheBatch = $("cacheBatch");
  const cacheDtype = $("cacheDtype");

  function renderCacheCalc() {
    if (!cacheReadout) return;
    const T = cacheCtx ? +cacheCtx.value : 262144;
    const B = cacheBatch ? +cacheBatch.value : 1;
    const bp = cacheDtype ? +cacheDtype.value : 2;

    // Attention-layer geometry straight from CFG — stays in sync with the W1 audit
    const nLayersAttn = CFG.nFull;
    const nKv = CFG.gqaKV;
    const dH = CFG.gqaDh;

    // Bytes per token per layer = 2 * n_kv * d_h * bp (K and V); decimal GB
    const bytesPerTokLayer = 2 * nKv * dH * bp;
    const totalGqaBytes = bytesPerTokLayer * nLayersAttn * T * B;
    const totalGqaGb = totalGqaBytes / 1e9;

    // Full MHA baseline (all 64 layers at gqaQ KV heads)
    const bytesPerTokMha = 2 * CFG.gqaQ * dH * bp;
    const totalMhaBytes = bytesPerTokMha * CFG.layers * T * B;
    const totalMhaGb = totalMhaBytes / 1e9;

    const savingsFactor = totalGqaGb > 0 ? (totalMhaGb / totalGqaGb).toFixed(1) : "—";

    cacheReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${totalGqaGb.toFixed(2)} GB</div>
          <div class="cs-lbl">GQA-4 缓存（16 层）</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${bytesPerTokLayer.toLocaleString()} B</div>
          <div class="cs-lbl">字节 / 词元 / 层</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--rose)">${totalMhaGb.toFixed(1)} GB</div>
          <div class="cs-lbl">标准 MHA（64 层 · 24 头）</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">减少 ${savingsFactor}×</div>
          <div class="cs-lbl">显存压缩</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        序列：<b>${T.toLocaleString()} 个词元</b> × 批大小 <b>${B}</b> @ <b>${bp === 2 ? "BF16" : (bp === 1 ? "FP8" : "FP4")}</b> · 48 个 DeltaNet 层的动态 KV 缓存需求为 <b>0 GB</b>（固定循环矩阵）。
      </div>
    `;
  }

  [cacheCtx, cacheBatch, cacheDtype].forEach(el => {
    if (el) {
      el.addEventListener("change", renderCacheCalc);
      el.addEventListener("input", renderCacheCalc);
    }
  });
  renderCacheCalc();


  /* ------------------------------------------------------------
     2. Widget: 3D MRoPE Phase Visualizer (#w-rope)
     ------------------------------------------------------------ */
  const svg = $("ropeSvg");
  const ropeOut = $("ropeOut");
  const ropePos = $("ropePos");
  const ropeDim = $("ropeDim");
  const mropeAxis = $("mropeAxis");

  if (svg && ropeOut) {
    const NS = "http://www.w3.org/2000/svg";
    const cx = 160, cy = 160, R = 108;

    function el(t, attrs, parent) {
      const e = document.createElementNS(NS, t);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(e);
      return e;
    }

    // Dial background — ink strokes on parchment (PAL mirrors the :root tokens)
    el("circle", { cx, cy, r: R + 18, fill: "rgba(184,52,31,.04)" });
    el("circle", { cx, cy, r: R, fill: "none", stroke: "rgba(26,22,18,.14)", "stroke-width": 1.2 });
    el("circle", { cx, cy, r: R * .66, fill: "none", stroke: "rgba(26,22,18,.06)" });
    el("circle", { cx, cy, r: R * .33, fill: "none", stroke: "rgba(26,22,18,.06)" });
    el("line", { x1: cx - R - 14, y1: cy, x2: cx + R + 14, y2: cy, stroke: "rgba(26,22,18,.08)" });
    el("line", { x1: cx, y1: cy - R - 14, x2: cx, y2: cy + R + 14, stroke: "rgba(26,22,18,.08)" });

    for (let k = 0; k < 12; k++) {
      const a = k * Math.PI / 6;
      const x1 = cx + (R - 2) * Math.sin(a), y1 = cy - (R - 2) * Math.cos(a);
      const x2 = cx + (R + 4) * Math.sin(a), y2 = cy - (R + 4) * Math.cos(a);
      el("line", { x1, y1, x2, y2, stroke: "rgba(26,22,18,.18)" });
    }

    const arcTrail = el("path", { fill: "none", stroke: "rgba(184,52,31,.35)", "stroke-width": 3, "stroke-linecap": "round" });
    const hand = el("line", { x1: cx, y1: cy, x2: cx, y2: cy - R, stroke: PAL.amber, "stroke-width": 2.5, "stroke-linecap": "round" });
    const tip = el("circle", { r: 6, fill: PAL.amber, stroke: "rgba(26,22,18,.35)", "stroke-width": 1.2 });

    function theta(j) {
      return Math.pow(CFG.ropeTheta, -2 * j / CFG.rotDims);
    }

    function renderRope() {
      const m = ropePos ? +ropePos.value : 64;
      const j = ropeDim ? +ropeDim.value : 0;
      const axis = mropeAxis ? mropeAxis.value : "time";

      const ang = m * theta(j);
      const hx = cx + R * Math.sin(ang), hy = cy - R * Math.cos(ang);
      hand.setAttribute("x2", hx); hand.setAttribute("y2", hy);
      tip.setAttribute("cx", hx); tip.setAttribute("cy", hy);

      const twoPi = 2 * Math.PI;
      const norm = ((ang % twoPi) + twoPi) % twoPi;
      const large = norm > Math.PI ? 1 : 0;
      arcTrail.setAttribute("d", `M ${cx} ${cy - R} A ${R} ${R} 0 ${large} 1 ${hx} ${hy}`);

      const turns = ang / twoPi;
      const wl = twoPi / theta(j);
      const wlTxt = wl >= 1e6 ? (wl / 1e6).toFixed(2) + "M" : (wl >= 1e3 ? (wl / 1e3).toFixed(1) + "k" : wl.toFixed(0));

      const axisLabel = axis === "time" ? "时间轴（T）" : (axis === "height" ? "垂直轴（H）" : "水平轴（W）");

      ropeOut.innerHTML = `
        <div class="rf-stat-box">
          <div class="rf-head">维度对 j = ${j} 的旋转相位</div>
          <div class="rf-row"><span>目标轴</span><span class="val amber">${axisLabel}</span></div>
          <div class="rf-row"><span>相位旋转角</span><span class="val mint">${norm.toFixed(2)} 弧度（${turns.toFixed(1)} 圈）</span></div>
          <div class="rf-row"><span>空间波长</span><span class="val">${wlTxt} 词元 / 周期</span></div>
          <div class="rf-row"><span>角频率 θ_j</span><span class="val" style="color:var(--iris)">${theta(j).toExponential(2)} 弧度/词元</span></div>
        </div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
          维度对 j=${j} 提供${j < 8 ? "高频局部语法辨别能力" : "长程全局文档相位定位能力"}。3D MRoPE 在三个轴上交错排列 $[11, 11, 10]$ 个维度对。
        </div>
      `;
      katexRender(ropeOut);
      if (ropePos?.type === "range") ropePos.setAttribute("aria-valuetext", `${ropePos.value} 词元位置`);
      if (ropeDim?.type === "range") ropeDim.setAttribute("aria-valuetext", `维度对 ${ropeDim.value}`);
    }

    [ropePos, ropeDim, mropeAxis].forEach(el => {
      if (el) {
        el.addEventListener("input", renderRope);
        el.addEventListener("change", renderRope);
      }
    });
    renderRope();
  }


  /* ------------------------------------------------------------
     3. Widget: Attention Output Sigmoid Gate Simulator (#w-attn-gate)
     ------------------------------------------------------------ */
  const attnGateVisual = $("attnGateVisual");
  const attnGateModeSeg = $("attnGateModeSeg");
  let attnGateMode = "active";

  function renderAttnGate() {
    if (!attnGateVisual) return;

    if (attnGateMode === "active") {
      attnGateVisual.innerHTML = `
        <div class="calc-stat-grid">
          <div class="cs-card">
            <div class="cs-num" style="color:var(--mint)">σ = 0.94（开启）</div>
            <div class="cs-lbl">输出门控值</div>
          </div>
          <div class="cs-card">
            <div class="cs-num">100%</div>
            <div class="cs-lbl">信号传输率</div>
          </div>
          <div class="cs-card">
            <div class="cs-num" style="color:var(--amber)">高注意力</div>
            <div class="cs-lbl">检索置信度</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--mint);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-top:2px solid var(--mint);margin-top:8px">
          ✓ <b>已摄取相关上下文：</b>输出 sigmoid 门控 $\sigma(x W_g) \approx 0.94$ 将完整注意力值传入 $W_{\text{out}}$，把清晰的事实检索结果送入残差流。
        </div>
      `;
    } else {
      attnGateVisual.innerHTML = `
        <div class="calc-stat-grid">
          <div class="cs-card">
            <div class="cs-num" style="color:var(--rose)">σ = 0.08（钳制）</div>
            <div class="cs-lbl">输出门控值</div>
          </div>
          <div class="cs-card">
            <div class="cs-num">8.0%</div>
            <div class="cs-lbl">信号传输率</div>
          </div>
          <div class="cs-card">
            <div class="cs-num" style="color:var(--faint)">噪声已滤除</div>
            <div class="cs-lbl">干扰项抑制</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--rose);line-height:1.5;background:rgba(138,51,36,0.06);padding:8px 12px;border-radius:6px;border-top:2px solid var(--rose);margin-top:8px">
          ✗ <b>已抑制干扰上下文：</b>当检索上下文含有陈旧历史时，学习到的门控关闭（$\sigma \approx 0.08$），防止带噪注意力 logits 污染下游 DeltaNet 状态。
        </div>
      `;
    }
    katexRender(attnGateVisual);
  }

  if (attnGateModeSeg) {
    bindGroup(attnGateModeSeg, b => { attnGateMode = b.dataset.gate; renderAttnGate(); });
  }
  renderAttnGate();
})();


/* ============================================================
   Section 06: FFN & Residual Stream Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: SwiGLU Activation Explorer (#w-ffn)
     ------------------------------------------------------------ */
  const svg = $("ffnSvg");
  if (svg) {
    const NS = "http://www.w3.org/2000/svg", W = 720, H = 300, ox = 56, oy = 250, pw = 620, ph = 200;
    const X = x => ox + (x + 6) / 12 * pw, Y = y => oy - y / 24 * ph;

    function line(x1, y1, x2, y2, st) {
      const l = document.createElementNS(NS, "line");
      Object.entries({ x1, y1, x2, y2, ...st }).forEach(([k, v]) => l.setAttribute(k, v));
      svg.appendChild(l);
      return l;
    }
    function path(d, st) {
      const p = document.createElementNS(NS, "path");
      Object.entries({ d, fill: "none", ...st }).forEach(([k, v]) => p.setAttribute(k, v));
      svg.appendChild(p);
      return p;
    }
    function txt(x, y, s, fill, anchor, size) {
      const t = document.createElementNS(NS, "text");
      Object.entries({ x, y, fill: fill || PAL.muted, "text-anchor": anchor || "start", "font-size": size || 11, "font-family": '"IBM Plex Mono", ui-monospace, monospace' }).forEach(([k, v]) => t.setAttribute(k, v));
      t.textContent = s;
      svg.appendChild(t);
      return t;
    }

    for (let v = 0; v <= 20; v += 4) {
      const y = Y(v);
      line(ox, y, ox + pw, y, { stroke: "rgba(26,22,18,.05)" });
      txt(ox - 8, y + 3, String(v), PAL.muted, "end", 9);
    }
    for (let x = -6; x <= 6; x += 2) {
      const px = X(x);
      line(px, oy, px, oy + 4, { stroke: "rgba(26,22,18,.18)" });
      txt(px, oy + 16, String(x), PAL.muted, "middle", 9);
    }
    line(ox, oy, ox + pw, oy, { stroke: "rgba(26,22,18,.18)" });
    line(X(0), Y(0), X(0), Y(22), { stroke: "rgba(26,22,18,.18)" });

    const relu = x => Math.max(0, x);
    const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    const silu = x => x / (1 + Math.exp(-x));

    function plot(fn, color, dash) {
      let d = "";
      for (let x = -6; x <= 6.001; x += 0.15) d += (x === -6 ? "M" : "L") + X(x) + "," + Y(fn(x)) + " ";
      path(d, { stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "none" });
    }
    plot(relu, PAL.muted, "3 3");
    plot(gelu, PAL.mint);
    plot(silu, PAL.amber);

    const unitPath = path("", { stroke: PAL.iris, "stroke-width": 2.5 });
    const marker = (() => {
      const c = document.createElementNS(NS, "circle");
      Object.entries({ r: 6, fill: PAL.iris, stroke: PAL.paper, "stroke-width": 1.5, cx: -99, cy: -99 }).forEach(([k, v]) => c.setAttribute(k, v));
      svg.appendChild(c);
      return c;
    })();

    // swatch and label share the same pigment so they never disagree
    const legend = document.createElement("div");
    legend.className = "ffn-legend";
    legend.innerHTML = `
      <span class="lg-item" style="color:${PAL.muted}"><span class="lg-sw" style="background:transparent;border-top:1.5px dashed ${PAL.muted}"></span>ReLU</span>
      <span class="lg-item" style="color:${PAL.mint}"><span class="lg-sw" style="background:${PAL.mint}"></span>GELU</span>
      <span class="lg-item" style="color:${PAL.amber}"><span class="lg-sw" style="background:${PAL.amber}"></span>SiLU</span>
      <span class="lg-item" style="color:${PAL.iris}"><span class="lg-sw" style="background:${PAL.iris}"></span>SwiGLU 单元（SiLU(x)·x）</span>
    `;
    svg.parentNode.insertBefore(legend, svg.nextSibling);

    function update(x) {
      x = Math.max(-6, Math.min(6, x));
      let d = "";
      for (let xx = -6; xx <= 6.001; xx += 0.15) d += (xx === -6 ? "M" : "L") + X(xx) + "," + Y(silu(xx) * xx) + " ";
      unitPath.setAttribute("d", d);
      marker.setAttribute("cx", X(x)); marker.setAttribute("cy", Y(silu(x) * x));
      if ($("ffnOut")) {
        $("ffnOut").innerHTML = `
          <div class="row"><span>x = ${x.toFixed(2)} · SiLU(x) = ${silu(x).toFixed(2)} · SwiGLU 门控输出 = <b style="color:var(--iris)">${(silu(x) * x).toFixed(2)}</b></span><span class="v"></span></div>
          <div class="row"><span>ReLU 传递${relu(x) > 0 ? "线性信号" : "零值"} · GELU 进行软门控 · SiLU 保留平滑负半叶以维持梯度稳定</span><span class="v"></span></div>
          <div style="color:var(--faint);font-size:12px;margin-top:4px">跨 17,408 个隐藏单元的乘法门控 $(x W_{\\text{gate}}) \\odot \\text{SiLU}(x W_{\\text{up}})$ 充当世界知识的键值记忆检索。</div>
        `;
        katexRender($("ffnOut"));
      }
      const ffnX = $("ffnX");
      if (ffnX) {
        ffnX.value = x;
        if (ffnX.type === "range") ffnX.setAttribute("aria-valuetext", `x = ${x.toFixed(2)}`);
      }
    }

    function updateFromPointer(e) {
      const r = svg.getBoundingClientRect();
      update(((e.clientX - r.left) / r.width * W - ox) / pw * 12 - 6);
    }
    svg.addEventListener("pointermove", updateFromPointer);
    if ($("ffnX")) $("ffnX").addEventListener("input", e => update(+e.target.value));
    update(0);
  }


  /* ------------------------------------------------------------
     2. Widget: Residual Bus Propagation & RMSNorm Tracker (#w-residual-bus)
     ------------------------------------------------------------ */
  const resBusSvg = $("resBusSvg");
  const resBusStats = $("resBusStats");
  const resBusNormSeg = $("resBusNormSeg");
  let resBusMode = "prenorm";

  function renderResBus() {
    if (!resBusSvg || !resBusStats) return;

    const NS = "http://www.w3.org/2000/svg";
    resBusSvg.innerHTML = "";

    const W = 560, H = 220, P = { l: 48, r: 24, t: 20, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Layers 0 to 64
    const fx = l => P.l + (l / 64) * IW;
    const fy = v => P.t + (1 - Math.min(1.0, v / 10.0)) * IH;

    [0, 16, 32, 48, 64].forEach(lVal => {
      const gx = fx(lVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      resBusSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `L${lVal}`;
      resBusSvg.appendChild(txt);
    });

    [1, 2, 5, 10].forEach(vVal => {
      const gy = fy(vVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      resBusSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = `${vVal}σ`;
      resBusSvg.appendChild(txt);
    });

    let dPath = "";
    for (let layer = 0; layer <= 64; layer++) {
      let variance;
      if (resBusMode === "prenorm") {
        variance = 1.0 + 0.12 * Math.sin(layer / 4) + 0.05 * (layer / 64);
      } else {
        variance = 1.0 + Math.pow(layer / 20, 1.8);
      }
      const gx = fx(layer);
      const gy = fy(variance);
      dPath += (layer === 0 ? "M" : "L") + gx + "," + gy + " ";
    }

    const pathEl = document.createElementNS(NS, "path");
    pathEl.setAttribute("d", dPath);
    pathEl.setAttribute("fill", "none");
    pathEl.setAttribute("stroke", resBusMode === "prenorm" ? PAL.mint : PAL.rose);
    pathEl.setAttribute("stroke-width", "2.5");
    resBusSvg.appendChild(pathEl);

    resBusStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">残差通信总线仲裁</div>
        <div class="rf-row"><span>总线宽度</span><span class="val amber">d_model = 5,120</span></div>
        <div class="rf-row"><span>RMSNorm 堆叠</span><span class="val mint">129 个核心归一化（分支前）</span></div>
        <div class="rf-row"><span>第 64 层信号方差</span><span class="val ${resBusMode === "prenorm" ? "mint" : "rose"}">${resBusMode === "prenorm" ? "1.05σ（有界）✓" : "9.82σ（爆炸）✗"}</span></div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
        ${resBusMode === "prenorm" 
          ? "✓ 前置 RMSNorm 确保恒等路径不发生衰减，同时限制 48 个 DeltaNet 和 16 个注意力混合器写回的幅值。"
          : "⚠️ 若无 RMSNorm 总线仲裁，方差会随深度按 $\\mathcal{O}(\\sqrt{L})$ 增长，使 Softmax logits 不稳定并导致数值溢出。"}
      </div>
    `;
    katexRender(resBusStats);
  }

  if (resBusNormSeg) {
    bindGroup(resBusNormSeg, b => { resBusMode = b.dataset.mode; renderResBus(); });
  }
  renderResBus();
})();

/* ============================================================
   Section 07: Vision Front-End & Multimodal Projection Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 5-Stage Vision-to-Language Pipeline (#w-vision-pipe)
     ------------------------------------------------------------ */
  const visionPipe = $("visionPipe");
  const visionDetailCard = $("visionDetailCard");

  const VISION_STAGES = {
    "1": {
      name: "阶段 01 · 图像分块与空间平铺",
      tensor: "输入 [B, C=3, H, W] → 图像块 [B, N_patches, 3×14×14 = 588]",
      desc: "将输入 RGB 图像分解为规则的二维网格，其中包含互不重叠的 14×14 像素图像块。初始二维卷积将 588 个像素值投影为 1,536 维线性嵌入。",
      insight: "动态原生分辨率支持通过将任意宽高比平铺到 224px 坐标块中，避免几何失真和宽高比扭曲。"
    },
    "2": {
      name: "阶段 02 · SigLIP 视觉 Transformer（ViT）编码",
      tensor: "图像块 [B, N, 1536] → ViT 特征 [B, N, 1536]",
      desc: "30 层视觉 Transformer（16 个注意力头、6,144 FFN 宽度）同时对所有图像块应用双向自注意力，提取语义和细粒度视觉表示。",
      insight: "使用 SigLIP 成对 sigmoid 损失而非 CLIP Softmax 进行训练，消除批大小通信瓶颈，并生成更清晰的空间特征。"
    },
    "3": {
      name: "阶段 03 · 2×2 空间邻域拼接",
      tensor: "ViT 特征 [B, N, 1536] → 合并后 [B, N/4, 4×1536 = 6144]",
      desc: "将 2×2 个相邻空间图像块向量组合为统一的 6,144 维表示。视觉词元总数恰好压缩 75%（减少 4 倍）。",
      insight: "这对长上下文预算至关重要：一张 800×600 截图从 2,494 个原始图像块词元缩减到仅 624 个合并词元，从而支持多图对话和长达数小时的视频摄取。"
    },
    "4": {
      name: "阶段 04 · 两层 MLP 投影器与残差对齐",
      tensor: "合并后 [B, N/4, 6144] → 残差流 [B, N/4, 5120]",
      desc: "采用 SiLU 激活的两层 MLP 将 6,144 维视觉向量转换到语言模型的 5,120 维残差空间：Linear(6144 → 6144) → SiLU → Linear(6144 → 5120)。",
      insight: "在多模态 SFT 期间通过精确输出缩放进行训练，以匹配文本预训练的自然激活方差。"
    },
    "5": {
      name: "阶段 05 · 3D MRoPE 坐标与残差交错",
      tensor: "带 [T, H, W] 位置 ID 的残差流 [B, N/4, 5120]",
      desc: "将视觉词元包裹在 `<|vision_start|>` 与 `<|vision_end|>` 分隔符之间。3D MRoPE 分别分配时间、垂直和水平旋转位置频率。",
      insight: "使语言模型能在同一旋转表内理解原生二维布局几何和视频时间连续性。"
    }
  };

  function renderVisionStage(stageNum) {
    if (!visionDetailCard) return;
    const info = VISION_STAGES[stageNum] || VISION_STAGES["1"];

    visionDetailCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:12px;color:var(--mint);font-weight:600">阶段 ${stageNum} / 5</span>
      </div>
      <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);font-family:var(--mono);font-size:12px;color:var(--amber);margin-bottom:8px">
        ${info.tensor}
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        ${info.desc}
      </div>
      <div style="font-size:12px;color:var(--faint);font-family:var(--mono)">
        <b>架构要点：</b>${info.insight}
      </div>
    `;
  }

  if (visionPipe) {
    visionPipe.querySelectorAll(".stage").forEach(btn => {
      btn.addEventListener("click", () => {
        visionPipe.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        renderVisionStage(btn.dataset.stage);
      });
    });
    enableTabs(visionPipe, ".stage");                              // div tabs: focus + Enter/Space
  }
  renderVisionStage("1");


  /* ------------------------------------------------------------
     2. Widget: Vision Token Economics Calculator (#w-vision-calc)
     ------------------------------------------------------------ */
  const visCalcReadout = $("visCalcReadout");
  const visPresetSel = $("visPresetSel");
  const visMergeToggle = $("visMergeToggle");

  const PRESETS = {
    "224x224x1":   { w: 224,  h: 224,  frames: 1,  name: "标准图块（224×224）" },
    "800x600x1":   { w: 800,  h: 600,  frames: 1,  name: "网页截图（800×600）" },
    "1920x1080x1": { w: 1920, h: 1080, frames: 1,  name: "1080p 幻灯片（1920×1080）" },
    "3840x2160x1": { w: 3840, h: 2160, frames: 1,  name: "4K 高分辨率图表（3840×2160）" },
    "448x448x16":  { w: 448,  h: 448,  frames: 16, name: "短视频片段（448×448 · 16 帧）" },
    "640x360x64":  { w: 640,  h: 360,  frames: 64, name: "长视频序列（640×360 · 64 帧）" }
  };

  function renderVisionCalc() {
    if (!visCalcReadout) return;
    const presetKey = visPresetSel ? visPresetSel.value : "800x600x1";
    const useMerge = visMergeToggle ? visMergeToggle.value === "true" : true;

    const p = PRESETS[presetKey] || PRESETS["800x600x1"];
    const patchesPerFrame = Math.ceil(p.w / 14) * Math.ceil(p.h / 14);
    const rawTokens = patchesPerFrame * p.frames;
    const finalTokens = useMerge ? Math.ceil(rawTokens / 4) : rawTokens;

    // VRAM Embedding size (BF16 = 2 bytes * 5120 dim per token)
    const embeddingBytes = finalTokens * 5120 * 2;
    const embeddingMb = (embeddingBytes / (1024 * 1024)).toFixed(2);

    // Context budget usage of 262,144 tokens
    const contextPct = ((finalTokens / 262144) * 100).toFixed(2);

    visCalcReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${finalTokens.toLocaleString()} 个词元</div>
          <div class="cs-lbl">${useMerge ? "摄取词元（4× 合并）" : "原始图像块词元"}</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${rawTokens.toLocaleString()}</div>
          <div class="cs-lbl">原始 14×14 图像块</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${embeddingMb} MB</div>
          <div class="cs-lbl">残差显存占用</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${contextPct}%</div>
          <div class="cs-lbl">占 262k 上下文预算</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        输入：<b>${p.name}</b> · 图像块：每帧 <b>${Math.ceil(p.w/14)} × ${Math.ceil(p.h/14)}</b> × ${p.frames} 帧 · 压缩：<b>${useMerge ? "4× 空间合并已启用 ✓" : "1:1 原始图像块"}</b>。
      </div>
    `;
  }

  [visPresetSel, visMergeToggle].forEach(el => {
    if (el) {
      el.addEventListener("change", renderVisionCalc);
      el.addEventListener("input", renderVisionCalc);
    }
  });
  renderVisionCalc();
})();

/* ============================================================
   Section 08: Multi-Token Prediction & Speculative Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Speculative Decoding Simulator (#w-mtp)
     ------------------------------------------------------------ */
  const row = $("mtpRow");
  if (!row) return;
  const out = $("mtpOut");
  const K_EL = $("mtpK");
  const DOMAIN_SEL = $("mtpDomainSel");
  const AUTO_BTN = $("mtpAuto");

  // §08's acceptance formula is the single speedup convention — regenerate every
  // "~N.N×" option label from it so the dropdown can never drift from §08's prose.
  DOMAIN_SEL?.querySelectorAll("option").forEach(o => {
    const b = parseFloat(o.value);
    if (Number.isFinite(b)) o.textContent = o.textContent.replace(/~\d+(\.\d+)?\s*×/, `~${mtpSpeedup(b, 2).toFixed(2)}×`);
  });

  const words = [
    "def ", "calculate_loss", "(", "self", ", ", "hidden_states", ", ", "labels", "):", "\n    ",
    "logits ", "= ", "self", ".", "lm_head", "(", "hidden_states", ")", "\n    ",
    "shift_logits ", "= ", "logits", "[", "...", ", ", ":-1", ", ", ":", "]", ".", "contiguous", "()", "\n    ",
    "shift_labels ", "= ", "labels", "[", "...", ", ", "1:", "]", ".", "contiguous", "()", "\n    ",
    "return ", "self", ".", "loss_fct", "(", "shift_logits", ", ", "shift_labels", ")"
  ];
  let pos = 0, timer = null, accepted = 0, proposed = 0, steps = 0, totalTokens = 0;

  function push(cls, text) {
    const t = document.createElement("div");
    t.className = "tok " + cls;
    t.textContent = text.replace("\n", "\\n").trim() || "·";
    if (text.includes("\n")) t.style.opacity = "0.75";
    row.appendChild(t);
  }

  function step() {
    const beta = DOMAIN_SEL ? +DOMAIN_SEL.value : 0.78;
    const K = K_EL ? +K_EL.value : 2;

    if (pos >= words.length) {
      stop();
      setTimeout(reset, 2400);
      return;
    }
    steps++;

    // Draft K continuations from MTP heads
    const drafts = [];
    for (let k = 0; k < K && pos + 1 + k < words.length; k++) {
      drafts.push({ ok: Math.random() < beta, word: words[pos + 1 + k] });
      proposed++;
    }

    // Root target token (always generated)
    push("cur", words[pos]);
    totalTokens++;

    // Verify drafts sequentially
    let acc = 0;
    while (acc < drafts.length && drafts[acc].ok) acc++;

    for (let k = 0; k < acc; k++) {
      push("acc", drafts[k].word);
      accepted++;
      totalTokens++;
    }

    if (acc < drafts.length) {
      // Rejection: target resamples the first disagreement token
      push("rej", drafts[acc].word);
      const correction = words[pos + 1 + acc] || "";
      push("cur", correction);
      totalTokens++;
      pos += acc + 2; // Jump past accepted drafts + rejected + correction
    } else {
      pos += acc + 1;
    }

    const effRate = mtpEtoks(beta, K);
    const speedup = (totalTokens / Math.max(1, steps)).toFixed(2);

    out.innerHTML = `
      <div class="row"><span>目标模型前向传播次数</span><span class="v">${steps} 次</span></div>
      <div class="row"><span>输出词元总数</span><span class="v a">${totalTokens} 个词元</span></div>
      <div class="row"><span>草稿接受率</span><span class="v m">${accepted}/${proposed} = ${(accepted / Math.max(1, proposed) * 100).toFixed(0)}%</span></div>
      <div class="row"><span>有效加速倍数</span><span class="v" style="color:var(--mint);font-size:13px">加速 <b>${speedup}×</b></span></div>
      <div class="row"><span>理论 E[词元/次] @ β=${beta.toFixed(2)}, K=${K}</span><span class="v i">${effRate.toFixed(2)} 词元/次</span></div>
      <div style="color:var(--faint);font-size:12px;margin-top:6px;line-height:1.4">
        <span style="color:var(--mint)">■ 绿色</span> = 已接受的 MTP 草稿 · <span style="color:var(--rose)">■ 红色</span> = 被拒绝的尾部 · <span style="color:var(--amber)">■ 琥珀色</span> = 目标根词元 / 重采样词元
      </div>
    `;

    if (pos >= words.length) {
      stop();
      setTimeout(reset, 2800);
    }
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    AUTO_BTN.textContent = "播放 \u25b6";
    AUTO_BTN.setAttribute("aria-label", "自动运行推测解码循环");
    AUTO_BTN.classList.remove("on");
  }

  function reset() {
    stop();
    row.innerHTML = "";
    pos = 0; steps = 0; accepted = 0; proposed = 0; totalTokens = 0;
    out.innerHTML = `<div class="row" style="border:0"><span style="color:var(--faint)">按<b style="color:var(--amber)">单步</b>或<b style="color:var(--amber)">播放</b>，开始 MTP 推测草拟</span><span class="v"></span></div>`;
  }

  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function play() {
    AUTO_BTN.textContent = "暂停 \u23f8";
    AUTO_BTN.setAttribute("aria-label", "暂停自动运行");
    AUTO_BTN.classList.add("on");
    timer = setInterval(step, 900);
  }
  $("mtpStep").addEventListener("click", step);
  AUTO_BTN.addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (REDUCED) { step(); return; }               // reduced motion: single step, no autoplay loop
    play();
  });
  let autoPausedByTab = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && timer) { stop(); autoPausedByTab = true; }
    else if (!document.hidden && autoPausedByTab && !timer) { autoPausedByTab = false; play(); }
  });
  $("mtpReset").addEventListener("click", reset);
  reset();


  /* ------------------------------------------------------------
     2. Widget: MTP Architectural Layout & Verification Tree (#w-mtp-tree)
     ------------------------------------------------------------ */
  const mtpTreeSvg = $("mtpTreeSvg");
  const mtpViewSeg = $("mtpViewSeg");
  const mtpTreeDepth = $("mtpTreeDepth");
  const mtpMaskTopologySeg = $("mtpMaskTopologySeg");
  const mtpMaskTableWrap = $("mtpMaskTableWrap");
  const mtpMaskExplainer = $("mtpMaskExplainer");
  const mtpVerifPromptText = $("mtpVerifPromptText");
  const mtpVerifStepBtn = $("mtpVerifStepBtn");
  const mtpVerifCandidates = $("mtpVerifCandidates");
  const mtpVerifComparison = $("mtpVerifComparison");
  const mtpStatWeight = $("mtpStatWeight");
  const mtpStatLatency = $("mtpStatLatency");
  const mtpStatVocab = $("mtpStatVocab");
  const mtpStatDist = $("mtpStatDist");

  // Dynamic Inspector elements
  const mtpInspectBadge = $("mtpInspectBadge");
  const mtpInspectTitle = $("mtpInspectTitle");
  const mtpInspectMath = $("mtpInspectMath");
  const mtpInspectDesc = $("mtpInspectDesc");
  const mtpInspectShape = $("mtpInspectShape");
  const mtpInspectDtype = $("mtpInspectDtype");

  let currentView = "arch";
  let depthK = 2;
  let selectedNode = "backbone";
  let maskTopo = "chain";
  let verifPresetIdx = 0;

  const MTP_NODES = {
    backbone: {
      badge: "目标主干",
      title: "主 Transformer 主干（第 64 层）",
      math: "h_t = RMSNorm(Block₆₄(h_t⁽⁶³⁾)) ∈ ℝ^(B × 1 × 5120)",
      desc: "经过全部 64 层（48 个 DeltaNet 层 + 16 个完整注意力层）生成最终的 5120 维隐藏状态 h_t。该激活向量是标准生成与推测分支扩展的基础根状态。",
      shape: "[批, 1, 5120]",
      dtype: "bfloat16（第 64 层）"
    },
    lm_head: {
      badge: "词表投影",
      title: "共享反嵌入 LM 头",
      math: "x_(t+1) ~ Softmax(W_u · RMSNorm(h_t)),  W_u ∈ ℝ^(248320 × 5120)",
      desc: "绑定或共享的反嵌入矩阵，将 5120 维隐藏状态映射为 248,320 个词表 logits。在 Qwen 3.8 中，所有 MTP 草稿头共享完全相同的 LM 头权重矩阵，避免 1.27B 重复参数（节省约 2.6 GB 显存）。",
      shape: "[248320, 5120]",
      dtype: "bfloat16（共享）"
    },
    head_1: {
      badge: "MTP 草稿头（K=1）",
      title: "MTP 头 1（深度 1 推测头）",
      math: "h̃_(t+1) = Block_MTP1([RMSNorm(h_t); Embed(x_(t+1))])",
      desc: "专用于预测提前 1 步词元的混合 Transformer 块（DeltaNet + RMSNorm）。它接收拼接后的主干表示和根词元嵌入，并投影为草稿词元 x̃_(t+2)。",
      shape: "[批, 1, 5120]",
      dtype: "bfloat16（约 1.3B 参数）"
    },
    head_2: {
      badge: "MTP 草稿头（K=2）",
      title: "MTP 头 2（深度 2 推测头）",
      math: "h̃_(t+2) = Block_MTP2([RMSNorm(h̃_(t+1)); Embed(x̃_(t+2))])",
      desc: "第二个推测头预测提前 2 步的词元 x̃_(t+3)。在 Qwen 3.8-27B 原生配置中，K=2 可同时提出 3 个候选词元，同时将参数开销限制在 +4.8%（约 2.6 GB）。",
      shape: "[批, 1, 5120]",
      dtype: "bfloat16（约 1.3B 参数）"
    },
    head_3: {
      badge: "MTP 草稿头（K=3）",
      title: "MTP 头 3（深度 3 推测头）",
      math: "h̃_(t+3) = Block_MTP3([RMSNorm(h̃_(t+2)); Embed(x̃_(t+3))])",
      desc: "扩展的深度 3 推测头，用于在高度可预测的领域（如结构化代码语法 / JSON 序列化）实现超高加速。提出候选 x̃_(t+4)。",
      shape: "[批, 1, 5120]",
      dtype: "bfloat16（约 1.3B 参数）"
    },
    verifier: {
      badge: "单次验证器",
      title: "目标主干并行树验证器",
      math: "P(x_(t+1), x̃_(t+2), x̃_(t+3) | x_(≤t)) = Forward_Target(Batch, M_tree)",
      desc: "使用二维因果树注意力掩码，对所有候选词元执行单次前向传播。验证延迟与标准单词元生成相同，因为内存带宽只需为整个候选批读取一次权重。",
      shape: "[批, K+1, 5120]",
      dtype: "bfloat16（目标权重）"
    }
  };

  const VERIF_PRESETS = [
    {
      domain: "结构化 Python AST",
      prompt: "def compute_attention_scores(q, k, v):",
      tokens: [
        { role: "根（t+1）", text: "\\n    ", prob: "98.4%", accept: true, reason: "确定的缩进" },
        { role: "草稿 1（t+2）", text: "scores = ", prob: "94.2%", accept: true, reason: "标准惯用写法" },
        { role: "草稿 2（t+3）", text: "torch.matmul(", prob: "91.7%", accept: true, reason: "匹配 AST 先验" },
        { role: "草稿 3（t+4）", text: "q, k.transpose(-1,-2))", prob: "88.5%", accept: true, reason: "张量矩阵乘参数" }
      ]
    },
    {
      domain: "TypeScript 类型注解",
      prompt: "type TransformerConfig = { hidden_size: ",
      tokens: [
        { role: "根（t+1）", text: "number", prob: "96.1%", accept: true, reason: "字段类型" },
        { role: "草稿 1（t+2）", text: "; num_heads: ", prob: "92.8%", accept: true, reason: "配置属性" },
        { role: "草稿 2（t+3）", text: "string", prob: "31.4%", accept: false, correction: "number", reason: "目标模型纠正为“number”" },
        { role: "草稿 3（t+4）", text: "; num_layers", prob: "0.0%", accept: false, pruned: true, reason: "分歧后剪枝" }
      ]
    },
    {
      domain: "SQL 聚合查询",
      prompt: "SELECT user_id, COUNT(id) AS total_orders",
      tokens: [
        { role: "根（t+1）", text: " FROM ", prob: "99.1%", accept: true, reason: "SQL 关键字" },
        { role: "草稿 1（t+2）", text: "orders ", prob: "95.6%", accept: true, reason: "源表" },
        { role: "草稿 2（t+3）", text: "GROUP BY ", prob: "93.0%", accept: true, reason: "聚合子句" },
        { role: "草稿 3（t+4）", text: "user_id;", prob: "94.8%", accept: true, reason: "分组键" }
      ]
    }
  ];

  function updateInspector(nodeKey) {
    selectedNode = nodeKey;
    const data = MTP_NODES[nodeKey] || MTP_NODES.backbone;
    if (mtpInspectBadge) mtpInspectBadge.textContent = data.badge;
    if (mtpInspectTitle) mtpInspectTitle.textContent = data.title;
    if (mtpInspectMath) mtpInspectMath.textContent = data.math;
    if (mtpInspectDesc) mtpInspectDesc.textContent = data.desc;
    if (mtpInspectShape) mtpInspectShape.textContent = data.shape;
    if (mtpInspectDtype) mtpInspectDtype.textContent = data.dtype;

    if (mtpTreeSvg) {
      mtpTreeSvg.querySelectorAll("g.clickable-node").forEach(g => {
        if (g.getAttribute("data-node") === nodeKey) {
          g.classList.add("selected");
        } else {
          g.classList.remove("selected");
        }
      });
    }
  }

  function renderMtpTreeSvg() {
    if (!mtpTreeSvg) return;
    const NS = "http://www.w3.org/2000/svg";
    mtpTreeSvg.innerHTML = "";

    const H = depthK === 3 ? 290 : 250;
    mtpTreeSvg.setAttribute("viewBox", `0 0 680 ${H}`);

    // Definitions
    const defs = document.createElementNS(NS, "defs");
    defs.innerHTML = `
      <marker id="mtpArrowHead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="rgba(26,22,18,0.45)" />
      </marker>
      <marker id="mtpArrowMint" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.mint}" />
      </marker>
      <marker id="mtpArrowAmber" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.amber}" />
      </marker>
      <marker id="mtpArrowIris" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.iris}" />
      </marker>
      <marker id="mtpArrowSky" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.sky}" />
      </marker>
    `;
    mtpTreeSvg.appendChild(defs);

    function drawNode(key, x, y, w, h, title, sub, sub2, fill, stroke, textFill) {
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "clickable-node" + (key === selectedNode ? " selected" : ""));
      g.setAttribute("data-node", key);
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", title);

      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", w); rect.setAttribute("height", h);
      rect.setAttribute("rx", "6");
      rect.setAttribute("fill", fill);
      rect.setAttribute("stroke", stroke);
      rect.setAttribute("stroke-width", "1.2");
      g.appendChild(rect);

      const t1 = document.createElementNS(NS, "text");
      t1.setAttribute("x", x + w / 2); t1.setAttribute("y", y + 17);
      t1.setAttribute("text-anchor", "middle");
      t1.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      t1.setAttribute("font-size", "10.5");
      t1.setAttribute("font-weight", "600");
      t1.setAttribute("fill", textFill || PAL.ink);
      t1.textContent = title;
      g.appendChild(t1);

      if (sub) {
        const t2 = document.createElementNS(NS, "text");
        t2.setAttribute("x", x + w / 2); t2.setAttribute("y", y + 32);
        t2.setAttribute("text-anchor", "middle");
        t2.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
        t2.setAttribute("font-size", "8.5");
        t2.setAttribute("fill", "rgba(26,22,18,0.65)");
        t2.textContent = sub;
        g.appendChild(t2);
      }

      if (sub2) {
        const t3 = document.createElementNS(NS, "text");
        t3.setAttribute("x", x + w / 2); t3.setAttribute("y", y + 46);
        t3.setAttribute("text-anchor", "middle");
        t3.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
        t3.setAttribute("font-size", "8");
        t3.setAttribute("font-weight", "600");
        t3.setAttribute("fill", textFill || PAL.ink);
        t3.textContent = sub2;
        g.appendChild(t3);
      }

      g.addEventListener("click", () => updateInspector(key));
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          updateInspector(key);
        }
      });
      mtpTreeSvg.appendChild(g);
      return { x, y, w, h };
    }

    function drawLine(x1, y1, x2, y2, color, markerId) {
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1);
      line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      line.setAttribute("stroke", color || "rgba(26,22,18,0.3)");
      line.setAttribute("stroke-width", "1.5");
      if (markerId) line.setAttribute("marker-end", `url(#${markerId})`);
      mtpTreeSvg.appendChild(line);
    }

    // 1. Target Backbone
    const backboneY = depthK === 1 ? 65 : depthK === 2 ? 100 : 120;
    const bb = drawNode("backbone", 20, backboneY, 130, 68, "主干网络", "第 64 层隐藏态（h_t）", "d=5120 · 27.58B", "rgba(184,52,31,0.1)", PAL.amber, PAL.amber);

    // 2. Shared LM Head
    const lm = drawNode("lm_head", 185, 18, 150, 52, "共享 LM 头", "Softmax(W_u · h_t)", "→ 根 x_{t+1}（248k）", "rgba(29,110,107,0.1)", PAL.mint, PAL.mint);

    // 3. MTP Head 1 (K=1)
    const h1 = drawNode("head_1", 185, 82, 150, 52, "MTP 头 1（K=1）", "RMSNorm + 混合块", "→ 草稿 x̃_{t+2}", "rgba(43,58,138,0.1)", PAL.iris, PAL.iris);

    // 4. MTP Head 2 (K=2)
    let h2 = null;
    if (depthK >= 2) {
      h2 = drawNode("head_2", 185, 146, 150, 52, "MTP 头 2（K=2）", "RMSNorm + 混合块", "→ 草稿 x̃_{t+3}", "rgba(186,117,23,0.1)", PAL.amber, PAL.amber);
    }

    // 5. MTP Head 3 (K=3)
    let h3 = null;
    if (depthK >= 3) {
      h3 = drawNode("head_3", 185, 210, 150, 52, "MTP 头 3（K=3）", "RMSNorm + 混合块", "→ 草稿 x̃_{t+4}", "rgba(43,122,158,0.1)", PAL.sky, PAL.sky);
    }

    // 6. Target Verifier Container
    const verifHeight = depthK === 1 ? 140 : depthK === 2 ? 195 : 245;
    const vG = document.createElementNS(NS, "g");
    vG.setAttribute("class", "clickable-node" + (selectedNode === "verifier" ? " selected" : ""));
    vG.setAttribute("data-node", "verifier");
    vG.setAttribute("tabindex", "0");
    vG.setAttribute("role", "button");
    vG.setAttribute("aria-label", "单次目标验证器");

    const vRect = document.createElementNS(NS, "rect");
    vRect.setAttribute("x", "380"); vRect.setAttribute("y", "18");
    vRect.setAttribute("width", "280"); vRect.setAttribute("height", verifHeight);
    vRect.setAttribute("rx", "6");
    vRect.setAttribute("fill", "rgba(26,22,18,0.03)");
    vRect.setAttribute("stroke", "rgba(26,22,18,0.22)");
    vRect.setAttribute("stroke-width", "1.2");
    vG.appendChild(vRect);

    const vt1 = document.createElementNS(NS, "text");
    vt1.setAttribute("x", "520"); vt1.setAttribute("y", "38");
    vt1.setAttribute("text-anchor", "middle");
    vt1.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    vt1.setAttribute("font-size", "10");
    vt1.setAttribute("font-weight", "700");
    vt1.setAttribute("letter-spacing", "0.05em");
    vt1.setAttribute("fill", PAL.ink);
    vt1.textContent = "单次目标验证器";
    vG.appendChild(vt1);

    const vt2 = document.createElementNS(NS, "text");
    vt2.setAttribute("x", "520"); vt2.setAttribute("y", "52");
    vt2.setAttribute("text-anchor", "middle");
    vt2.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    vt2.setAttribute("font-size", "8");
    vt2.setAttribute("fill", "rgba(26,22,18,0.5)");
    vt2.textContent = "1 次目标前向传播 · 二维树掩码";
    vG.appendChild(vt2);

    // Inner slots in Verifier
    const slots = [
      { label: "位置 0：根 x_{t+1}", color: PAL.mint },
      { label: "位置 1：草稿 x̃_{t+2}（头 1）", color: PAL.iris },
      ...(depthK >= 2 ? [{ label: "位置 2：草稿 x̃_{t+3}（头 2）", color: PAL.amber }] : []),
      ...(depthK >= 3 ? [{ label: "位置 3：草稿 x̃_{t+4}（头 3）", color: PAL.sky }] : [])
    ];

    slots.forEach((s, idx) => {
      const sy = 66 + idx * 36;
      const sr = document.createElementNS(NS, "rect");
      sr.setAttribute("x", "395"); sr.setAttribute("y", sy);
      sr.setAttribute("width", "250"); sr.setAttribute("height", "28");
      sr.setAttribute("rx", "4");
      sr.setAttribute("fill", "#fff");
      sr.setAttribute("stroke", s.color);
      sr.setAttribute("stroke-width", "1");
      vG.appendChild(sr);

      const st = document.createElementNS(NS, "text");
      st.setAttribute("x", "405"); st.setAttribute("y", sy + 18);
      st.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      st.setAttribute("font-size", "9");
      st.setAttribute("font-weight", "600");
      st.setAttribute("fill", s.color);
      st.textContent = s.label;
      vG.appendChild(st);

      const stScore = document.createElementNS(NS, "text");
      stScore.setAttribute("x", "635"); stScore.setAttribute("y", sy + 18);
      stScore.setAttribute("text-anchor", "end");
      stScore.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      stScore.setAttribute("font-size", "8");
      stScore.setAttribute("fill", "rgba(26,22,18,0.6)");
      stScore.textContent = `P(x|ctx) ✓`;
      vG.appendChild(stScore);
    });

    vG.addEventListener("click", () => updateInspector("verifier"));
    vG.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        updateInspector("verifier");
      }
    });
    mtpTreeSvg.appendChild(vG);

    // Connector arrows from Backbone
    const bbMidY = bb.y + bb.h / 2;
    drawLine(150, bbMidY, 185, 44, "rgba(26,22,18,0.3)", "mtpArrowHead");
    drawLine(150, bbMidY, 185, 108, "rgba(26,22,18,0.3)", "mtpArrowHead");
    if (depthK >= 2) drawLine(150, bbMidY, 185, 172, "rgba(26,22,18,0.3)", "mtpArrowHead");
    if (depthK >= 3) drawLine(150, bbMidY, 185, 236, "rgba(26,22,18,0.3)", "mtpArrowHead");

    // Connector arrows into Verifier
    drawLine(335, 44, 395, 80, PAL.mint, "mtpArrowMint");
    drawLine(335, 108, 395, 116, PAL.iris, "mtpArrowIris");
    if (depthK >= 2) drawLine(335, 172, 395, 152, PAL.amber, "mtpArrowAmber");
    if (depthK >= 3) drawLine(335, 236, 395, 188, PAL.sky, "mtpArrowSky");
  }

  function renderMtpMaskTable() {
    if (!mtpMaskTableWrap) return;

    let headers = [];
    let fullLabels = [];
    let matrix = [];
    let descriptions = [];

    if (maskTopo === "chain") {
      headers = [
        "前缀",
        "根",
        "草稿 1",
        ...(depthK >= 2 ? ["草稿 2"] : []),
        ...(depthK >= 3 ? ["草稿 3"] : [])
      ];
      fullLabels = [
        "前缀（x ≤ t）",
        "根（x_{t+1}）",
        "草稿 1（x̃_{t+2}）",
        ...(depthK >= 2 ? ["草稿 2（x̃_{t+3}）"] : []),
        ...(depthK >= 3 ? ["草稿 3（x̃_{t+4}）"] : [])
      ];

      const N = headers.length;
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          row.push(j <= i ? 1 : 0);
        }
        matrix.push(row);
      }

      descriptions = [
        "前缀词元严格关注先前的上下文历史。",
        "根词元 x_{t+1} 关注前缀历史和自身。",
        "草稿词元 x̃_{t+2} 关注前缀、根和自身（无法看到未来草稿）。",
        ...(depthK >= 2 ? ["草稿词元 x̃_{t+3} 关注前缀、根、草稿 1 和自身。"] : []),
        ...(depthK >= 3 ? ["草稿词元 x̃_{t+4} 关注完整的线性推测分支。"] : [])
      ];
    } else {
      // Branching Tree Beam 2
      headers = [
        "前缀",
        "根",
        "1A",
        "1B",
        "2A",
        "2B"
      ];
      fullLabels = [
        "前缀（x ≤ t）",
        "根（x_{t+1}）",
        "草稿 1A（分支 A）",
        "草稿 1B（分支 B）",
        "草稿 2A（分支 A）",
        "草稿 2B（分支 B）"
      ];
      matrix = [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0], // Sibling 1B sees Prefix+Root, but NOT 1A!
        [1, 1, 1, 0, 1, 0], // 2A descends from 1A
        [1, 1, 0, 1, 0, 1]  // 2B descends from 1B
      ];
      descriptions = [
        "前缀词元关注先前的上下文历史。",
        "根词元关注前缀历史和自身。",
        "分支 A（深度 1）关注前缀、根和自身。",
        "分支 B（深度 1）关注前缀和根，但不能关注分支 A（独立假设）。",
        "分支 A（深度 2）关注前缀、根、草稿 1A 和自身。",
        "分支 B（深度 2）关注前缀、根、草稿 1B 和自身（与分支 A 完全隔离）。"
      ];
    }

    let html = `<table class="mtp-mask-table"><thead><tr><th>查询 ↓ \\ 键 →</th>`;
    headers.forEach((h, idx) => {
      html += `<th title="${fullLabels[idx]}">${h}</th>`;
    });
    html += `</tr></thead><tbody>`;

    matrix.forEach((row, rIdx) => {
      html += `<tr data-row="${rIdx}"><th style="text-align:left" title="${fullLabels[rIdx]}">${headers[rIdx]}</th>`;
      row.forEach((val, cIdx) => {
        const isAllowed = val === 1;
        const cls = isAllowed ? "allowed" : "masked";
        const txt = isAllowed ? "1" : "0";
        html += `<td class="mtp-mask-cell ${cls}" data-row="${rIdx}" data-col="${cIdx}" title="查询：${fullLabels[rIdx]} → 键：${fullLabels[cIdx]}：${isAllowed ? '允许（0）' : '已掩码（-∞）'}">${txt}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;

    mtpMaskTableWrap.innerHTML = html;

    // Add interactivity to mask cells and rows
    mtpMaskTableWrap.querySelectorAll(".mtp-mask-cell").forEach(cell => {
      cell.addEventListener("mouseenter", () => {
        const r = parseInt(cell.getAttribute("data-row"), 10);
        const c = parseInt(cell.getAttribute("data-col"), 10);
        const isAllowed = cell.classList.contains("allowed");
        if (mtpMaskExplainer) {
          mtpMaskExplainer.innerHTML = `<strong>查询：</strong>${fullLabels[r]} &nbsp;→&nbsp; <strong>键：</strong>${fullLabels[c]}：<span style="color:${isAllowed ? 'var(--mint)' : 'var(--amber)'}">${isAllowed ? '✓ 允许关注（0 dB）' : '✕ 已掩码（-∞ dB）'}</span>。${descriptions[r]}`;
        }
      });
      cell.addEventListener("mouseleave", () => {
        if (mtpMaskExplainer) {
          mtpMaskExplainer.textContent = "将鼠标悬停在矩阵中的任意单元格或词元行上，即可查看其有效注意力感受野。";
        }
      });
    });
  }

  function renderMtpVerifCycle() {
    if (!mtpVerifCandidates || !mtpVerifComparison) return;

    const preset = VERIF_PRESETS[verifPresetIdx % VERIF_PRESETS.length];
    if (mtpVerifPromptText) {
      mtpVerifPromptText.innerHTML = `<code>${preset.prompt}</code> <span style="font-size:12px;color:var(--faint);margin-left:6px">(${preset.domain})</span>`;
    }

    const visibleTokens = preset.tokens.slice(0, depthK + 1);
    let acceptedCount = 0;
    let diverged = false;

    let candHtml = "";
    visibleTokens.forEach((t, idx) => {
      let isAcc = t.accept && !diverged;
      if (isAcc) acceptedCount++;
      else if (!diverged) diverged = true;

      const cls = isAcc ? "accepted" : "rejected";
      const statusText = isAcc ? "✓ 已接受" : (t.pruned ? "— 已剪枝" : `✕ 已重采样（${t.correction || '目标'}）`);
      const statusCls = isAcc ? "accept" : "reject";

      candHtml += `
        <div class="verif-candidate-row ${cls}">
          <span class="verif-role-tag">${t.role}</span>
          <span class="verif-token-text"><code>"${t.text}"</code></span>
          <span class="verif-prob-val">P = ${t.prob}</span>
          <span class="verif-status-badge ${statusCls}">${statusText}</span>
        </div>
      `;
    });
    mtpVerifCandidates.innerHTML = candHtml;

    // Comparison summary
    const totalTokensEvaluated = depthK + 1;
    const seqLatencyMs = (totalTokensEvaluated * 12.0).toFixed(1);
    const mtpLatencyMs = (12.6).toFixed(1);
    const speedup = (seqLatencyMs / mtpLatencyMs).toFixed(2);
    const acceptedTokensTotal = diverged ? acceptedCount + 1 : acceptedCount; // includes corrected target token if resampled

    mtpVerifComparison.innerHTML = `
      <div class="verif-cmp-card">
        <div class="cmp-title">顺序生成（${totalTokensEvaluated} 次前向传播）</div>
        <div class="cmp-stat">${seqLatencyMs} ms <span style="font-size:12px;color:var(--muted)">（${totalTokensEvaluated} × 内存加载）</span></div>
      </div>
      <div class="verif-cmp-card">
        <div class="cmp-title">MTP 单次树（接受 ${acceptedTokensTotal} 个词元）</div>
        <div class="cmp-stat mint">${mtpLatencyMs} ms <span style="font-size:12px;color:var(--mint);font-weight:600">（快 ${speedup}×）</span></div>
      </div>
    `;
  }

  function updateMtpDepth(newK) {
    depthK = newK;
    if (mtpStatWeight) {
      mtpStatWeight.textContent = newK === 1 ? "+2.4% (~1.3 GB)" : newK === 2 ? "+4.8% (~2.6 GB)" : "+7.2% (~3.9 GB)";
    }
    renderMtpTreeSvg();
    renderMtpMaskTable();
    renderMtpVerifCycle();
  }

  function setupMtpTreeWidget() {
    // View tabs
    if (mtpViewSeg) {
      mtpViewSeg.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          mtpViewSeg.querySelectorAll("button").forEach(b => {
            b.classList.remove("on");
            b.setAttribute("aria-pressed", "false");
          });
          btn.classList.add("on");
          btn.setAttribute("aria-pressed", "true");

          currentView = btn.getAttribute("data-view");
          const panelArch = $("mtpPanelArch");
          const panelMask = $("mtpPanelMask");
          const panelVerif = $("mtpPanelVerif");

          if (panelArch) panelArch.style.display = currentView === "arch" ? "block" : "none";
          if (panelMask) panelMask.style.display = currentView === "mask" ? "block" : "none";
          if (panelVerif) panelVerif.style.display = currentView === "verif" ? "block" : "none";
        });
      });
    }

    // Depth selector
    if (mtpTreeDepth) {
      mtpTreeDepth.addEventListener("change", (e) => {
        updateMtpDepth(parseInt(e.target.value, 10));
      });
    }

    // Mask topology toggle
    if (mtpMaskTopologySeg) {
      mtpMaskTopologySeg.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          mtpMaskTopologySeg.querySelectorAll("button").forEach(b => {
            b.classList.remove("on");
            b.setAttribute("aria-pressed", "false");
          });
          btn.classList.add("on");
          btn.setAttribute("aria-pressed", "true");
          maskTopo = btn.getAttribute("data-topo");
          renderMtpMaskTable();
        });
      });
    }

    // Verif cycle step button
    if (mtpVerifStepBtn) {
      mtpVerifStepBtn.addEventListener("click", () => {
        verifPresetIdx = (verifPresetIdx + 1) % VERIF_PRESETS.length;
        renderMtpVerifCycle();
      });
    }

    // Initial renders
    updateInspector("backbone");
    renderMtpTreeSvg();
    renderMtpMaskTable();
    renderMtpVerifCycle();
  }

  setupMtpTreeWidget();
})();

/* ============================================================
   Section 09: Data Pipeline & Synthesis Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 5-Stage Ingestion Funnel (#w-data-funnel)
     ------------------------------------------------------------ */
  const funnelPipe = $("funnelPipeline");
  const funnelDetailCard = $("funnelDetailCard");

  const FUNNEL_STAGES = {
    "1": {
      name: "阶段 01 · 原始数据采集与网页抓取",
      volIn: "120+ PB 原始 WARC / HTML / PDF / Git 仓库",
      volOut: "45 PB 规范化原始文档",
      retention: "数据量保留率 37.5%",
      heuristics: "去除网页样板内容、提取 HTML 标签、移除广告、验证 MIME 类型并进行 UTF-8 规范化。",
      details: "汇集海量公开网页抓取数据（Common Crawl、精选域名）、ArXiv LaTeX 源文件、公开 GitHub 代码和百科档案。语言分类器识别出 119 种候选语言。"
    },
    "2": {
      name: "阶段 02 · 多模态布局感知 OCR",
      volIn: "45 PB 未解析的 PDF、图表和数学论文扫描件",
      volOut: "18 PB 含 LaTeX 公式的结构化 Markdown",
      retention: "解析文本保留率 40.0%",
      heuristics: "Qwen2.5-VL 布局分割、边界框对齐、双栏重排，以及表格转 Markdown。",
      details: "标准文本解析器会破坏复杂的 STEM 文档、表格和数学证明。经过微调的视觉语言模型将多栏布局解析为整洁、线性的 Markdown。"
    },
    "3": {
      name: "阶段 03 · 质量过滤、去重与去污染",
      volIn: "18 PB 结构化文本（约 48 万亿候选词元）",
      volOut: "22 万亿高密度自然词元",
      retention: "词元保留率 45.8%",
      heuristics: "MinHash-LSH 近重复去重、困惑度分类器集成、重复惩罚和 13-gram 去污染。",
      details: "过滤低熵机器垃圾、SEO 垃圾内容和有害内容。通过 13-gram 精确匹配扫描，永久移除与 GSM8K、MATH、HumanEval 和 SWE-bench 基准重叠的内容。"
    },
    "4": {
      name: "阶段 04 · 模型闭环验证器合成",
      volIn: "22 万亿自然基础词元",
      volOut: "+14 万亿经验证的合成词元（总计 36T）",
      retention: "通过合成扩充数据量 +63.6%",
      heuristics: "自动验证判定器：SymPy 计算机代数系统、Lean 4 证明器和容器化的单元测试执行。",
      details: "Qwen2.5-Math 和 Qwen2.5-Coder 合成教材中缺失的推导、算法解法和多轮智能体工具轨迹。只有验证正确的输出才会进入预训练分片。"
    },
    "5": {
      name: "阶段 05 · 最佳适配序列打包与分片",
      volIn: "36 万亿经验证的干净词元",
      volOut: "36 万亿词元，打包为 4,096 与 262,144 长度的分片",
      retention: "序列装箱填充率 98.4%",
      heuristics: "最佳适配装箱、注意力掩码边界重置、位置 ID 重置和确定性分片混洗。",
      details: "在大规模分布式预训练期间消除填充词元浪费，同时严格防止跨文档边界的注意力泄漏。"
    }
  };

  function renderFunnelStage(stepNum) {
    if (!funnelDetailCard) return;
    const info = FUNNEL_STAGES[stepNum] || FUNNEL_STAGES["1"];

    funnelDetailCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:12px;color:var(--sky);font-weight:600">${info.retention}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px;font-family:var(--mono);font-size:12px">
        <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line)">
          <span style="color:var(--faint);text-transform:uppercase;font-size:12px;display:block;margin-bottom:2px">输入数据量</span>
          <span style="color:var(--fg);font-weight:600">${info.volIn}</span>
        </div>
        <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line)">
          <span style="color:var(--faint);text-transform:uppercase;font-size:12px;display:block;margin-bottom:2px">输出分片产量</span>
          <span style="color:var(--mint);font-weight:600">${info.volOut}</span>
        </div>
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        <b>流水线策略：</b> ${info.details}
      </div>
      <div style="font-size:12px;color:var(--faint);font-family:var(--mono)">
        <b>过滤规则：</b> ${info.heuristics}
      </div>
    `;
  }

  if (funnelPipe) {
    funnelPipe.querySelectorAll(".funnel-step").forEach(btn => {
      btn.addEventListener("click", () => {
        funnelPipe.querySelectorAll(".funnel-step").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        renderFunnelStage(btn.dataset.step);
      });
    });
    enableTabs(funnelPipe, ".funnel-step");
  }
  renderFunnelStage("1");


  /* ------------------------------------------------------------
     2. Widget: Dynamic Corpus Mix Visualizer (#w-data-mix)
     ------------------------------------------------------------ */
  const mixStageSeg = $("mixStageSeg");
  const msbWeb = $("msbWeb");
  const msbCode = $("msbCode");
  const msbStem = $("msbStem");
  const msbMulti = $("msbMulti");
  const mixStats = $("mixStats");

  const STAGE_MIXES = {
    s1: {
      name: "阶段 1 · 通用基础混合（30 万亿词元）",
      web: 35, code: 30, stem: 20, multi: 15,
      tokens: { web: "10.5T", code: "9.0T", stem: "6.0T", multi: "4.5T" },
      desc: "均衡分布，在 119 种自然语言中建立广泛的语言基础、句法掌握能力和基础世界知识。"
    },
    s2: {
      name: "阶段 2 · 推理与 STEM 混合（+5 万亿词元）",
      web: 15, code: 35, stem: 40, multi: 10,
      tokens: { web: "0.75T", code: "1.75T", stem: "2.0T", multi: "0.5T" },
      desc: "大幅提高经验证的合成数学数据（由 SymPy 检查）、竞赛编程和形式化算法推理的权重。"
    },
    s3: {
      name: "阶段 3 · 262k 长上下文混合（约 1 万亿词元）",
      web: 25, code: 45, stem: 20, multi: 10,
      tokens: { web: "0.25T", code: "0.45T", stem: "0.20T", multi: "0.10T" },
      desc: "完整仓库级代码库、多文档综合和长程大海捞针任务，逐步退火至 262k 词元。"
    }
  };

  function renderCorpusMix(stageKey) {
    if (!mixStats) return;
    const mix = STAGE_MIXES[stageKey] || STAGE_MIXES.s1;

    stackSegments(
      [[msbWeb, mix.web], [msbCode, mix.code], [msbStem, mix.stem], [msbMulti, mix.multi]]
    );

    mixStats.innerHTML = `
      <div style="font-family:var(--display);font-size:14px;font-weight:700;color:var(--fg);margin-bottom:6px">${mix.name}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:8px;margin-bottom:8px;font-family:var(--mono);font-size:12px">
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--sky)">网页与图书：</span> <b>${mix.web}% (${mix.tokens.web})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--mint)">代码与仓库：</span> <b>${mix.code}% (${mix.tokens.code})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--amber)">STEM 与数学：</span> <b>${mix.stem}% (${mix.tokens.stem})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--iris)">多语言：</span> <b>${mix.multi}% (${mix.tokens.multi})</b>
        </div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45">${mix.desc}</div>
    `;
  }

  if (mixStageSeg) {
    bindGroup(mixStageSeg, b => renderCorpusMix(b.dataset.stage));
  }
  renderCorpusMix("s1");


  /* ------------------------------------------------------------
     3. Widget: Sequence Packing & Mask Inspector (#w-data-packing)
     ------------------------------------------------------------ */
  const packingVisual = $("packingVisual");
  const packingModeSeg = $("packingModeSeg");
  let packingMode = "isolated";

  function renderPacking() {
    if (!packingVisual) return;

    if (packingMode === "isolated") {
      packingVisual.innerHTML = `
        <div class="pv-seq-track">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:12px;color:var(--muted)">
            <span>打包后的 4,096 词元训练序列（3 篇文档）</span>
            <span style="color:var(--mint);font-weight:600">填充率：99.8% · 8 个填充词元</span>
          </div>
          <div class="pv-track-bar">
            <div class="pv-doc d1" style="flex:1850">
              <span class="doc-lbl">文档 1：Python AST</span>
              <span class="pos-lbl">1,850 词元 · 位置 0–1,849 · 掩码 [1..1850]</span>
            </div>
            <div class="pv-doc d2" style="flex:1420">
              <span class="doc-lbl">文档 2：Lean 4 证明</span>
              <span class="pos-lbl">1,420 词元 · 位置 0–1,419 · 掩码已重置 ✓</span>
            </div>
            <div class="pv-doc d3" style="flex:818">
              <span class="doc-lbl">文档 3：多语言</span>
              <span class="pos-lbl">818 词元 · 位置 0–817 · 掩码已重置 ✓</span>
            </div>
          </div>
        </div>
        <div class="pv-stats-grid">
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">注意力 FLOPs</span>
            <span class="pv-stat-val" style="color:var(--mint)">6.13M（块对角：−63.5%）</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">FlashAttention 变长模式</span>
            <span class="pv-stat-val" style="color:var(--sky)">cu_seqlens = [0, 1850, 3270, 4088]</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">RoPE 位置范围</span>
            <span class="pv-stat-val" style="color:var(--fg)">在 EOS 处严格重置为位置 0</span>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--mint);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-top:2px solid var(--mint)">
          ✓ <b>正确的掩码隔离已启用：</b> 注意力矩阵通过 FlashAttention 变长路径阻断跨文档注意力。每个 EOS 分隔符处的位置 ID 都重置为 0，防止不相关文章之间发生 RoPE 位置污染。
        </div>
      `;
    } else {
      packingVisual.innerHTML = `
        <div class="pv-seq-track">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:12px;color:var(--rose)">
            <span>朴素拼接（忽略注意力掩码）</span>
            <span style="font-weight:600">跨文档污染缺陷 ✗</span>
          </div>
          <div class="pv-track-bar">
            <div class="pv-doc d1 bleed" style="flex:1850">
              <span class="doc-lbl">文档 1：Python AST</span>
              <span class="pos-lbl">1,850 词元 · 位置 0–1,849</span>
            </div>
            <div class="pv-doc d2 bleed" style="flex:1420">
              <span class="doc-lbl">文档 2：Lean 4 ⚠️</span>
              <span class="pos-lbl">1,420 词元 · 关注文档 1</span>
            </div>
            <div class="pv-doc d3 bleed" style="flex:818">
              <span class="doc-lbl">文档 3：多语言 ⚠️</span>
              <span class="pos-lbl">818 词元 · 关注文档 1 和 2</span>
            </div>
          </div>
        </div>
        <div class="pv-stats-grid">
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">注意力 FLOPs</span>
            <span class="pv-stat-val" style="color:var(--rose)">16.78M（完整 4k：浪费 +174%）</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">注意力泄漏</span>
            <span class="pv-stat-val" style="color:var(--rose)">未掩码（跨文档渗漏）</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">RoPE 位置范围</span>
            <span class="pv-stat-val" style="color:var(--rose)">位置 0 → 4,087 的伪漂移</span>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--rose);line-height:1.5;background:rgba(138,51,36,0.06);padding:8px 12px;border-radius:6px;border-top:2px solid var(--rose)">
          ✗ <b>注意力泄漏缺陷：</b> 若不重置注意力掩码，文档 2 和文档 3 会直接关注先前文档的词元。关联记忆写入会被无关文档历史污染，RoPE 频率也会漂移并超出文档的真实长度。
        </div>
      `;
    }
  }

  if (packingModeSeg) {
    bindGroup(packingModeSeg, b => { packingMode = b.dataset.mode; renderPacking(); });
  }
  renderPacking();
})();

/* ============================================================
   Section 10: Pretraining & Scaling Curriculum Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Interactive 3-Stage Pretraining Curriculum Tabs
     ------------------------------------------------------------ */
  const curr = $("pretrainCurriculum");
  if (curr) {
    curr.querySelectorAll(".cur-item").forEach(item => {
      item.addEventListener("click", () => {
        curr.querySelectorAll(".cur-item").forEach(c => c.classList.remove("active"));
        item.classList.add("active");
      });
    });
    enableTabs(curr, ".cur-item");
  }

  /* ------------------------------------------------------------
     2. Widget: Pretraining Loss Trajectory & Milestone Explorer (#w-pretrain-loss)
     ------------------------------------------------------------ */
  const lossSvg = $("lossSvg");
  const lossStats = $("lossStats");
  const lossTokenSlider = $("lossTokenSlider");
  const lossCurveToggle = $("lossCurveToggle");

  function renderLossTrajectory() {
    if (!lossSvg || !lossStats) return;
    const curTokens = +lossTokenSlider.value || 30;
    const trackMode = lossCurveToggle.value || "all";

    const NS = "http://www.w3.org/2000/svg";
    lossSvg.innerHTML = "";

    const W = 640, H = 260, P = { l: 48, r: 24, t: 24, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // X: 0 to 36T tokens, Y: Loss 0.8 to 3.5
    const fx = t => P.l + (t / 36) * IW;
    const fy = l => P.t + (1 - (l - 0.8) / (3.5 - 0.8)) * IH;

    // Grid lines & Axis labels
    [0, 10, 20, 30, 36].forEach(tVal => {
      const gx = fx(tVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      lossSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `${tVal}T`;
      lossSvg.appendChild(txt);
    });

    [1.0, 1.5, 2.0, 2.5, 3.0].forEach(lVal => {
      const gy = fy(lVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      lossSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = lVal.toFixed(1);
      lossSvg.appendChild(txt);
    });

    // Stage boundary markers
    const s2X = fx(30);
    const s2Line = document.createElementNS(NS, "line");
    s2Line.setAttribute("x1", s2X); s2Line.setAttribute("y1", P.t);
    s2Line.setAttribute("x2", s2X); s2Line.setAttribute("y2", P.t + IH);
    s2Line.setAttribute("stroke", "rgba(184,52,31,0.3)");
    s2Line.setAttribute("stroke-dasharray", "4 3");
    lossSvg.appendChild(s2Line);

    const s2Lbl = document.createElementNS(NS, "text");
    s2Lbl.setAttribute("x", s2X + 4); s2Lbl.setAttribute("y", P.t + 14);
    s2Lbl.setAttribute("fill", PAL.amber);
    s2Lbl.setAttribute("font-size", "9.5");
    s2Lbl.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    s2Lbl.textContent = "阶段 2（STEM 重启）";
    lossSvg.appendChild(s2Lbl);

    // Compute Loss curves
    // 1. General language loss
    function getGenLoss(t) {
      return 1.42 + 1.65 * Math.pow(t + 0.5, -0.18);
    }
    // 2. STEM Math/Code loss
    function getStemLoss(t) {
      if (t < 30) {
        return 1.85 + 1.3 * Math.pow(t + 0.5, -0.12);
      } else {
        const dt = t - 30;
        return 2.15 * Math.exp(-dt * 0.22) + 1.15;
      }
    }
    // 3. MTP Aux Loss
    function getMtpLoss(t) {
      return 1.65 + 1.4 * Math.pow(t + 0.5, -0.15);
    }

    let dGen = "", dStem = "", dMtp = "";
    for (let t = 0; t <= 36.001; t += 0.25) {
      const gx = fx(t);
      dGen += (t === 0 ? "M" : "L") + gx + "," + fy(getGenLoss(t)) + " ";
      dStem += (t === 0 ? "M" : "L") + gx + "," + fy(getStemLoss(t)) + " ";
      dMtp += (t === 0 ? "M" : "L") + gx + "," + fy(getMtpLoss(t)) + " ";
    }

    if (trackMode === "all" || trackMode === "general") {
      const pGen = document.createElementNS(NS, "path");
      pGen.setAttribute("d", dGen); pGen.setAttribute("fill", "none");
      pGen.setAttribute("stroke", PAL.mint); pGen.setAttribute("stroke-width", "2.2");
      lossSvg.appendChild(pGen);
    }

    if (trackMode === "all" || trackMode === "stem") {
      const pStem = document.createElementNS(NS, "path");
      pStem.setAttribute("d", dStem); pStem.setAttribute("fill", "none");
      pStem.setAttribute("stroke", PAL.amber); pStem.setAttribute("stroke-width", "2.2");
      lossSvg.appendChild(pStem);
    }

    if (trackMode === "all" || trackMode === "mtp") {
      const pMtp = document.createElementNS(NS, "path");
      pMtp.setAttribute("d", dMtp); pMtp.setAttribute("fill", "none");
      pMtp.setAttribute("stroke", PAL.iris); pMtp.setAttribute("stroke-width", "1.8");
      pMtp.setAttribute("stroke-dasharray", "4 3");
      lossSvg.appendChild(pMtp);
    }

    // Scrubber Marker
    const curX = fx(curTokens);
    const sLine = document.createElementNS(NS, "line");
    sLine.setAttribute("x1", curX); sLine.setAttribute("y1", P.t);
    sLine.setAttribute("x2", curX); sLine.setAttribute("y2", P.t + IH);
    sLine.setAttribute("stroke", PAL.rose);
    sLine.setAttribute("stroke-width", "2");
    lossSvg.appendChild(sLine);

    const activeLoss = getGenLoss(curTokens);
    const activeStem = getStemLoss(curTokens);

    const sDot = document.createElementNS(NS, "circle");
    sDot.setAttribute("cx", curX); sDot.setAttribute("cy", fy(activeLoss));
    sDot.setAttribute("r", "5.5");
    sDot.setAttribute("fill", PAL.rose);
    sDot.setAttribute("stroke", PAL.ink);
    sDot.setAttribute("stroke-width", "2");
    lossSvg.appendChild(sDot);

    const stageName = curTokens < 30 ? "阶段 1（通用基础）" : (curTokens < 35 ? "阶段 2（推理与 STEM）" : "阶段 3（262k 长上下文）");
    const pct = ((curTokens / 36) * 100).toFixed(1);

    lossStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">课程里程碑 @ ${curTokens.toFixed(1)}T 词元（${pct}%）</div>
        <div class="rf-row"><span>当前阶段</span><span class="val amber" style="font-size:12px">${stageName}</span></div>
        <div class="rf-row"><span>通用交叉熵损失</span><span class="val mint">${activeLoss.toFixed(3)}（PPL 约 ${Math.exp(activeLoss).toFixed(2)}）</span></div>
        <div class="rf-row"><span>STEM / 代码损失</span><span class="val">${activeStem.toFixed(3)}</span></div>
        <div class="rf-row"><span>MTP 辅助损失</span><span class="val" style="color:var(--iris)">${getMtpLoss(curTokens).toFixed(3)}</span></div>
      </div>
      <div class="rf-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:12px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">里程碑解读</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          ${curTokens < 30
            ? "阶段 1 持续压缩自然语言语法、事实性世界知识，以及覆盖 119 种语言的广泛多语言表征。"
            : (curTokens < 35
              ? "阶段 2 重启学习率（LR），在不损害阶段 1 核心表征的前提下，迅速提升推理和竞赛编程基准表现。"
              : "阶段 3 对分块 WY 循环扫描核和 YaRN RoPE 缩放进行退火，固化完整的 262k 上下文长度能力。")}
        </div>
      </div>
    `;
  }

  [lossTokenSlider, lossCurveToggle].forEach(el => {
    if (el) {
      el.addEventListener("input", renderLossTrajectory);
      el.addEventListener("change", renderLossTrajectory);
    }
  });
  renderLossTrajectory();
  if (lossTokenSlider?.type === "range") {
    const vt = () => lossTokenSlider.setAttribute("aria-valuetext", `${(+lossTokenSlider.value).toFixed(1)} 万亿词元`);
    lossTokenSlider.addEventListener("input", vt); vt();
  }


  /* ------------------------------------------------------------
     3. Widget: Cluster Compute & Supercomputing Budget Calculator (#w-pretrain-calc)
     ------------------------------------------------------------ */
  const calcReadout = $("calcReadout");
  const calcTokens = $("calcTokens");
  const calcGpuType = $("calcGpuType");
  const calcGpuCount = $("calcGpuCount");
  const calcMfu = $("calcMfu");
  const GPU_SPECS = {
    h100: { name: "NVIDIA H100 SXM5", tflops: 989, watts: 700 },
    b200: { name: "NVIDIA B200 NVL", tflops: 2250, watts: 1000 },
    h200: { name: "NVIDIA H200 SXM5", tflops: 989, watts: 700 },
    a100: { name: "NVIDIA A100 SXM4", tflops: 312,  watts: 400 }
  };

  function renderComputeCalc() {
    if (!calcReadout) return;
    const tokensT = +calcTokens.value || 36;
    const gpuKey = calcGpuType.value || "h100";
    const gpuCount = +calcGpuCount.value || 4096;
    const mfu = +calcMfu.value || 0.42;

    const gpu = GPU_SPECS[gpuKey];
    const totalTokens = tokensT * 1e12;
    const N = 27.58e9; // 27.58B parameters

    // Chinchilla compute FLOPs: 6 * N * D
    const totalFlops = 6 * N * totalTokens;
    const yottaFlops = totalFlops / 1e24;
    const exaFlopDays = totalFlops / (1e18 * 86400);

    // Cluster sustained compute in FLOPs/s
    const clusterFlopsPerSec = gpuCount * (gpu.tflops * 1e12) * mfu;
    const totalSeconds = totalFlops / clusterFlopsPerSec;
    const totalDays = totalSeconds / 86400;

    // Power & Energy
    const pue = 1.25;
    const clusterKw = (gpuCount * gpu.watts * pue) / 1000;
    const clusterMw = clusterKw / 1000;
    const totalMWh = (clusterMw * totalDays * 24);

    calcReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num">${yottaFlops.toFixed(2)} YFLOPs</div>
          <div class="cs-lbl">总计算量</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${exaFlopDays.toFixed(0)} EFD</div>
          <div class="cs-lbl">ExaFLOP-天</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${totalDays.toFixed(1)} 天</div>
          <div class="cs-lbl">实际耗时</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${totalMWh.toLocaleString(undefined, {maximumFractionDigits: 0})} MWh</div>
          <div class="cs-lbl">能耗（PUE 1.25）</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        集群吞吐量：${gpuCount.toLocaleString()} × ${gpu.name}、MFU ${(mfu * 100).toFixed(0)}% 时，<b>持续 ${(clusterFlopsPerSec / 1e18).toFixed(1)} ExaFLOPs/s</b> · 功耗：<b>${clusterMw.toFixed(2)} MW</b>。
      </div>
    `;
  }

  [calcTokens, calcGpuType, calcGpuCount, calcMfu].forEach(el => {
    if (el) {
      el.addEventListener("change", renderComputeCalc);
      el.addEventListener("input", renderComputeCalc);
    }
  });
  renderComputeCalc();


  /* ------------------------------------------------------------
     4. Widget: Optimizer Dynamics & Beta2 Inspector (#w-optim-dyn)
     ------------------------------------------------------------ */
  const optimSvg = $("optimSvg");
  const optimStats = $("optimStats");
  const optimBetaSeg = $("optimBetaSeg");
  let curBeta2 = 0.95;

  function renderOptimDynamics() {
    if (!optimSvg || !optimStats) return;

    const NS = "http://www.w3.org/2000/svg";
    optimSvg.innerHTML = "";

    const W = 560, H = 220, P = { l: 48, r: 24, t: 20, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Steps 0 to 100
    const fx = step => P.l + (step / 100) * IW;
    const fy = scale => P.t + (1 - scale) * IH;

    // Grid lines
    [0, 25, 50, 75, 100].forEach(sVal => {
      const gx = fx(sVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      optimSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `步骤 ${sVal}`;
      optimSvg.appendChild(txt);
    });

    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(scaleVal => {
      const gy = fy(scaleVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      optimSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = `${(scaleVal * 100).toFixed(0)}%`;
      optimSvg.appendChild(txt);
    });

    // Simulate 10x gradient spike at step 30
    let v_t = 1.0;
    let dPath = "";
    const halfLifeSteps = curBeta2 === 0.95 ? 14 : 693;

    for (let step = 0; step <= 100; step++) {
      let g_t = 1.0;
      if (step === 30) g_t = 10.0; // Gradient shock
      v_t = curBeta2 * v_t + (1 - curBeta2) * (g_t * g_t);
      const effectiveStepScale = Math.min(1.0, 1.0 / Math.sqrt(v_t));

      const gx = fx(step);
      const gy = fy(effectiveStepScale);
      dPath += (step === 0 ? "M" : "L") + gx + "," + gy + " ";
    }

    const pathEl = document.createElementNS(NS, "path");
    pathEl.setAttribute("d", dPath);
    pathEl.setAttribute("fill", "none");
    pathEl.setAttribute("stroke", curBeta2 === 0.95 ? PAL.mint : PAL.rose);
    pathEl.setAttribute("stroke-width", "2.5");
    optimSvg.appendChild(pathEl);

    // Shock annotation
    const shockX = fx(30);
    const sLine = document.createElementNS(NS, "line");
    sLine.setAttribute("x1", shockX); sLine.setAttribute("y1", P.t);
    sLine.setAttribute("x2", shockX); sLine.setAttribute("y2", P.t + IH);
    sLine.setAttribute("stroke", "rgba(138,51,36,0.4)");
    sLine.setAttribute("stroke-dasharray", "3 3");
    optimSvg.appendChild(sLine);

    optimStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">优化器二阶矩记忆</div>
        <div class="rf-row"><span>所选 β₂ 参数</span><span class="val amber">${curBeta2}</span></div>
        <div class="rf-row"><span>方差半衰期</span><span class="val ${curBeta2 === 0.95 ? "mint" : "rose"}">${halfLifeSteps} 步</span></div>
        <div class="rf-row"><span>从梯度冲击中恢复</span><span class="val">${curBeta2 === 0.95 ? "迅速（约 14 步）✓" : "严重停滞（约 693 步）✗"}</span></div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
        ${curBeta2 === 0.95
          ? "✓ 调优后的 β₂ = 0.95 可快速清除受污染的二阶矩方差，防止过时的阻尼影响 DeltaNet 循环状态更新。"
          : "⚠️ 标准 β₂ = 0.999 会在瞬时梯度异常值过后仍长期压低有效学习率，导致混合状态收敛停滞。"}
      </div>
    `;
  }

  if (optimBetaSeg) {
    bindGroup(optimBetaSeg, b => { curBeta2 = +b.dataset.beta; renderOptimDynamics(); });
  }
  renderOptimDynamics();
})();

/* ============================================================
   Section 11: Post-Training & Reasoning Alignment Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Interactive 4-Stage Post-Training Pipeline Tabs
     ------------------------------------------------------------ */
  const pipe = $("postTrainPipe");
  if (pipe) {
    pipe.querySelectorAll(".stage").forEach(st => {
      st.addEventListener("click", () => {
        pipe.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
        st.classList.add("active");
      });
    });
  }

  /* ------------------------------------------------------------
     2. Widget: Live GRPO Rollout & Policy Gradient Simulator (#w-grpo-sim)
     ------------------------------------------------------------ */
  const grpoRolloutsList = $("grpoRolloutsList");
  const grpoGroupStats = $("grpoGroupStats");
  const grpoPromptSel = $("grpoPromptSel");
  const grpoGroupSize = $("grpoGroupSize");
  const grpoTemp = $("grpoTemp");
  const grpoKlBeta = $("grpoKlBeta");

  const PROMPT_BANKS = {
    math: {
      title: "数学：丢番图方程中的模逆元",
      passRollouts: [
        "扩展欧几里得算法：43 = 2(17) + 9，17 = 1(9) + 8，9 = 1(8) + 1。反向代入：1 = 9 - (17 - 9) = 2(43 - 2(17)) - 17 = 2(43) - 5(17)。因此 17⁻¹ ≡ -5 ≡ 38 (mod 43)。x ≡ 5 × 38 = 190 ≡ 18 (mod 43)。验证：17(18) = 306 = 7(43) + 5。✓",
        "直接进行模搜索：x = (5 + 43k)/17。当 k=7 时：5 + 301 = 306，306 / 17 恰好等于 18。最小正整数 x = 18。✓",
        "费马小定理：17⁴¹ ≡ 1 (mod 43)。通过二进制快速幂计算 17⁴¹ * 5 mod 43……得到 x = 18。✓",
        "分解线性同余式：17x - 43y = 5。贝祖恒等式给出特解 (18, 7)。通解为 x = 18 + 43n。最小正整数 x = 18。✓"
      ],
      failRollouts: [
        "算术失误：将 17⁻¹ 算成 28 而非 38。x ≡ 5 × 28 = 140 ≡ 11 (mod 43)。验证失败：17(11) = 187 ≠ 5 mod 43。✗",
        "模数混淆：误将 43 视为合数并进行质因数分解。输出 x = 24。✗",
        "循环超时／无休止推理：反复计算 17k 而不终止。✗",
        "格式违规：未用方框标出最终整数答案。✗"
      ]
    },
    code: {
      title: "代码：并发线程池内存泄漏修复",
      passRollouts: [
        "发现工作线程清理循环中存在未关闭的通道。添加 defer close(worker.jobs)，并使用 sync.WaitGroup 同步。所有竞态测试均通过，泄漏为 0。✓",
        "用有界工作池信号量替代直接启动 goroutine。取消上下文时会彻底清空待处理任务。单元测试断言通过。✓",
        "修复任务闭包中的循环引用，该引用导致工作线程指针滞留在堆中。垃圾收集器可立即释放缓冲区，已使用 pprof 验证。✓",
        "在线程池调整大小期间采用原子指针交换。在每秒 1 万请求的负载测试下消除了互斥锁竞争死锁。✓"
      ],
      failRollouts: [
        "引入互斥锁死锁：在接收通道消息之前于 defer 中获取锁。线程池在关闭时卡死。✗",
        "内存泄漏恶化：添加缓冲区时未检查边界，突发流量下触发 OOM panic。✗",
        "语法错误：闭包签名缺少返回类型，编译失败。✗",
        "竞态条件：读取工作线程状态时未使用原子加载，触发数据竞态检测器。✗"
      ]
    },
    agent: {
      title: "智能体：带重试回退的三步工具调用",
      passRollouts: [
        "<think>API 端点返回 503。需要退避并查询备用镜像。</think> <call:fetch_mirror(url='api.v2')/> -> 状态：200。已提取模式并验证校验和。✓",
        "<think>JSON 响应缺少 'id' 键。正在检查原始载荷……</think> 已解析 'data.item.id' 下的嵌套载荷。回退成功。✓",
        "<think>遇到 429 速率限制。</think> 执行指数抖动休眠（150ms）。重试并成功解析响应。✓",
        "<think>模式格式不匹配。</think> 已将时间戳从 ISO-8601 转换为纪元毫秒。断言通过。✓"
      ],
      failRollouts: [
        "<think>API 返回 503。</think> 无延迟紧密循环重试 10 次，耗尽重试预算。✗",
        "臆造不存在的工具参数 'force_override=True'。工具调用因 ValidationError 崩溃。✗",
        "向用户泄露内部草稿，却未生成结构化 JSON 输出。✗",
        "忽略错误返回码并臆造模拟数据。✗"
      ]
    }
  };

  function renderGrpo() {
    if (!grpoRolloutsList || !grpoGroupStats) return;
    const taskKey = grpoPromptSel.value || "math";
    const G = +grpoGroupSize.value || 8;
    const temp = +grpoTemp.value || 0.7;
    const beta = +grpoKlBeta.value || 0.04;

    const bank = PROMPT_BANKS[taskKey];
    const passProb = Math.max(0.2, Math.min(0.85, 0.75 - (temp - 0.7) * 0.4));

    const rollouts = [];
    for (let i = 0; i < G; i++) {
      const isPass = Math.random() < passProb;
      const textPool = isPass ? bank.passRollouts : bank.failRollouts;
      const text = textPool[i % textPool.length];
      const r = isPass ? 1.0 : (Math.random() < 0.15 ? 0.2 : 0.0);
      rollouts.push({ id: `y_${i + 1}`, text, r, isPass });
    }

    const meanR = rollouts.reduce((acc, r) => acc + r.r, 0) / G;
    const varR = rollouts.reduce((acc, r) => acc + Math.pow(r.r - meanR, 2), 0) / G;
    const stdR = Math.sqrt(varR);

    // Compute standardized advantages
    rollouts.forEach(r => {
      r.adv = (r.r - meanR) / (stdR + 1e-5);
    });

    const estKl = 0.025 * temp * temp;
    const lossGrpo = -(rollouts.reduce((acc, r) => acc + Math.min(r.adv, 1.2 * r.adv), 0) / G) + beta * estKl;

    grpoGroupStats.innerHTML = `
      组均值 μ：<b>${meanR.toFixed(2)}</b> · 标准差 σ：<b>${stdR.toFixed(2)}</b> · KL：<b>${estKl.toFixed(3)}</b> · J_GRPO：<b style="color:var(--mint)">${lossGrpo.toFixed(3)}</b>
    `;

    grpoRolloutsList.innerHTML = rollouts.map(r => {
      const isPos = r.adv >= 0;
      return `
        <div class="grpo-card ${isPos ? "pos" : "neg"}">
          <div class="grpo-id">${r.id}</div>
          <div class="grpo-text">${r.text}</div>
          <div class="grpo-meta">
            <span class="grpo-reward-badge ${r.isPass ? "pass" : "fail"}">奖励 r = ${r.r.toFixed(2)}</span>
            <span class="grpo-adv ${isPos ? "pos" : "neg"}">优势 A = ${isPos ? "+" : ""}${r.adv.toFixed(2)} ${isPos ? "▲ 增强" : "▼ 抑制"}</span>
          </div>
        </div>
      `;
    }).join("");
  }

  [grpoPromptSel, grpoGroupSize, grpoTemp, grpoKlBeta].forEach(sel => {
    if (sel) {
      sel.addEventListener("change", renderGrpo);
      sel.addEventListener("input", renderGrpo);
    }
  });
  renderGrpo();


  /* ------------------------------------------------------------
     3. Widget: Upgraded Thinking-Budget Explorer (#w-effort)
     ------------------------------------------------------------ */
  const effortSvg = $("effortSvg");
  const effortOut = $("effortOut");
  const effortSel = $("effortSel");
  const effortDiff = $("effortDiff");
  const effortPreserve = $("effortPreserve");

  function renderEffort() {
    if (!effortSvg || !effortOut) return;
    const effort = +effortSel.value;
    const diff = +effortDiff.value;
    const preserve = effortPreserve.value === "true";
    if (effortSel.type === "range") effortSel.setAttribute("aria-valuetext", `推理强度 ${effort}，共 3 级`);
    if (effortDiff.type === "range") effortDiff.setAttribute("aria-valuetext", `任务难度 ${diff}%`);

    const NS = "http://www.w3.org/2000/svg";
    effortSvg.innerHTML = "";

    const W = 720, H = 260, ox = 56, oy = 215, pw = 620, ph = 175;
    const X = f => ox + (f / 3) * pw;
    const Ya = a => oy - (a / 100) * ph;
    const Yt = t => oy - (t / 16384) * ph;

    // Grid lines & Axes
    const l1 = document.createElementNS(NS, "line");
    l1.setAttribute("x1", ox); l1.setAttribute("y1", oy);
    l1.setAttribute("x2", ox + pw); l1.setAttribute("y2", oy);
    l1.setAttribute("stroke", "rgba(26,22,18,.15)");
    effortSvg.appendChild(l1);

    const l2 = document.createElementNS(NS, "line");
    l2.setAttribute("x1", ox); l2.setAttribute("y1", oy - ph);
    l2.setAttribute("x2", ox); l2.setAttribute("y2", oy);
    l2.setAttribute("stroke", "rgba(26,22,18,.15)");
    effortSvg.appendChild(l2);

    [[0, "关闭（0 词元）"], [1, "低（1k）"], [2, "中（4k）"], [3, "极高（16k）"]].forEach(([f, s]) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", X(f)); t.setAttribute("y", oy + 18);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", PAL.muted);
      t.setAttribute("font-size", "10");
      t.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      t.textContent = s;
      effortSvg.appendChild(t);
    });

    function getAcc(eff, d) {
      const base = 25 + 0.65 * d;
      const gain = eff === 0 ? 0.35 : (eff === 1 ? 0.72 : (eff === 2 ? 0.91 : 0.98));
      return Math.min(99.4, base * gain);
    }
    function getToks(eff) {
      return eff === 0 ? 0 : (eff === 1 ? 1024 : (eff === 2 ? 4096 : 16384));
    }

    // Drawn token curve interpolates the same budgets getToks reports,
    // so the plotted value at each effort level matches the readout exactly.
    const TOK_BUDGETS = [0, 1024, 4096, 16384];
    const simToks = f => {
      const lo = Math.min(2, Math.floor(f));
      return TOK_BUDGETS[lo] + (TOK_BUDGETS[lo + 1] - TOK_BUDGETS[lo]) * (f - lo);
    };

    let da = "", dt = "";
    for (let f = 0; f <= 3.001; f += 0.05) {
      const simulatedAcc = getAcc(f, diff);
      da += (f === 0 ? "M" : "L") + X(f) + "," + Ya(simulatedAcc) + " ";
      dt += (f === 0 ? "M" : "L") + X(f) + "," + Yt(Math.min(16384, simToks(f))) + " ";
    }

    const accP = document.createElementNS(NS, "path");
    accP.setAttribute("d", da);
    accP.setAttribute("fill", "none");
    accP.setAttribute("stroke", PAL.mint);
    accP.setAttribute("stroke-width", "2.5");
    effortSvg.appendChild(accP);

    const tokP = document.createElementNS(NS, "path");
    tokP.setAttribute("d", dt);
    tokP.setAttribute("fill", "none");
    tokP.setAttribute("stroke", PAL.iris);
    tokP.setAttribute("stroke-width", "2");
    tokP.setAttribute("stroke-dasharray", "5 4");
    effortSvg.appendChild(tokP);

    const curAcc = getAcc(effort, diff);
    const curToks = getToks(effort);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", X(effort)); dot.setAttribute("cy", Ya(curAcc));
    dot.setAttribute("r", "6");
    dot.setAttribute("fill", PAL.amber);
    dot.setAttribute("stroke", PAL.ink);
    dot.setAttribute("stroke-width", "2");
    effortSvg.appendChild(dot);

    const tokDot = document.createElementNS(NS, "circle");
    tokDot.setAttribute("cx", X(effort)); tokDot.setAttribute("cy", Yt(curToks));
    tokDot.setAttribute("r", "5");
    tokDot.setAttribute("fill", PAL.iris);
    effortSvg.appendChild(tokDot);

    const recoveryRate = preserve ? Math.min(96, curAcc * 1.08) : Math.max(40, curAcc * 0.82);

    effortOut.innerHTML = `
      <div class="row"><span>难度 ${diff}% 时的预期基准 Pass@1</span><span class="v m" style="font-size:14px;font-weight:700">${curAcc.toFixed(1)}%</span></div>
      <div class="row"><span>分配的审慎思考词元</span><span class="v i">${curToks.toLocaleString()} 词元</span></div>
      <div class="row"><span>智能体多轮自我纠正概率（preserve_thinking=${preserve}）</span><span class="v a">${recoveryRate.toFixed(1)}%</span></div>
      <div class="row"><span>相较极高强度尚未获得的边际准确率</span><span class="v" style="color:var(--faint)">${(getAcc(3, diff) - curAcc).toFixed(1)} 个百分点</span></div>
      <div style="color:var(--faint);font-size:12px;margin-top:8px;line-height:1.45">
        ${effort === 0
          ? "已启用直接即时模式：无额外延迟，最适合聊天和简单检索。"
          : `已启用思考模式：子目标分解分配约 ${curToks.toLocaleString()} 个词元的测试时计算量。`}
      </div>
    `;
  }

  [effortSel, effortDiff, effortPreserve].forEach(el => {
    if (el) {
      el.addEventListener("input", renderEffort);
      el.addEventListener("change", renderEffort);
    }
  });
  renderEffort();


  /* ------------------------------------------------------------
     4. Widget: Chain-of-Thought & Agentic Trace Visualizer (#w-cot-trace)
     ------------------------------------------------------------ */
  const cotTraceViewer = $("cotTraceViewer");
  const traceScenarioSeg = $("traceScenarioSeg");
  let curScenario = "math_backtrack";

  const TRACES = {
    math_backtrack: [
      {
        role: "user",
        text: "求方程 $\\sqrt{x + 3} - \\sqrt{x - 2} = 1$ 的所有实数解。"
      },
      {
        role: "assistant",
        think: `1. 检查定义域：x + 3 >= 0 => x >= -3，且 x - 2 >= 0 => x >= 2。交集定义域为 x ∈ [2, +∞)。
2. 分离根式：sqrt(x + 3) = 1 + sqrt(x - 2)。
3. 两边平方：x + 3 = 1 + 2*sqrt(x - 2) + (x - 2)。
4. 化简代数式：x + 3 = x - 1 + 2*sqrt(x - 2) => 4 = 2*sqrt(x - 2) => sqrt(x - 2) = 2。
5. 等一下，再检查一下符号：3 - (-1) = 4，正确！
6. 再次平方：x - 2 = 4 => x = 6。
7. 验证：sqrt(6 + 3) - sqrt(6 - 2) = sqrt(9) - sqrt(4) = 3 - 2 = 1。与右侧完全一致！`,
        output: "唯一的实数解是 <strong>x = 6</strong>。"
      }
    ],
    agent_swe: [
      {
        role: "user",
        text: "修复在 CI 中间歇性失败的不稳定测试 `test_websocket_heartbeat_timeout`。"
      },
      {
        role: "assistant",
        think: `问题报告了一处竞态条件：心跳计时器在模拟客户端套接字注册握手之前触发。
让我们检查测试文件和连接清理过程。`,
        toolCall: "run_command(CommandLine='pytest tests/test_ws.py -k test_websocket_heartbeat_timeout --count=5')",
        toolObs: "失败 tests/test_ws.py::test_websocket_heartbeat_timeout（5 次中有 2 次失败：AssertionError：50ms 内未收到心跳）",
        thinkCont: `根因已确认：测试使用硬编码的 50ms 休眠，而不是等待连接状态条件。
将测试改为使用事件驱动的 'wait_for_condition(timeout=2.0)'。`,
        toolCall2: "replace_file_content(TargetFile='tests/test_ws.py', Instruction='Replace fixed sleep with event waiter')",
        toolObs2: "文件编辑成功。",
        output: "已将脆弱的 50ms 休眠替换为对 `client.is_connected` 的显式条件等待器，从而修复不稳定测试。5 次验证测试现均可稳定通过。"
      }
    ],
    direct_bypass: [
      {
        role: "user",
        text: "澳大利亚的首都是哪里？"
      },
      {
        role: "assistant",
        output: "澳大利亚的首都是**堪培拉**。"
      }
    ]
  };

  function renderCotTrace() {
    if (!cotTraceViewer) return;
    const steps = TRACES[curScenario] || TRACES.math_backtrack;

    cotTraceViewer.innerHTML = steps.map(step => {
      if (step.role === "user") {
        return `
          <div class="cot-step">
            <span class="cot-role">用户</span>
            <div class="cot-bubble" style="background:var(--bg-1)">${step.text}</div>
          </div>
        `;
      } else {
        let innerHtml = "";
        if (step.think) {
          innerHtml += `<div class="cot-think-block"><b>&lt;think&gt;</b><br>${step.think.replace(/\n/g, "<br>")}</div>`;
        }
        if (step.toolCall) {
          innerHtml += `<div class="cot-tool-block"><b>工具调用：</b> <code>${step.toolCall}</code></div>`;
        }
        if (step.toolObs) {
          innerHtml += `<div class="cot-obs-block"><b>工具输出：</b> ${step.toolObs}</div>`;
        }
        if (step.thinkCont) {
          innerHtml += `<div class="cot-think-block"><b>&lt;think&gt;（自我反思）</b><br>${step.thinkCont.replace(/\n/g, "<br>")}</div>`;
        }
        if (step.toolCall2) {
          innerHtml += `<div class="cot-tool-block"><b>工具调用：</b> <code>${step.toolCall2}</code></div>`;
        }
        if (step.toolObs2) {
          innerHtml += `<div class="cot-obs-block"><b>工具输出：</b> ${step.toolObs2}</div>`;
        }
        if (step.output) {
          innerHtml += `<div style="margin-top:8px">${step.output}</div>`;
        }
        return `
          <div class="cot-step">
            <span class="cot-role" style="color:var(--iris)">Qwen 3.8-27B 助手</span>
            <div class="cot-bubble">${innerHtml}</div>
          </div>
        `;
      }
    }).join("");
    katexRender(cotTraceViewer);                                  // user turns carry inline math
  }

  if (traceScenarioSeg) {
    bindGroup(traceScenarioSeg, b => { curScenario = b.dataset.scen; renderCotTrace(); });
  }
  renderCotTrace();
})();



/* ============================================================
   W10, W11, W12 — Section 12: Inference & Optimizations Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Upgraded VRAM Planner (#w-quant)
     ------------------------------------------------------------ */
  const quantOut = $("quantOut");
  const vsbWeights = $("vsbWeights");
  const vsbKv = $("vsbKv");
  const vsbDeltanet = $("vsbDeltanet");
  const vsbOverhead = $("vsbOverhead");
  const vramTotalLabel = $("vramTotalLabel");

  function renderQuant() {
    if (!quantOut) return;
    const weightsGB = +$("quantPrec").value;
    const T = +$("quantCtx").value;
    const batch = Math.max(1, +$("quantBatch").value || 1);
    const budget = +$("quantGpu").value;
    const kvB = +$("quantKv").value;

    // KV: 16 GQA layers · 2 (K & V) · 4 heads · 256 dim · kvB bytes/token/layer
    const kvBytes = 2 * 4 * 256 * 16 * kvB * T * batch;
    const kvGB = kvBytes / 1e9;

    // DeltaNet: 48 layers · 48 heads · 128 · 128 · 2 bytes (BF16 state) · batch
    const dnStateBytes = 48 * 48 * 128 * 128 * 2 * batch;
    const dnStateGB = dnStateBytes / 1e9;

    // Runtime Activation & CUDA context overhead
    const overheadGB = 1.5 + (0.1 * batch);
    const totalGB = weightsGB + kvGB + dnStateGB + overheadGB;
    const fits = totalGB <= budget;

    // Update stacked visual memory bar
    const maxBar = Math.max(budget, totalGB);
    stackSegments([
      [vsbWeights, (weightsGB / maxBar) * 100],
      [vsbKv, (kvGB / maxBar) * 100],
      [vsbDeltanet, Math.max(1.5, (dnStateGB / maxBar) * 100)],
      [vsbOverhead, (overheadGB / maxBar) * 100],
    ]);

    if (vramTotalLabel) {
      vramTotalLabel.innerHTML = `<span style="color:${fits ? "var(--mint)" : "var(--rose)"}">${totalGB.toFixed(2)} GB</span> / ${budget} GB`;
    }

    // Precision name derives from the CFG.total footprints (BF16 55.2 / FP8 27.6 / NVFP4 13.8),
    // not from hardcoded option strings — stays correct if the weights total ever moves.
    const precTable = [
      [CFG.total * 2 / 1e9, "BF16"],
      [CFG.total / 1e9, "FP8"],
      [CFG.total / 2e9, "NVFP4"],
      [16, "Q4_K_M"]
    ];
    const precHit = precTable.find(([gb]) => Math.abs(gb - weightsGB) < 0.75);
    const precName = precHit ? precHit[1] : "自定义";
    const kvName = { "2": "BF16", "1": "FP8", "0.5": "NVFP4" }[String(kvB)] || "自定义";

    quantOut.innerHTML = `
      <div class="row"><span>1. 模型权重（${precName}量化）</span><span class="v">${weightsGB.toFixed(1)} GB</span></div>
      <div class="row"><span>2. GQA KV 缓存（16 层 × ${batch} 个序列，${kvName} 格式，共 ${T.toLocaleString()} 个词元）</span><span class="v a">${kvGB < 1 ? (kvGB * 1000).toFixed(0) + " MB" : kvGB.toFixed(2) + " GB"}</span></div>
      <div class="row"><span>3. DeltaNet 循环状态（48 层，固定矩阵）</span><span class="v m">${(dnStateGB * 1000).toFixed(1)} MB</span></div>
      <div class="row"><span>4. CUDA 驱动与运行时激活缓冲区</span><span class="v">${overheadGB.toFixed(2)} GB</span></div>
      <div class="row" style="border-top:1px solid var(--line-strong);margin-top:6px;padding-top:10px">
        <span><b>服务总占用与 ${budget} GB 目标 GPU 对比</b></span>
        <span class="big" style="color:${fits ? "var(--mint)" : "var(--rose)"}">
          ${totalGB.toFixed(2)} GB ${fits ? `· 轻松容纳（余量 ${(budget - totalGB).toFixed(1)} GB ✓）` : `· 超出预算 ${(totalGB - budget).toFixed(1)} GB ✗`}
        </span>
      </div>
      <div style="color:var(--faint);font-size:12px;margin-top:8px;line-height:1.45">
        ${fits 
          ? "✓ 配置已验证。可进行高吞吐服务，不存在显存溢出风险。"
          : "⚠️ 解决溢出：（1）将 KV 缓存数据类型从 BF16 切换为 FP8/NVFP4；（2）将权重精度降至 NVFP4；或（3）降低并发数。"}
      </div>
    `;
  }

  ["quantPrec", "quantCtx", "quantBatch", "quantGpu", "quantKv"].forEach(id => {
    const el = $(id);
    if (el) {
      el.addEventListener("input", renderQuant);
      el.addEventListener("change", renderQuant);
    }
  });
  renderQuant();


  /* ------------------------------------------------------------
     2. Widget: Hardware Roofline & Tok/s Simulator (#w-roofline)
     ------------------------------------------------------------ */
  const rfSvg = $("rooflineSvg");
  const rfStats = $("rooflineStats");
  const rfHwSelect = $("rfHardware");
  const rfQuantSelect = $("rfQuant");
  const rfMtpSelect = $("rfMtp");

  // Regenerate β option labels from §08's acceptance formula so the dropdown
  // always agrees with the simulator (labels ship as static "~N.N tok/step").
  rfMtpSelect?.querySelectorAll("option").forEach(o => {
    const b = parseFloat(o.value);
    if (Number.isFinite(b)) o.textContent = o.textContent.replace(/~\d+(\.\d+)?/, `~${mtpEtoks(b, 2).toFixed(2)}`);
  });

  const HW_PROFILES = {
    "5090":  { name: "NVIDIA RTX 5090", vram: 32, bw: 1792, tflops: 1800, bus: "GDDR7 512-bit" },
    "4090":  { name: "NVIDIA RTX 4090", vram: 24, bw: 1008, tflops: 660,  bus: "GDDR6X 384-bit" },
    "m4max": { name: "Apple M4 Max",    vram: 128, bw: 546,  tflops: 120,  bus: "Unified LPDDR5X" },
    "a6000": { name: "RTX 6000 Ada",   vram: 48, bw: 960,   tflops: 750,  bus: "GDDR6 384-bit" },
    "h100":  { name: "NVIDIA H100 SXM", vram: 80, bw: 3350,  tflops: 2000, bus: "HBM3 5120-bit" },
    "b200":  { name: "NVIDIA B200 NVL", vram: 192, bw: 8000, tflops: 4500, bus: "HBM3e 8192-bit" }
  };

  function renderRoofline() {
    if (!rfSvg || !rfStats) return;
    const hwKey = rfHwSelect.value || "5090";
    const weightFootprintGB = +rfQuantSelect.value || 13.8;   // NVFP4 footprint from CFG.total
    const mtpBeta = +rfMtpSelect.value || 0.8;

    const hw = HW_PROFILES[hwKey];
    const baselineTokS = hw.bw / weightFootprintGB;
    // §08's acceptance convention: net speedup S ≈ E[tokens/pass] / (1 + γ_verify)
    const effectiveTokS = baselineTokS * mtpSpeedup(mtpBeta, 2);
    const latencyPerTokMs = 1000 / effectiveTokS;
    const ttftShortMs = (weightFootprintGB / hw.bw) * 1000 + 12;

    // Operational Intensity for single batch decode (2 FLOP × params in B / GB)
    const decodeIntensity = (2 * 27.58) / weightFootprintGB; // ~3.56 FLOP/Byte at NVFP4 footprint

    const NS = "http://www.w3.org/2000/svg";
    rfSvg.innerHTML = "";

    const W = 560, H = 260, P = { l: 48, r: 24, t: 24, b: 40 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Log-log scales: X = Operational Intensity (0.1 to 1000), Y = Performance (1 to 5000 TFLOPs)
    const minX = Math.log10(0.2), maxX = Math.log10(1000);
    const minY = Math.log10(1), maxY = Math.log10(10000);

    const fx = v => P.l + ((Math.log10(Math.max(0.2, v)) - minX) / (maxX - minX)) * IW;
    const fy = v => P.t + (1 - (Math.log10(Math.max(1, v)) - minY) / (maxY - minY)) * IH;

    // Grid lines
    [1, 10, 100, 1000].forEach(xVal => {
      const gx = fx(xVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", fy(1));
      l.setAttribute("x2", gx); l.setAttribute("y2", fy(10000));
      rfSvg.appendChild(l);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", "bp-tick");
      t.setAttribute("x", gx); t.setAttribute("y", fy(1) + 16);
      t.setAttribute("text-anchor", "middle");
      t.textContent = `${xVal}`;
      rfSvg.appendChild(t);
    });

    [10, 100, 1000, 5000].forEach(yVal => {
      const gy = fy(yVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", fx(0.2)); l.setAttribute("y1", gy);
      l.setAttribute("x2", fx(1000)); l.setAttribute("y2", gy);
      rfSvg.appendChild(l);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", "bp-tick");
      t.setAttribute("x", P.l - 8); t.setAttribute("y", gy + 3);
      t.setAttribute("text-anchor", "end");
      t.textContent = `${yVal}T`;
      rfSvg.appendChild(t);
    });

    // Roofline ceiling line (Diagonal BW limit -> Horizontal Compute Peak)
    // Knee = arithmetic intensity where ai·BW/1000 reaches the compute peak
    const kneePoint = hw.tflops * 1000 / hw.bw;
    const kneeX = fx(kneePoint);
    const kneeY = fy(hw.tflops);

    const roofPath = document.createElementNS(NS, "path");
    const dStr = `M ${fx(0.2)},${fy(0.2 * hw.bw / 1000)} L ${kneeX},${kneeY} L ${fx(1000)},${kneeY}`;
    roofPath.setAttribute("d", dStr);
    roofPath.setAttribute("fill", "none");
    roofPath.setAttribute("stroke", PAL.amber);
    roofPath.setAttribute("stroke-width", "2.5");
    rfSvg.appendChild(roofPath);

    // Shaded Memory Bound Region
    const memShade = document.createElementNS(NS, "polygon");
    memShade.setAttribute("points", `${fx(0.2)},${fy(1)} ${fx(0.2)},${fy(0.2 * hw.bw / 1000)} ${kneeX},${kneeY} ${kneeX},${fy(1)}`);
    memShade.setAttribute("fill", "rgba(184,52,31,0.06)");
    rfSvg.appendChild(memShade);

    // Current Decode Operating Point Dot (Memory Bound)
    const opX = fx(decodeIntensity);
    const opY = fy(decodeIntensity * hw.bw / 1000);

    const halo = document.createElementNS(NS, "circle");
    halo.setAttribute("cx", opX); halo.setAttribute("cy", opY);
    halo.setAttribute("r", "12");
    halo.setAttribute("fill", "rgba(29,110,107,0.25)");
    rfSvg.appendChild(halo);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", opX); dot.setAttribute("cy", opY);
    dot.setAttribute("r", "5.5");
    dot.setAttribute("fill", PAL.mint);
    dot.setAttribute("stroke", PAL.ink);
    dot.setAttribute("stroke-width", "2");
    rfSvg.appendChild(dot);

    const dotLbl = document.createElementNS(NS, "text");
    dotLbl.setAttribute("x", opX + 12);
    dotLbl.setAttribute("y", opY + 4);
    dotLbl.setAttribute("fill", PAL.mint);
    dotLbl.setAttribute("font-size", "10");
    dotLbl.setAttribute("font-weight", "700");
    dotLbl.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    dotLbl.textContent = `批大小=1 解码（${decodeIntensity.toFixed(1)} FLOP/B）`;
    rfSvg.appendChild(dotLbl);

    // Roofline text captions
    const capX = document.createElementNS(NS, "text");
    capX.setAttribute("class", "bp-cap");
    capX.setAttribute("x", P.l + IW / 2); capX.setAttribute("y", H - 6);
    capX.setAttribute("text-anchor", "middle");
    capX.textContent = "运算强度（FLOPs / 字节，对数刻度）→";
    rfSvg.appendChild(capX);

    const capY = document.createElementNS(NS, "text");
    capY.setAttribute("class", "bp-cap");
    capY.setAttribute("transform", `rotate(-90 12 ${P.t + IH / 2})`);
    capY.setAttribute("x", 12); capY.setAttribute("y", P.t + IH / 2);
    capY.setAttribute("text-anchor", "middle");
    capY.textContent = "可达性能（TFLOPs）→";
    rfSvg.appendChild(capY);

    // Stats Sidebar
    rfStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">硬件服务上限</div>
        <div class="rf-row"><span>内存带宽</span><span class="val amber">${hw.bw} GB/s</span></div>
        <div class="rf-row"><span>基准生成速度</span><span class="val">${baselineTokS.toFixed(1)} 词元/秒</span></div>
        <div class="rf-row"><span>MTP 推测速度</span><span class="val mint" style="font-size:14px;font-weight:700">${effectiveTokS.toFixed(1)} 词元/秒</span></div>
        <div class="rf-row"><span>词元间延迟</span><span class="val">${latencyPerTokMs.toFixed(1)} ms</span></div>
        <div class="rf-row"><span>首词元时间（估算）</span><span class="val">${ttftShortMs.toFixed(0)} ms</span></div>
      </div>

      <div class="rf-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:12px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">屋顶线要点</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          单流生成稳定运行在<b>内存带宽受限斜坡</b>内（${decodeIntensity.toFixed(1)} 对比 ${kneePoint.toFixed(0)} FLOP/B 拐点）。MTP 推测草稿头无需增加权重内存流量，即可将有效速度提升至 <b>${mtpSpeedup(mtpBeta, 2).toFixed(2)}×</b>（§08 接受模型，β=${mtpBeta.toFixed(2)}，γ_verify=0.15）。
        </div>
      </div>
    `;
  }

  [rfHwSelect, rfQuantSelect, rfMtpSelect].forEach(sel => {
    if (sel) sel.addEventListener("change", renderRoofline);
  });
  renderRoofline();


  /* ------------------------------------------------------------
     3. Widget: Chunked Prefill & Batch Scheduler (#w-batch-sim)
     ------------------------------------------------------------ */
  const schedPipeline = $("schedPipeline");
  const schedModeSeg = $("schedModeSeg");
  let schedMode = "continuous";

  function renderScheduler() {
    if (!schedPipeline) return;

    if (schedMode === "continuous") {
      schedPipeline.innerHTML = `
        <div class="sched-track">
          <span class="sched-seq-name">请求 1（128k 文档）</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:2" title="预填充分块 1（16k）">分块 1/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="预填充分块 2（16k）">分块 2/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="预填充分块 3（16k）">分块 3/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="预填充分块 4（16k）">分块 4/4</div>
            <div class="sched-chunk decode" style="flex:1" title="自回归解码步骤">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">请求 2（2k 对话）</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:1.5" title="即时预填充（2k）">预填充</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">请求 3（软件工程智能体）</span>
          <div class="sched-blocks">
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk prefill" style="flex:1.5" title="工具观察结果（4k）">工具调用</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--mint);margin-top:6px;display:flex;justify-content:space-between">
          <span>✓ 无流水线空泡 · Tensor Core 利用率：<b>84%</b></span>
          <span>最大词元间抖动：<b>&lt;18 ms</b></span>
        </div>
      `;
    } else {
      // Naive batching
      schedPipeline.innerHTML = `
        <div class="sched-track">
          <span class="sched-seq-name">请求 1（128k 文档）</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:6" title="整体式 128k 预填充停顿（3.8 秒）">完整整体式预填充（GPU 锁定 3.8 秒）</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">请求 2（2k 对话）</span>
          <div class="sched-blocks">
            <div class="sched-chunk bubble" style="flex:6" title="流水线空泡（阻塞等待请求 1）">阻塞 / 停顿（抖动 3.8 秒）</div>
            <div class="sched-chunk prefill" style="flex:1">预填充</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">请求 3（软件工程智能体）</span>
          <div class="sched-blocks">
            <div class="sched-chunk bubble" style="flex:6" title="流水线空泡（已阻塞）">阻塞 / 停顿</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
            <div class="sched-chunk decode" style="flex:1">解码</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--rose);margin-top:6px;display:flex;justify-content:space-between">
          <span>✗ 严重延迟空泡 · Tensor Core 利用率：<b>31%</b></span>
          <span>词元间延迟峰值：<b>3,800 ms</b></span>
        </div>
      `;
    }
  }

  if (schedModeSeg) {
    bindGroup(schedModeSeg, b => { schedMode = b.dataset.mode; renderScheduler(); });
  }
  renderScheduler();
})();

/* ============================================================
   W11 & W12 — Section 14: PyTorch Code Studio & Tensor Tracer
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Live Tensor Dimension & Memory Tracer (#w-code-tracer)
     ------------------------------------------------------------ */
  const tracerFlow = $("tracerFlow");
  const tracerStats = $("tracerStats");
  const traceBatchSelect = $("traceBatch");
  const traceCtxSelect = $("traceCtx");
  const traceDtypeSelect = $("traceDtype");
  const traceModuleSelect = $("traceModule");

  if (!tracerFlow || !tracerStats) return;

  // decimal units — the site's locked convention (GB = 1e9 bytes, matching HF/prose figures)
  function formatBytes(bytes) {
    if (bytes < 1e3) return `${bytes.toFixed(0)} B`;
    if (bytes < 1e6) return `${(bytes / 1e3).toFixed(1)} KB`;
    if (bytes < 1e9) return `${(bytes / 1e6).toFixed(2)} MB`;
    return `${(bytes / 1e9).toFixed(2)} GB`;
  }

  function formatFlops(flops) {
    if (flops < 1e9) return `${(flops / 1e6).toFixed(1)} MFLOPs`;
    if (flops < 1e12) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
    return `${(flops / 1e12).toFixed(2)} TFLOPs`;
  }

  function renderTracer() {
    const B = +traceBatchSelect.value || 1;
    const T = +traceCtxSelect.value || 2048;
    const b = +traceDtypeSelect.value || 2;
    const mod = traceModuleSelect.value || "deltanet";

    const D = 5120;
    const D_FF = 17408;
    const VOCAB = 248320;

    let cards = [];
    let stats = {
      params: 0,
      flops: 0,
      actMemory: 0,
      stateMemory: 0,
      throughputTokS: 0
    };

    if (mod === "deltanet") {
      // 1 Gated DeltaNet layer
      const qk_heads = 16, v_heads = 48, dh = 128;
      const in_proj_dim = (qk_heads * 2 + v_heads) * dh; // 10,240
      const out_proj_dim = v_heads * dh; // 6,144

      const in_bytes = B * T * D * b;
      const qkv_bytes = B * T * in_proj_dim * b;
      const beta_bytes = B * T * v_heads * b;
      const state_bytes = B * v_heads * dh * dh * b;
      const out_bytes = B * T * D * b;

      stats.params = (D * in_proj_dim) + (D * v_heads) + (v_heads * dh * D);
      stats.flops = (2 * B * T * D * in_proj_dim) + (3 * B * T * v_heads * dh * dh) + (2 * B * T * out_proj_dim * D);
      stats.actMemory = in_bytes + qkv_bytes + beta_bytes + out_bytes;
      stats.stateMemory = state_bytes;

      cards = [
        {
          op: "1. 残差输入向量（x）",
          tag: "激活",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "来自前置 RMSNorm 的输入残差隐藏流（d_model = 5120）"
        },
        {
          op: "2. 融合 Q、K、V 线性投影（W_in）",
          tag: "GEMM 投影",
          shape: `[${B}, ${T}, 10240]`,
          bytes: formatBytes(qkv_bytes),
          math: `Q, K: <code>[${B}, ${T}, 16, 128]</code> · V: <code>[${B}, ${T}, 48, 128]</code>`
        },
        {
          op: "3. L2 头归一化与写入门（β_t）",
          tag: "归一化 + 门控",
          shape: `[${B}, ${T}, 48]`,
          bytes: formatBytes(beta_bytes),
          math: `对 Q、K 做 L2 归一化；Q/K 重复 3× 扩展至 48 个头；β_t = 2σ(xW_β + b_β) ∈ (0, 2)`
        },
        {
          op: "4. 关联循环状态矩阵（S_t）",
          tag: "状态内存",
          shape: `[${B}, 48, 128, 128]`,
          highlight: true,
          bytes: formatBytes(state_bytes),
          math: `Delta 规则矩阵更新：<code>S_t = α_t S_{t-1} − β_t (S_{t-1} k_t − v_t) k_tᵀ</code>`
        },
        {
          op: "5. 关联读取与输出投影（W_out）",
          tag: "输出 GEMM",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `y_t = S_tᵀ q_t ∈ <code>[${B}, ${T}, 6144]</code> → 线性投影至 <code>[${B}, ${T}, 5120]</code>`
        }
      ];
    } else if (mod === "attention") {
      // 1 Gated Full Attention layer (GQA 24:4)
      const q_heads = 24, kv_heads = 4, dh = 256;
      const q_dim = q_heads * dh; // 6,144
      const kv_dim = kv_heads * dh; // 1,024

      const in_bytes = B * T * D * b;
      const q_bytes = B * T * q_dim * b;
      const kv_bytes = 2 * B * T * kv_dim * b;
      const gate_bytes = B * T * q_dim * b;
      const attn_map_bytes = B * q_heads * T * Math.min(T, 4096) * b;
      const out_bytes = B * T * D * b;

      stats.params = (D * q_dim) + (2 * D * kv_dim) + (D * q_dim) + (q_dim * D);
      stats.flops = (2 * B * T * D * (q_dim + 2 * kv_dim + q_dim)) + (4 * B * q_heads * T * T * dh) + (2 * B * T * q_dim * D);
      stats.actMemory = in_bytes + q_bytes + kv_bytes + gate_bytes + out_bytes;
      stats.stateMemory = kv_bytes; // Unbounded growing KV buffer

      cards = [
        {
          op: "1. 残差输入向量（x）",
          tag: "激活",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "来自 RMSNorm(x) 的输入残差流"
        },
        {
          op: "2. Q、K、V 与门控投影",
          tag: "线性 GEMM",
          shape: `Q: [${B}, ${T}, 24, 256] · KV: [${B}, ${T}, 4, 256]`,
          bytes: formatBytes(q_bytes + kv_bytes + gate_bytes),
          math: `分组查询注意力（24 个查询头 : 4 个 KV 头 = 6:1 共享）`
        },
        {
          op: "3. 3D MRoPE 旋转相位嵌入",
          tag: "坐标旋转",
          shape: `[${B}, ${T}, 24, 64] 旋转`,
          bytes: formatBytes(q_bytes * (64 / 256)),
          math: `按 <code>[11, 11, 10]</code> 分段旋转头通道 0..63；通道 64..255 保持不变`
        },
        {
          op: "4. 自回归 KV 缓存缓冲区",
          tag: "无界缓冲区",
          shape: `[${B}, 4, ${T}, 256] × 2 (K & V)`,
          highlight: true,
          bytes: formatBytes(kv_bytes),
          math: `随序列长度 T 线性增长：<code>2 · 4 · 256 · T · dtype</code>`
        },
        {
          op: "5. FlashAttention SDPA 与输出 σ 门",
          tag: "注意力 + 门控",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `输出 = <code>(SDPA(Q, K_rep, V_rep) ⊙ σ(x W_g)) @ W_out</code>`
        }
      ];
    } else if (mod === "swiglu") {
      // 1 SwiGLU FFN block
      const in_bytes = B * T * D * b;
      const gate_up_bytes = 2 * B * T * D_FF * b;
      const out_bytes = B * T * D * b;

      stats.params = (2 * D * D_FF) + (D_FF * D);
      stats.flops = (2 * B * T * D * 2 * D_FF) + (3 * B * T * D_FF) + (2 * B * T * D_FF * D);
      stats.actMemory = in_bytes + gate_up_bytes + out_bytes;
      stats.stateMemory = 0;

      cards = [
        {
          op: "1. FFN 前残差输入",
          tag: "激活",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "混合器后的残差流经过 RMSNorm"
        },
        {
          op: "2. 并行门控与上投影（W_gate、W_up）",
          tag: "GEMM 投影",
          shape: `[${B}, ${T}, 17408] × 2`,
          highlight: true,
          bytes: formatBytes(gate_up_bytes),
          math: `扩展比 3.4 × 5120 = 17,408 个中间隐藏维度（填充至 2,048 的倍数）`
        },
        {
          op: "3. Swish（SiLU）激活与逐元素 Hadamard 积",
          tag: "非线性",
          shape: `[${B}, ${T}, 17408]`,
          bytes: formatBytes(B * T * D_FF * b),
          math: `中间张量：<code>h_ffn = SiLU(x @ W_gate) ⊙ (x @ W_up)</code>`
        },
        {
          op: "4. 下投影至残差总线（W_down）",
          tag: "下投影 GEMM",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `最终投影：将 <code>h_ffn @ W_down</code> 加入残差流`
        }
      ];
    } else if (mod === "mtp") {
      // MTP Speculative Prediction Module
      const K = 2;
      const in_bytes = B * T * D * b;
      const proj_bytes = K * B * T * D * b;
      const logits_bytes = K * B * T * VOCAB * b;

      // K draft blocks (one d×d projection each) + ONE shared d·V vocab head —
      // the MTP heads reuse the backbone LM head rather than owning private copies.
      stats.params = K * (D * D) + D * VOCAB;
      stats.flops = K * ((2 * B * T * D * D) + (2 * B * T * D * VOCAB));
      stats.actMemory = in_bytes + proj_bytes + logits_bytes;
      stats.stateMemory = 0;

      cards = [
        {
          op: "1. 骨干网络最终隐藏状态（h_64）",
          tag: "骨干网络状态",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "来自第 64 个超级块层的潜在输出"
        },
        {
          op: "2. MTP 草稿头投影（深度 k=1、k=2）",
          tag: "草稿 GEMM",
          shape: `[${B}, ${T}, 2, 5120]`,
          bytes: formatBytes(proj_bytes),
          math: `同时预测未来第 +1 和 +2 个词元的草稿表示`
        },
        {
          op: "3. 非绑定词表分类 logits",
          tag: "推测 logits",
          shape: `[${B}, ${T}, 2, 248320]`,
          highlight: true,
          bytes: formatBytes(logits_bytes),
          math: `在一次前向步骤中并行评估草稿 logits 是否达到验证阈值`
        }
      ];
    } else if (mod === "superblock") {
      // 1 Hybrid Super-Block = 3 DeltaNet + 1 Attention + 4 FFNs
      const dn_state_bytes = 3 * (B * 48 * 128 * 128 * b);
      const attn_kv_bytes = 2 * B * 4 * T * 256 * b;
      const total_params = (3 * (5120 * 10240 + 5120 * 48 + 6144 * 5120)) +
                           (5120 * 6144 + 2 * 5120 * 1024 + 5120 * 6144 + 6144 * 5120) +
                           (4 * (2 * 5120 * 17408 + 17408 * 5120));

      stats.params = total_params;
      stats.flops = total_params * 2 * B * T;
      stats.actMemory = B * T * 5120 * b * 8;
      stats.stateMemory = dn_state_bytes + attn_kv_bytes;

      cards = [
        {
          op: "层 4k+0：门控 DeltaNet #1 + SwiGLU",
          tag: "线性注意力",
          shape: `状态：[${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "次二次复杂度预填充 + 常量空间递归"
        },
        {
          op: "层 4k+1：门控 DeltaNet #2 + SwiGLU",
          tag: "线性注意力",
          shape: `状态：[${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "带学习遗忘机制的关联记忆检索"
        },
        {
          op: "层 4k+2：门控 DeltaNet #3 + SwiGLU",
          tag: "线性注意力",
          shape: `状态：[${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "深层语法与语言词元追踪"
        },
        {
          op: "层 4k+3：门控全注意力 + SwiGLU",
          tag: "全注意力",
          shape: `KV 缓存：[${B}, 4, ${T}, 256]`,
          highlight: true,
          bytes: formatBytes(attn_kv_bytes),
          math: "利用 3D MRoPE 位置实现精确的全局跨词元路由"
        }
      ];
    }

    // Render flow cards
    tracerFlow.innerHTML = cards.map((c, i) => `
      <div class="trace-card">
        <div class="trace-card-head">
          <span class="trace-op">${c.op}</span>
          <span class="trace-tag">${c.tag}</span>
        </div>
        <div class="trace-tensor">
          <span class="trace-shape ${c.highlight ? "highlight" : ""}">${c.shape}</span>
          <span class="trace-bytes">${c.bytes}</span>
        </div>
        <div class="trace-math">${c.math}</div>
      </div>
      ${i < cards.length - 1 ? `<div class="trace-connector">↓</div>` : ""}
    `).join("");

    // Render statistics sidebar
    tracerStats.innerHTML = `
      <div class="tracer-stat-box">
        <div class="tsb-head">子系统资源概况</div>
        <div class="tsb-row"><span>参数量</span><span class="val amber">${(stats.params / 1e6).toFixed(1)} M</span></div>
        <div class="tsb-row"><span>前向 FLOPs</span><span class="val">${formatFlops(stats.flops)}</span></div>
        <div class="tsb-row"><span>峰值激活内存</span><span class="val">${formatBytes(stats.actMemory)}</span></div>
        <div class="tsb-row"><span>持久状态内存</span><span class="val mint">${formatBytes(stats.stateMemory)}</span></div>
      </div>

      <div class="tracer-stat-box">
        <div class="tsb-head">服务硬件上下文</div>
        <div class="tsb-row"><span>批大小（B）</span><span class="val">${B}</span></div>
        <div class="tsb-row"><span>上下文词元（T）</span><span class="val">${T.toLocaleString()}</span></div>
        <div class="tsb-row"><span>精度数据类型</span><span class="val">${b === 2 ? "BF16（16 位）" : b === 1 ? "FP8（8 位）" : "FP32（32 位）"}</span></div>
        <div class="tsb-row"><span>架构比例</span><span class="val amber">3:1 混合</span></div>
      </div>

      <div class="tracer-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:12px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">关键张量洞见</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          ${mod === "deltanet" 
            ? "无论序列长度 T 是 1 还是 262,144 个词元，DeltaNet 的关联矩阵 <code>S</code> 大小始终严格固定（<b>" + formatBytes(stats.stateMemory) + "</b>）。"
            : mod === "attention"
            ? "全注意力的 KV 缓存随 T 线性增长（<b>" + formatBytes(stats.stateMemory) + "</b>），因此需要采用 3:1 混合设计来限制总占用。"
            : "SwiGLU 提供宽达 3.4× 的扩展，在不产生循环状态开销的情况下增强多步推理能力。"}
        </div>
      </div>
    `;
  }

  [traceBatchSelect, traceCtxSelect, traceDtypeSelect, traceModuleSelect].forEach(sel => {
    if (sel) sel.addEventListener("change", renderTracer);
  });
  renderTracer();


  /* ------------------------------------------------------------
     2. Widget: PyTorch Implementation Studio (#w-code-lab)
     ------------------------------------------------------------ */
  const PYTHON_PROSE_TRANSLATIONS = [
    ["Qwen 3.8-27B — Complete Hybrid Architecture From Scratch (PyTorch)", "Qwen 3.8-27B——从零实现的完整混合架构（PyTorch）"],
    ["64-Layer Hybrid Backbone (16 super-blocks of 3x Gated DeltaNet + 1x Gated Attention)", "64 层混合骨干网络（16 个超级块，每块含 3× 门控 DeltaNet + 1× 门控注意力）"],
    ["Apache 2.0 Reference Implementation · August 2026", "Apache 2.0 参考实现 · 2026 年 8 月"],
    ["3 DeltaNet : 1 Gated Attention", "3 个 DeltaNet : 1 个门控注意力"],
    ["DeltaNet specs (48 layers)", "DeltaNet 规格（48 层）"],
    ["Gated Attention specs (16 layers)", "门控注意力规格（16 层）"],
    ["Grouped-Query Attention (GQA 6:1)", "分组查询注意力（GQA 6:1）"],
    ["[Time, Height, Width]", "[时间、高度、宽度]"],
    ["Feed-Forward specs", "前馈网络规格"],
    ["SwiGLU intermediate dimension", "SwiGLU 中间维度"],
    ["Vocab & Context", "词表与上下文"],
    ["248k vocab with 32-multiple padding", "248k 词表，填充至 32 的倍数"],
    ["262k native context window", "262k 原生上下文窗口"],
    ["2-step speculative prediction heads", "2 步推测预测头"],
    ["Root Mean Square Layer Normalization", "均方根层归一化"],
    ["Swish-Gated Linear Unit Feed-Forward Network", "Swish 门控线性单元前馈网络"],
    ["Gated DeltaNet Linear Attention Layer", "门控 DeltaNet 线性注意力层"],
    ["Features L2 key/query normalization, input-dependent write gate β_t,", "采用 L2 键/查询归一化、依赖输入的写入门 β_t、"],
    ["learned decay rate α_t, and associative memory state update.", "学习得到的衰减率 α_t，以及关联记忆状态更新。"],
    ["Single fused input projection for Q, K, V", "Q、K、V 的单个融合输入投影"],
    ["Input-dependent write gate beta_t", "依赖输入的写入门 beta_t"],
    ["Log-parameterized learned decay rate alpha", "以对数参数化的学习衰减率 alpha"],
    ["Output projection", "输出投影"],
    ["Reshape to multi-head format", "重塑为多头格式"],
    ["L2-Normalization across head dimension", "沿头维度进行 L2 归一化"],
    ["L2-Normalization across feature dimension", "沿特征维度进行 L2 归一化"],
    ["Expand Q/K from 16 to 48 heads to match V (1:3 GQA mapping)", "将 Q/K 从 16 个头扩展至 48 个头以匹配 V（1:3 GQA 映射）"],
    ["Gates: write strength beta in (0, 2), decay alpha in (0, 1)", "门控：写入强度 beta ∈ (0, 2)，衰减 alpha ∈ (0, 1)"],
    ["Initialize recurrent associative state", "初始化循环关联状态"],
    [": S in R^[B, H_v, d_h, d_h]", "：S ∈ R^[B, H_v, d_h, d_h]"],
    ["Stepwise / Chunked delta-rule associative scan", "逐步 / 分块 Delta 规则关联扫描"],
    ["Error prediction", "误差预测"],
    ["Predict retrieved value", "预测检索值"],
    ["Delta state update", "Delta 状态更新"],
    ["Associative delta update", "关联 Delta 更新"],
    ["Readout", "读取"],
    ["Gated Full Self-Attention with 3D MRoPE and Sigmoid Output Gate", "采用 3D MRoPE 和 Sigmoid 输出门的门控全自注意力"],
    ["Gated Full Attention with 3D Multimodal RoPE and Sigmoid Gate", "采用 3D 多模态 RoPE 和 Sigmoid 门的门控全注意力"],
    ["Partial rotary embedding across 3D coordinates", "跨 3D 坐标的部分旋转嵌入"],
    ["Channels 0..63 receive rotary phase; channels 64..255 remain unmodified", "通道 0..63 接收旋转相位；通道 64..255 保持不变"],
    ["Fallback 1D linear positions", "回退到 1D 线性位置"],
    ["Apply 3D Multimodal Rotary Position Embedding", "应用 3D 多模态旋转位置嵌入"],
    ["Handle KV-cache concatenation for autoregressive inference", "处理自回归推理中的 KV 缓存拼接"],
    ["Grouped-Query Attention (GQA 24:4 -> 6:1 repeat)", "分组查询注意力（GQA 24:4 -> 6:1 重复）"],
    ["Transpose to [B, H, T, d_h] for Scaled Dot-Product Attention", "转置为 [B, H, T, d_h] 以执行缩放点积注意力"],
    ["Output Sigmoid Gate modulation", "输出 Sigmoid 门调制"],
    ["1 of 16 Super-Blocks containing 3x Gated DeltaNet + 1x Gated Attention layers.", "16 个超级块之一，包含 3× 门控 DeltaNet 层 + 1× 门控注意力层。"],
    ["Total layers per super-block: 4. Total model layers: 16 * 4 = 64.", "每个超级块共 4 层。模型总层数：16 * 4 = 64。"],
    ["3 Linear-Attention DeltaNet sub-layers", "3 个线性注意力 DeltaNet 子层"],
    ["1 Full-Attention Gated sub-layer", "1 个全注意力门控子层"],
    ["Pass through 3 Gated DeltaNet layers", "通过 3 个门控 DeltaNet 层"],
    ["Pass through 1 Gated Attention layer", "通过 1 个门控注意力层"],
    ["Multi-Token Prediction Speculative Decoding Head", "多词元预测推测解码头"],
    ["K lightweight draft blocks (norm + linear) share ONE dedicated vocab", "K 个轻量草稿块（归一化 + 线性层）共享一个专用词表"],
    ["projection: params ≈ K·d² + d·V ≈ 1.32B at full scale.", "投影：完整规模下参数量约为 K·d² + d·V ≈ 1.32B。"],
    ["Predicts K future draft tokens in parallel during training and inference.", "在训练和推理期间并行预测未来 K 个草稿词元。"],
    ["Complete Qwen 3.8-27B Language Model", "完整的 Qwen 3.8-27B 语言模型"],
    ["Assembled from 16 Hybrid Super-Blocks (64 layers total), RMSNorm,", "由 16 个混合超级块（共 64 层）、RMSNorm、"],
    ["Language Modeling Head, and MTP Speculative Prediction Modules.", "语言模型头和 MTP 推测预测模块组成。"],
    ["Token Embeddings (248,320 padded vocabulary)", "词元嵌入（填充后的 248,320 词表）"],
    ["n_layers/4 Super-blocks = 64 layers at full scale (48 DeltaNet + 16 Gated Attention)", "n_layers/4 个超级块 = 完整规模下 64 层（48 个 DeltaNet + 16 个门控注意力）"],
    ["Final LayerNorm & Untied LM Head", "最终 LayerNorm 与非绑定 LM 头"],
    ["Multi-Token Prediction Head (K=2 drafts)", "多词元预测头（K=2 个草稿）"],
    ["Pass through all 16 super-blocks (64 layers total)", "通过全部 16 个超级块（共 64 层）"],
    ["Gated DeltaNet Layer: O(T) associative linear attention", "门控 DeltaNet 层：O(T) 关联线性注意力"],
    ["Key Properties:", "关键特性："],
    ["L2 normalization across Q and K prevents gradient explosions.", "对 Q 和 K 进行 L2 归一化可防止梯度爆炸。"],
    ["Input-dependent beta_t in (0, 2) controls associative write strength.", "依赖输入的 beta_t ∈ (0, 2) 控制关联写入强度。"],
    ["Parameterized alpha decay rate controls forgetting horizon.", "参数化的 alpha 衰减率控制遗忘范围。"],
    ["Recurrent matrix state S in R^[B, H_v, d_h, d_h] remains constant in memory.", "R^[B, H_v, d_h, d_h] 中的循环矩阵状态 S 占用固定内存。"],
    ["1D fallback or 3D MRoPE rotary frequencies", "1D 回退或 3D MRoPE 旋转频率"],
    ["GQA repeat", "GQA 重复"],
    ["GQA (24 Q heads : 4 KV heads = 6:1 compression ratio).", "GQA（24 个 Q 头 : 4 个 KV 头 = 6:1 压缩比）。"],
    ["MRoPE [11, 11, 10] splits rotary channels across Temporal/Height/Width coordinates.", "MRoPE [11, 11, 10] 按时间/高度/宽度坐标划分旋转通道。"],
    ["Partial RoPE (channels 0..63 rotated, 64..255 untouched).", "部分 RoPE（旋转通道 0..63，通道 64..255 保持不变）。"],
    ["Sigmoid output gate σ(x W_g) softly suppresses attention artifacts.", "Sigmoid 输出门 σ(x W_g) 柔和地抑制注意力伪影。"],
    ["RMSNorm with learned scale parameter", "采用可学习缩放参数的 RMSNorm"],
    ["SwiGLU with 3.4× expansion ratio", "扩展比为 3.4× 的 SwiGLU"],
    ["1 of 16 Super-Blocks: 3 DeltaNet Layers + 1 Gated Attention Layer", "16 个超级块之一：3 个 DeltaNet 层 + 1 个门控注意力层"],
    ["Multi-Token Prediction Heads for K-step speculation", "用于 K 步推测的多词元预测头"],
    ["K lightweight draft blocks share ONE dedicated vocab projection", "K 个轻量草稿块共享一个专用词表投影"],
    ["(params ≈ K·d² + d·V ≈ 1.32B at full scale). Allows generating 2–3", "（完整规模下参数量约为 K·d² + d·V ≈ 1.32B）。可在自回归解码期间"],
    ["tokens per verification step during autoregressive decoding.", "每个验证步骤生成 2–3 个词元。"],
    ["Returns [draft_1_logits, draft_2_logits]", "返回 [draft_1_logits, draft_2_logits]"],
    ["Qwen 3.8-27B From Scratch Architecture Verification", "Qwen 3.8-27B 从零实现架构验证"],
    ["Instantiate toy scale test configuration", "实例化小规模测试配置"],
    ["2 super-blocks", "2 个超级块"],
    ["Initializing model:", "正在初始化模型："],
    ["layers (", "层（"],
    ["hybrid pattern)...", "混合模式）……"],
    ["Model initialized successfully! Parameter count:", "模型初始化成功！参数数量："],
    ["elements.", "个。"],
    ["Test batch forward pass", "测试批量前向传播"],
    ["Running forward pass with batch size B=", "正在运行前向传播，批大小 B="],
    ["sequence length T=", "序列长度 T="],
    ["Forward pass completed in", "前向传播完成，用时"],
    ["Primary LM Logits shape:", "主 LM logits 形状："],
    ["Expected", "预期"],
    ["MTP Draft Logits shape:", "MTP 草稿 logits 形状："],
    ["draft heads", "个草稿头"],
    ["Shape mismatch!", "形状不匹配！"],
    ["[SUCCESS] All tensor shape assertions passed cleanly! Architecture is valid.", "[成功] 所有张量形状断言均已通过！架构有效。"],
    ["Multi-Token Prediction", "多词元预测"],
    ["Partial RoPE", "部分 RoPE"]
  ];

  function translatePythonProse(src) {
    return PYTHON_PROSE_TRANSLATIONS.reduce(
      (text, [from, to]) => text.replaceAll(from, to),
      src
    );
  }

  // Python sources live beside the studio markup as <script type="text/plain" data-py="…">
  // blocks (_parts/05_bench_code_end.html). text/plain is never executed by browsers and
  // needs no fetch(), so file:// keeps working; textContent is exact because the Python
  // body and each closing tag sit flush-left (bootstrap strips one edge newline per side).
  const CODE_FILES = {
    full: {
      name: "qwen38_full.py",
      desc: "完整、自包含的混合架构，含 16×[ΔΔΔA] 超级块、SwiGLU 和 MTP 头。",
      meta: "342 行 · Python 3.10+ · PyTorch 2.0+"
    },
    deltanet: {
      name: "01_deltanet.py",
      desc: "门控 DeltaNet 线性混合器，采用 L2 归一化和关联 Delta 规则状态更新。",
      meta: "68 行 · 线性注意力 §04"
    },
    attention: {
      name: "02_attention.py",
      desc: "门控注意力，采用分组查询注意力（GQA）、3D MRoPE 和 Sigmoid 输出门。",
      meta: "66 行 · 全注意力 §05"
    },
    swiglu: {
      name: "03_swiglu_block.py",
      desc: "RMSNorm、隐藏维度为 17,408 的 SwiGLU FFN，以及 16×[ΔΔΔA] 混合超级块结构。",
      meta: "55 行 · FFN 与残差总线 §06"
    },
    mtp: {
      name: "04_mtp_head.py",
      desc: "用于并行推测解码的多词元预测草稿头。",
      meta: "24 行 · 推测头 §08"
    },
    verify: {
      name: "05_run_verify.py",
      desc: "可执行测试工具，包含形状断言、参数计数和前向推理计时。",
      meta: "46 行 · 验证脚本"
    }
  };
  document.querySelectorAll('script[type="text/plain"][data-py]').forEach(el => {
    const f = CODE_FILES[el.dataset.py];
    if (!f) return;
    f.code = translatePythonProse(el.textContent.replace(/^\n/, "").replace(/\n$/, ""));
  });

  let activeFile = "full";
  let showAnnotations = true;

  const codeTabsContainer = $("codeStudioTabs");
  const codeContentEl = $("codeStudioContent");
  const cfFilenameEl = $("cfFilename");
  const cfDescEl = $("cfDesc");
  const cfMetaEl = $("cfMeta");
  const btnToggleAnnot = $("btnToggleAnnot");
  const annotStateEl = $("annotState");
  const btnCopyCode = $("btnCopyCode");
  const copyBtnText = $("copyBtnText");
  const btnDownloadCode = $("btnDownloadCode");

  function highlightPython(code) {
    return tokenizePython(code, showAnnotations);
  }

  // Tokenize a Python source string to a flat text + tokens map, then re-emit.
  // This avoids regex-pass chaining where later passes see HTML emitted by
  // earlier passes (e.g. `class` inside `<span class="k">…</span>`).
  function tokenizePython(src, showAnns) {
    let text = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // 1. Pull out docstrings / strings / comments / annotation badges into
    //    sentinel placeholders so the kw/fn/num passes below don't re-touch them.
    //    Placeholder uses non-word, non-digit characters only.
    const tokens = []; // [{ open, body }]
    // Encode id as a base-26 letter sequence so the number regex never matches
    // digits inside the placeholder token.
    const id2s = (n) => String.fromCharCode(97 + (n % 26)) + (n >= 26 ? id2s(Math.floor(n / 26) - 1) : "");
    const place = (open, body) => {
        const id = tokens.length;
        tokens.push({ open, body });
        return `{{§${id2s(id)}§}}`;
    };
    text = text.replace(/("""[\s\S]*?""")/g, (m) => place('<span class="s">', m));
    text = text.replace(/("(\\.|[^"\\])*"|'(\\.|[^'\\])*')/g, (m) => place('<span class="s">', m));
    text = text.replace(/(#[^\n]*)/g, (m) => place('<span class="c">', m));
    if (showAnns) {
        text = text.replace(/(\(eq\.\s*[\d–\-,\s]+\))/g, (m) => place('<span class="cm-badge amber">', m));
        text = text.replace(/(§0\d)/g, (m) => place('<span class="cm-badge mint">', m));
    }

    // 2. Keywords / functions / numbers — apply on flat source. The {{§N§}}
    //    sentinel uses non-word characters so neither kw nor num regex touches it.
    const kws = ["class", "def", "return", "import", "from", "super", "self", "for", "in", "range", "if", "else", "is", "not", "None", "True", "False", "with", "as", "assert", "pass"];
    kws.forEach(kw => {
        const re = new RegExp(`\\b(${kw})\\b`, "g");
        text = text.replace(re, '<span class="k">$1</span>');
    });
    text = text.replace(/\.([a-zA-Z_]\w*)\(/g, '.<span class="f">$1</span>(');
    text = text.replace(/\bdef\s+([a-zA-Z_]\w*)/g, 'def <span class="f">$1</span>');
    text = text.replace(/\bclass\s+([a-zA-Z_]\w*)/g, 'class <span class="f">$1</span>');
    text = text.replace(/\b(\d+(\.\d+)?(e[+-]?\d+)?)\b/g, '<span class="n">$1</span>');

    // 3. Restore placeholders. Match the letter-encoded id of length 1+ and reverse.
    const s2id = (s) => {
        let n = 0;
        for (const ch of [...s].reverse()) n = n * 26 + (ch.charCodeAt(0) - 96);
        return n - 1;
    };
    text = text.replace(/\{\{§([a-z]+)§\}\}/g, (_m, idStr) => {
        const t = tokens[s2id(idStr)];
        return t.open + t.body + '</span>';
    });

    return text;
}

  function renderCodeStudio() {
    const file = CODE_FILES[activeFile];
    if (!file || !codeContentEl) return;

    if (cfFilenameEl) cfFilenameEl.textContent = file.name;
    if (cfDescEl) cfDescEl.textContent = file.desc;
    if (cfMetaEl) cfMetaEl.textContent = file.meta;

    codeContentEl.innerHTML = highlightPython(file.code);

    if (codeTabsContainer) {
      codeTabsContainer.querySelectorAll(".code-tab").forEach(tab => {
        const isCur = tab.dataset.file === activeFile;
        tab.classList.toggle("on", isCur);
        tab.setAttribute("aria-selected", String(isCur));
      });
    }
  }

  if (codeTabsContainer) {
    codeTabsContainer.addEventListener("click", e => {
      const tab = e.target.closest(".code-tab");
      if (!tab || tab.dataset.file === activeFile) return;
      activeFile = tab.dataset.file;
      renderCodeStudio();
    });
  }

  if (btnToggleAnnot) {
    btnToggleAnnot.addEventListener("click", () => {
      showAnnotations = !showAnnotations;
      if (annotStateEl) annotStateEl.textContent = showAnnotations ? "开" : "关";
      renderCodeStudio();
    });
  }

  if (btnCopyCode) {
    const origLabel = copyBtnText ? copyBtnText.textContent : "复制代码";
    btnCopyCode.addEventListener("click", () => {
      const file = CODE_FILES[activeFile];
      if (!file) return;
      navigator.clipboard.writeText(file.code).then(() => {
        if (copyBtnText) copyBtnText.textContent = "代码已复制 ✓";
        btnCopyCode.classList.add("primary");
        setTimeout(() => {
          if (copyBtnText) copyBtnText.textContent = origLabel;
          btnCopyCode.classList.remove("primary");
        }, 2000);
      }).catch(() => {
        if (copyBtnText) copyBtnText.textContent = "复制失败——请选择代码并手动复制。";
        setTimeout(() => { if (copyBtnText) copyBtnText.textContent = origLabel; }, 2600);
      });
    });
  }

  if (btnDownloadCode) {
    btnDownloadCode.addEventListener("click", () => {
      const file = CODE_FILES[activeFile];
      if (!file) return;
      const blob = new Blob([file.code], { type: "text/x-python;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  renderCodeStudio();

  // KaTeX render pass — runs once after scripts load.
  // ponytail: scope to <main> to skip nav/footer; delimiters match § math.
  // ponytail: also render bare <div class="math"> blocks as display math
  // (the page has 13 such blocks holding unwrapped TeX).
  function katexReady(){ return !!(window.renderMathInElement && window.katex); }
  function runKatex(){
    if(!katexReady()) return;
    const root = document.querySelector("main") || document.body;
    katexRender(root);                       // shared helper — same config as dynamic re-renders
    // Render bare .math divs as display math (skip ones already rendered
    // by auto-render, and skip the synthetic sample math block at .w-out).
    document.querySelectorAll(".math").forEach((el) => {
      if (el.querySelector(".katex")) return;
      if (el.closest(".w-out")) return;
      const tex = el.textContent.trim();
      if (!tex) return;
      try {
        window.katex.render(tex, el, { displayMode: true, throwOnError: false });
      } catch (_) {}
    });
  }
  if(katexReady()) runKatex();
  else window.addEventListener("load", runKatex);
})();
