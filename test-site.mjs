import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("./css/styles.css", import.meta.url), "utf8");
const app = readFileSync(new URL("./js/app.js", import.meta.url), "utf8");

test("loads the canonical stylesheet", () => {
  assert.match(html, /<link\s+rel="stylesheet"\s+href="css\/styles\.css">/);
});

test("index.html is byte-identical to the concatenated _parts", () => {
  const parts = readdirSync(new URL("./_parts/", import.meta.url))
    .filter(f => /^0.*\.html$/.test(f))
    .sort()
    .map(f => readFileSync(new URL(`./_parts/${f}`, import.meta.url), "utf8"));
  assert.equal(parts.join(""), html);
});

test("the page declares Simplified Chinese and localizes all chapter navigation", () => {
  assert.match(html, /<html lang="zh-CN">/);
  assert.match(html, /<title>[^<]*[\u3400-\u9fff][^<]*<\/title>/);
  const navigation = [...html.matchAll(/<a href="#([^"]+)" data-num="(\d+)" data-label="([^"]+)">/g)];
  assert.equal(navigation.length, 15);
  for (const [, id, , label] of navigation) {
    assert.match(label, /[\u3400-\u9fff]/, `${id} needs a Chinese navigation label`);
    assert.ok(html.includes(`<section id="${id}">`), `${id} must still target its chapter`);
  }
});

test("Chinese heading anchors remain meaningful and distinguishable", () => {
  const declaration = app.match(/^const slug = .+;$/m);
  assert.ok(declaration, "heading slug helper must exist");
  const context = {};
  vm.runInNewContext(`${declaration[0]}\nfirst = slug("门控注意力"); second = slug("混合骨干网络");`, context);
  assert.equal(context.first, "门控注意力");
  assert.equal(context.second, "混合骨干网络");
  assert.notEqual(context.first, context.second);
});

test("chapter prose does not regress to untranslated English paragraphs", () => {
  for (const [, paragraph] of html.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/g)) {
    const prose = paragraph.replace(/\$\$?[\s\S]*?\$\$?/g, "").trim();
    if (/[A-Za-z]{3,}\s+[A-Za-z]{3,}/.test(prose)) {
      assert.match(prose, /[\u3400-\u9fff]/, `untranslated paragraph: ${prose.slice(0, 120)}`);
    }
  }
});

test("localization preserves the original license and attribution", () => {
  const license = readFileSync(new URL("./LICENSE", import.meta.url), "utf8");
  assert.match(license, /MIT License/);
  assert.match(license, /Copyright \(c\) 2026 Atandra Bharati/);
  const readme = readFileSync(new URL("./README.md", import.meta.url), "utf8");
  assert.match(readme, /https:\/\/github\.com\/atandra2000/);
  assert.match(readme, /Atandra Bharati/);
});

test("CFG.total × 2 lands at 55.17 GB BF16 (decimal GB)", () => {
  const start = app.indexOf("const CFG = {");
  const end = app.indexOf("const fmtP", start);
  assert.notEqual(start, -1, "CFG block must exist");
  assert.notEqual(end, -1, "CFG block must close before fmtP");

  const context = {};
  vm.runInNewContext(`${app.slice(start, end)}\ncfgOut = CFG;`, context);

  const gbBF16 = (context.cfgOut.total * 2) / 1e9;
  assert.ok(
    Math.abs(gbBF16 - 55.17) < 0.1,
    `expected ~55.17 GB BF16, got ${gbBF16}`
  );
});

test("educational content is never hidden behind a reveal effect", () => {
  assert.doesNotMatch(css, /\.reveal\s*\{[^}]*opacity\s*:\s*0/s);
  assert.doesNotMatch(app, /classList\.add\(["']reveal["']\)/);
});

test("the hidden back-to-top control stays out of the focus order", () => {
  assert.match(app, /btn\.tabIndex\s*=\s*visible\s*\?\s*0\s*:\s*-1/);
  assert.match(app, /btn\.setAttribute\("aria-hidden",\s*String\(!visible\)\)/);
});

test("keyboard focus reveals hover-only actions", () => {
  assert.match(css, /h2 \.anchor:focus-visible,h3 \.anchor:focus-visible\s*\{opacity:1;\}/);
  assert.match(css, /\.code-copy:focus-visible\s*\{opacity:1;\}/);
});

test("Python highlighting survives more than 52 protected tokens", () => {
  const start = app.indexOf("  function tokenizePython(");
  const end = app.indexOf("\n\n  function renderCodeStudio", start);
  assert.notEqual(start, -1, "tokenizePython must exist");
  assert.notEqual(end, -1, "renderCodeStudio must follow tokenizePython");

  const context = { sample: Array.from({ length: 60 }, (_, i) => `\"token ${i}\"`).join("\n") };
  vm.runInNewContext(`${app.slice(start, end)}\nresult = tokenizePython(sample, true);`, context);

  assert.match(context.result, /token 59/);
});

test("quantization bit styles stay scoped away from the hero register", () => {
  assert.doesNotMatch(css, /\n\.bit\s*\{/);
  assert.match(css, /\.qfc-bits \.bit\s*\{/);
});

test("the mobile hero keeps the page gutter", () => {
  assert.match(css, /\.hero\s*\{padding:var\(--s-7\) var\(--s-4\) var\(--s-6\);\}/);
});

test("the reading column contains wide technical artifacts", () => {
  assert.match(css, /main\.wrap\s*\{overflow-x:clip;\}/);
});

test("mobile and tablet controls keep accessible touch targets", () => {
  assert.match(css, /@media \(max-width:860px\)\{\n  button:not\(\.layer-cell\),select,input:not\(\[type=range\]\),\.nav-grid a\{min-height:44px;min-width:44px;\}/);
});
