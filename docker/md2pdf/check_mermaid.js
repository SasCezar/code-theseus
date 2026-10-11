#!/usr/bin/env node
/**
 * Diagram-compilation check: extract every ```mermaid block from
 * DOCS_DIR/<model>/<context>/c4/<repo_id>.md and try to render each with
 * mermaid.render() in a single headless Chromium session. Records
 * pass/fail per diagram (no PDF/PNG output — this only checks syntax
 * validity) to OUT_TSV.
 *
 * Usage: check_mermaid <docs_dir> <out_tsv>
 * Env:   MERMAID_JOBS  concurrent renderer tabs (default: 8)
 */

'use strict';
const fs   = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DOCS_DIR = process.argv[2];
const OUT_TSV  = process.argv[3];
if (!DOCS_DIR || !OUT_TSV) {
    console.error('Usage: check_mermaid <docs_dir> <out_tsv>');
    process.exit(1);
}

// ── 1. Walk docs dir (expects <model>/<context>/c4/<repo_id>.md) and extract mermaid blocks ──
function* walkMd(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) yield* walkMd(full);
        else if (entry.name.endsWith('.md')) yield full;
    }
}

const items = [];
for (const mdFile of walkMd(DOCS_DIR)) {
    const relPath = path.relative(DOCS_DIR, mdFile);
    const parts = relPath.split(path.sep); // model / context / task / repo_id.md
    const model = parts[0] || 'unknown';
    const context = parts[1] || 'unknown';
    const task = parts[2] || 'unknown';
    const repoId = path.basename(mdFile, '.md');

    const lines = fs.readFileSync(mdFile, 'utf8').split('\n');
    let inBlock = false, current = [], n = 0;
    for (const line of lines) {
        if (!inBlock && line.trim() === '```mermaid') { inBlock = true; current = []; }
        else if (inBlock && line.trim() === '```') {
            inBlock = false;
            n++;
            items.push({ model, context, task, repoId, diagramIdx: n, code: current.join('\n') });
            current = [];
        } else if (inBlock) { current.push(line); }
    }
}

console.log(`Found ${items.length} mermaid diagrams across ${new Set(items.map(i => i.model + '/' + i.repoId)).size} docs`);
if (items.length === 0) process.exit(0);

// ── 2. Locate puppeteer + mermaid.min.js ───────────────────────────────────────
function findModule(name) {
    for (const base of [
        '/usr/local/lib/node_modules',
        '/usr/local/lib/node_modules/@mermaid-js/mermaid-cli/node_modules',
        '/usr/lib/node_modules',
    ]) {
        const p = path.join(base, name);
        if (fs.existsSync(path.join(p, 'package.json'))) return p;
    }
    return null;
}
const puppeteerDir = findModule('puppeteer') || findModule('puppeteer-core');
if (!puppeteerDir) { console.error('puppeteer not found'); process.exit(1); }
const puppeteer = require(puppeteerDir);

const mermaidJsPath = execSync(
    'find /usr/local/lib/node_modules -name "mermaid.min.js" -path "*/mermaid/dist/*" 2>/dev/null | head -1'
).toString().trim();
if (!mermaidJsPath) { console.error('mermaid.min.js not found'); process.exit(1); }
const mermaidJs = fs.readFileSync(mermaidJsPath, 'utf8');

// ── 3. Render-check with ONE browser, N parallel pages (tabs) ──────────────────
const CONCURRENCY = parseInt(process.env.MERMAID_JOBS || String(Math.min(8, items.length)));

const PAGE_HTML = `<!DOCTYPE html><html><head>
<style>body{background:white;margin:0;padding:16px;font-family:sans-serif}#g{display:inline-block}</style>
</head><body><div id="g"></div>
<script>${mermaidJs}</script>
<script>mermaid.initialize({startOnLoad:false,securityLevel:'loose',theme:'default'})</script>
</body></html>`;

async function initPage(browser) {
    const p = await browser.newPage();
    await p.setContent(PAGE_HTML);
    return p;
}

async function checkOne(page, code, idx) {
    return page.evaluate(async (c, id) => {
        document.getElementById('g').innerHTML = '';
        try {
            const result = await mermaid.render(id, c);
            return { ok: !!(result && result.svg), error: null };
        } catch (e) {
            return { ok: false, error: String(e && e.message || e) };
        }
    }, code, `d${idx}`);
}

(async () => {
    const browser = await puppeteer.launch({
        executablePath: '/usr/bin/chromium-browser',
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
        headless: true,
    });

    const pages = await Promise.all(Array.from({ length: CONCURRENCY }, () => initPage(browser)));

    const results = new Array(items.length);
    let done = 0;
    const t0 = Date.now();
    const queue = items.map((item, i) => ({ ...item, idx: i }));

    await Promise.all(pages.map(async (page) => {
        while (true) {
            const item = queue.shift();
            if (!item) break;
            let outcome;
            try {
                outcome = await checkOne(page, item.code, item.idx);
            } catch (e) {
                outcome = { ok: false, error: String(e && e.message || e) };
            }
            results[item.idx] = { ...item, ...outcome };
            done++;
            if (done % 500 === 0) {
                const rate = (done / ((Date.now() - t0) / 1000)).toFixed(1);
                console.log(`  ${done}/${items.length}  (${rate} diagrams/s)`);
            }
        }
    }));

    await browser.close();

    const header = 'model\tcontext\ttask\trepo_id\tdiagram_idx\tstatus\terror\n';
    const tsv = results.map(r =>
        [r.model, r.context, r.task, r.repoId, r.diagramIdx, r.ok ? 'ok' : 'fail',
         (r.error || '').replace(/[\t\n]/g, ' ')].join('\t')
    ).join('\n');
    fs.writeFileSync(OUT_TSV, header + tsv + '\n');

    const ok = results.filter(r => r.ok).length;
    const sec = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`Done: ${ok}/${results.length} ok in ${sec}s [${CONCURRENCY} workers] -> ${OUT_TSV}`);
})();
