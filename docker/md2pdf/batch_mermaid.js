#!/usr/bin/env node
/**
 * Extract all mermaid blocks from .md files under DOCS_DIR and render them
 * to PNG using a SINGLE Chromium session (one browser startup for all diagrams).
 *
 * Usage: batch_mermaid <docs_dir> <cache_dir>
 *
 * Writes: <cache_dir>/<rel_path_with_slashes_as_dashes>/<n>.png
 *   where n is the 1-based index of the mermaid block within the file.
 */

'use strict';
const fs   = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DOCS_DIR  = process.argv[2];
const CACHE_DIR = process.argv[3];
if (!DOCS_DIR || !CACHE_DIR) {
    console.error('Usage: batch_mermaid <docs_dir> <cache_dir>');
    process.exit(1);
}
fs.mkdirSync(CACHE_DIR, { recursive: true });

// ── 1. Walk docs dir and extract mermaid blocks ───────────────────────────────
function* walkMd(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) yield* walkMd(full);
        else if (entry.name.endsWith('.md')) yield full;
    }
}

const items = [];
for (const mdFile of walkMd(DOCS_DIR)) {
    const relPath  = path.relative(DOCS_DIR, mdFile);
    const fileKey  = relPath.replace(/\//g, '__').replace(/\.md$/, '');
    const lines    = fs.readFileSync(mdFile, 'utf8').split('\n');
    let inBlock = false, current = [], n = 0;
    for (const line of lines) {
        if (!inBlock && line.trim() === '```mermaid') { inBlock = true; current = []; }
        else if (inBlock && line.trim() === '```') {
            inBlock = false;
            n++;
            const outDir = path.join(CACHE_DIR, fileKey);
            fs.mkdirSync(outDir, { recursive: true });
            items.push({ code: current.join('\n'), out: path.join(outDir, `${n}.pdf`) });
            current = [];
        } else if (inBlock) { current.push(line); }
    }
}

const toRender = items.filter(x => !fs.existsSync(x.out));
console.log(`Mermaid: ${items.length} total, ${items.length - toRender.length} cached, ${toRender.length} to render`);
if (toRender.length === 0) process.exit(0);

// ── 2. Locate puppeteer ───────────────────────────────────────────────────────
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

// ── 3. Locate mermaid.min.js ──────────────────────────────────────────────────
const mermaidJsPath = execSync(
    'find /usr/local/lib/node_modules -name "mermaid.min.js" -path "*/mermaid/dist/*" 2>/dev/null | head -1'
).toString().trim();
if (!mermaidJsPath) { console.error('mermaid.min.js not found'); process.exit(1); }
const mermaidJs = fs.readFileSync(mermaidJsPath, 'utf8');

// ── 4. Render with ONE browser, N parallel pages (tabs) ──────────────────────
const CONCURRENCY = parseInt(process.env.MERMAID_JOBS || String(Math.min(8, toRender.length)));

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

async function renderOne(page, code, out, idx) {
    const svg = await page.evaluate(async (c, id) => {
        document.getElementById('g').innerHTML = '';
        try {
            const result = await mermaid.render(id, c);
            return result.svg || result;
        } catch(e) { return null; }
    }, code, `d${idx}`);

    if (!svg) return false;

    await page.evaluate((s) => { document.getElementById('g').innerHTML = s; }, svg);
    const el = await page.$('#g svg');
    if (!el) return false;

    const box = await el.boundingBox();
    await page.pdf({
        path: out,
        width:  Math.ceil(box.width  + 32) + 'px',
        height: Math.ceil(box.height + 32) + 'px',
        printBackground: true,
    });
    return true;
}

(async () => {
    const browser = await puppeteer.launch({
        executablePath: '/usr/bin/chromium-browser',
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
        headless: true,
    });

    const pages = await Promise.all(Array.from({ length: CONCURRENCY }, () => initPage(browser)));

    let ok = 0, failed = 0;
    const t0 = Date.now();
    const queue = toRender.map((item, i) => ({ ...item, idx: i }));

    await Promise.all(pages.map(async (page) => {
        while (true) {
            const item = queue.shift();
            if (!item) break;
            try {
                const success = await renderOne(page, item.code, item.out, item.idx);
                success ? ok++ : failed++;
            } catch(e) {
                console.error(`  fail [${item.idx + 1}]: ${e.message}`);
                failed++;
            }
            const done = ok + failed;
            if (done % 200 === 0) {
                const rate = (done / ((Date.now() - t0) / 1000)).toFixed(1);
                console.log(`  ${done}/${toRender.length}  (${rate} diagrams/s)`);
            }
        }
    }));

    await browser.close();
    const sec = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`Done: ${ok} ok, ${failed} failed in ${sec}s  [${CONCURRENCY} workers]`);
})();
