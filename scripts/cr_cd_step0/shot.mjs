// node shot.mjs <playwright module dir> <page.html> <out.png>
const [, , pwDir, page, out] = process.argv;
const { chromium } = await import(`${pwDir}/index.mjs`);
const browser = await chromium.launch();
const p = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
await p.goto(`file://${page}`);
await p.waitForFunction(() => window.__ready === true);
await p.waitForTimeout(500);
await p.screenshot({ path: out, fullPage: true });
await browser.close();
