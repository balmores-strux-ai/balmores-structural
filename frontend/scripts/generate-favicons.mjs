/**
 * Generate raster favicons from public/logo.svg for Google Search / browsers.
 * Run: node scripts/generate-favicons.mjs
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import sharp from "sharp";
import toIco from "to-ico";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = path.join(root, "public");
const appDir = path.join(root, "app");
const svgPath = path.join(publicDir, "logo.svg");
const svg = fs.readFileSync(svgPath);

const render = (size) =>
  sharp(svg).resize(size, size, {
    fit: "contain",
    background: { r: 8, g: 10, b: 15, alpha: 1 },
  });

const pngSizes = [16, 32, 48, 180, 192, 512];

for (const size of pngSizes) {
  const name = size === 180 ? "apple-touch-icon.png" : `icon-${size}.png`;
  await render(size).png().toFile(path.join(publicDir, name));
  console.log(`wrote ${name}`);
}

const icoBuffers = await Promise.all([16, 32, 48].map((size) => render(size).png().toBuffer()));
const ico = await toIco(icoBuffers);
fs.writeFileSync(path.join(publicDir, "favicon.ico"), ico);
fs.writeFileSync(path.join(appDir, "favicon.ico"), ico);
fs.copyFileSync(path.join(publicDir, "apple-touch-icon.png"), path.join(appDir, "apple-icon.png"));

console.log(`wrote favicon.ico (${ico.length} bytes), app/favicon.ico, app/apple-icon.png`);
