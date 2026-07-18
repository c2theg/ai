# Master Prompt — Integrate an Icon/Image Pack into a FastAPI + PWA App

> **How to use this file:** Paste the section below ("PROMPT") to a coding agent (Claude Code,
> etc.) working inside the target repository. Fill in the four `{{PLACEHOLDERS}}` first, or let the
> agent discover them. The prompt is framework-aware for **FastAPI single-page apps that ship a
> PWA**, but the workflow generalizes to any static-served web app (Flask, Express, Django, Vite).

---

## PROMPT

You are integrating a pre-generated **icon/image pack** into this application so it provides correct
icons, favicons, install icons, maskable icons, splash/launch screens, and platform tiles across
**iOS / iPadOS, Android, Windows, desktop browsers, and TV/large-screen OSes**.

Inputs (discover if not given):
- `{{ICON_PACK_ZIP}}` — path to the icon pack archive (e.g. `*-icon-pack.zip`).
- `{{APP_ROOT}}` — repository root.
- `{{STATIC_DIR}}` — directory served as static assets (FastAPI: the dir behind `app.mount("/static", StaticFiles(...))`; often `web/static` or `app/static`).
- `{{HTML_SHELL}}` — the HTML file whose `<head>` carries icon/manifest links (SPA shell, e.g. `index.html`).

### Core principle: conform to the app, don't conform the app to the pack

The pack uses its own filenames (`android-chrome-192x192.png`, `favicon-32x32.png`, …). The app
almost certainly already has an icon naming convention and a **manifest that may be richer than the
pack's** (shortcuts, `share_target`, `file_handlers`, categories). **Do not blindly overwrite the
app's manifest or HTML with the pack's minimal versions.**

Prefer this strategy: **swap the new artwork into the app's existing in-use filenames.** Every
existing reference (HTML head, manifest, service worker, favicon route) then picks up the new
branding with **zero reference changes and no broken links.** Only add new files + new references
for platforms the app does not yet support (e.g. Windows tiles, TV banners).

### Phase 0 — Inventory (read before writing anything)

1. Extract the pack to a temp dir; list every file. Note each PNG's true pixel dimensions (`sips -g pixelWidth -g pixelHeight f.png`, or `identify`, or PIL). Read any `README`, `html-head-snippet.html`, `manifest.webmanifest`, `browserconfig.xml` the pack ships — they document intent.
2. Find the app's current icon wiring. Grep the shell, manifest, and service worker:
   ```
   grep -rniE "icon|favicon|manifest|apple-touch|mask|browserconfig|mstile|theme-color|splash|startup-image|tile" {{HTML_SHELL}} {{STATIC_DIR}}/*.js {{STATIC_DIR}}/*.json
   ```
   Also grep route handlers for `favicon.ico` / `manifest` endpoints (FastAPI apps often serve `/favicon.ico` and `/manifest.json` from explicit routes, not just the static mount).
3. Identify the canonical static mount and whether the app runs **behind a subpath/reverse proxy**
   (look for `X-Forwarded-Prefix`, `root_path`, injected `<base href>`). If so, **all asset paths you
   add must be relative** (`static/icons/x.png`, not `/static/icons/x.png`) so they resolve under the
   prefix.
4. Note the app's brand/theme color (`<meta name="theme-color">`, manifest `theme_color`/`background_color`). Reuse it for new tiles/splash so everything is visually consistent.

### Phase 1 — Map pack → app filenames

Build an explicit mapping table from pack files to the app's existing icon filenames, matching by
pixel size. Copy pack artwork onto the existing names (overwrite). Typical mapping:

| App file (keep name) | Pack source | Size |
|---|---|---|
| `favicon.ico` | `favicon.ico` | multi |
| `favicon-16.png` / `icon-16.png` | `favicon-16x16.png` | 16 |
| `favicon-32.png` / `icon-32.png` | `favicon-32x32.png` | 32 |
| `icon-72/96/144/192/384/512.png` | `android-chrome-*.png` | matched |
| `icon-128.png` | `favicon-128x128.png` | 128 |
| `icon-152/167/180.png`, `apple-touch-icon.png` | `apple-touch-icon-*.png` | matched |
| `maskable-192/512.png` | `maskable-icon-*.png` | matched |

**Leave untouched** assets the pack doesn't cover: a monochrome **notification `badge-*.png`** (must
be a flat silhouette, not the full-color icon) and any **`icon.svg`** (packs usually omit a Safari
pinned-tab/monochrome SVG by design). Verify every replaced file's dimensions afterward.

### Phase 2 — Wire each platform

Add only what's missing; keep existing links if they already point at the now-updated files.

**Desktop browsers / favicons** (`<head>`):
```html
<link rel="icon" href="favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="static/icons/favicon-32.png">
<link rel="icon" type="image/png" sizes="16x16" href="static/icons/favicon-16.png">
```
If a route serves `/favicon.ico`, point it at the updated `static/icons/favicon.ico`.

**iOS / iPadOS** (`<head>` + manifest):
```html
<link rel="apple-touch-icon" href="static/icons/apple-touch-icon.png"><!-- 180×180 -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="{{APP_SHORT_NAME}}">
```
iOS ignores manifest icons for the home-screen icon — `apple-touch-icon` is mandatory. iOS launch
screens use `apple-touch-startup-image` links with exact device media queries (see Phase 3).

**Android / Chrome / installable PWA** (manifest `icons[]`): include `any` purpose at 192 + 512 and
**separate `maskable` entries** at 192 + 512 (`"purpose": "maskable"`). Maskable icons need ~20% safe
padding — use the pack's maskable variants, never reuse the plain icon as maskable. Keep the app's
existing `theme_color`, `background_color`, `shortcuts`, `share_target`, `file_handlers`.

**Windows tiles** (add `browserconfig.xml` + `<head>` metas):
```html
<meta name="msapplication-config" content="static/browserconfig.xml">
<meta name="msapplication-TileColor" content="{{THEME_COLOR}}">
<meta name="msapplication-TileImage" content="static/icons/mstile-144x144.png">
```
`browserconfig.xml` — place it next to the icons or in the static root and use paths **relative to
the browserconfig file's own URL** (the browser resolves tile `src` relative to that file, not the
page), so it survives a subpath proxy:
```xml
<?xml version="1.0" encoding="utf-8"?>
<browserconfig><msapplication><tile>
  <square70x70logo src="icons/mstile-70x70.png"/>
  <square150x150logo src="icons/mstile-150x150.png"/>
  <wide310x150logo src="icons/mstile-310x150.png"/>
  <square310x310logo src="icons/mstile-310x310.png"/>
  <TileColor>{{THEME_COLOR}}</TileColor>
</tile></msapplication></browserconfig>
```

**TV / large-screen OSes** (set expectations honestly):
- **Android TV / Google TV** installed PWAs and webOS (LG) / Tizen (Samsung) use the **manifest
  `icons[]`** — ensuring a clean **512×512** (and 1024 if available) covers them. There is no
  standard *web-manifest* field for the Android TV launcher **banner** (320×180); that banner is an
  Android *native* resource (`android:banner` in the APK/TWA), so only set it if a TWA/native wrapper
  exists — generate a 320×180 landscape banner from the icon + wordmark and place it in the wrapper's
  `res/drawable`.
- **Apple TV (tvOS)** uses **layered** App Store / Top Shelf image stacks (e.g. 1280×768 layers) in a
  native app only — not reachable from a web PWA. Note this as native-only; if a tvOS wrapper exists,
  the pack's flat icon must be split into parallax layers separately.
- Net effect for a pure PWA: a high-res square manifest icon is the deliverable; flag native TV
  banners/layers as out-of-scope unless a native wrapper is present.

### Phase 3 — Splash / launch screens (generate if the pack omits them)

Most packs ship **no** splash screens. iOS needs them as `apple-touch-startup-image` links with
device-specific media queries; Android/desktop synthesize splash from manifest `background_color` +
icon + name (no images needed).

If splash images are missing, **generate** them from the largest icon (1024) onto a solid field:
- Tooling preference: ImageMagick (`magick`) or PIL if available, else macOS `sips`:
  ```bash
  sips -s format png icon-1024.png --resampleHeightWidth 600 600 --out /tmp/logo.png
  sips /tmp/logo.png --padToHeightWidth 2532 1170 --padColor 07070E --out splash-1170x2532.png
  ```
- **Match the field color to the icon's edge/background** so a rounded app-tile doesn't leave a
  visible square seam. Detect it: downsample the icon to 3×3 and read the corner pixels (the corners
  of an app-icon tile reveal the true background; a centered crop samples the logo, not the field).
  Use that color as `--padColor`. For a perfectly seamless full-bleed splash you need a
  transparent-background glyph or SVG — if the pack only has the rounded tile, accept the faint edge
  or request a transparent glyph.
- Cover the common device sizes the app already targets; reuse the app's existing
  `apple-touch-startup-image` media queries and just overwrite the image files so links keep working.

### Phase 4 — Cache invalidation (critical for PWAs)

A service worker will serve **stale cached icons/splash/shell** forever unless its cache key changes.
After editing icons, the manifest, the HTML shell, or `browserconfig.xml`:
- Bump the service worker `CACHE_VERSION` (or equivalent precache revision/`__WB_MANIFEST` hash).
- If the SW has an explicit precache list (`SHELL_ASSETS`), ensure the newly-referenced critical
  icons are included (typically 192 + 512 + maskable + favicon); splash/browserconfig can stay
  network-fetched.

### Phase 5 — Verify & report

1. Confirm every replaced/added PNG has the exact dimensions its filename/role claims.
2. Confirm every `<head>` link, manifest `src`, `browserconfig` `src`, and favicon route resolves to
   a file that exists (no 404s) — and that paths are relative if behind a subpath proxy.
3. Validate JSON/XML: `python3 -m json.tool manifest.json`, and that `browserconfig.xml` is well-formed.
4. Report a concise summary: which files were overwritten, which were added, which references changed,
   the new cache version, and any **platform left as native-only** (TV banners/layers, monochrome
   SVG/badge) with the reason.

### Decisions & gotchas (carry these every time)

- **Preserve the richer artifact.** Never downgrade an app manifest that has shortcuts/share_target/
  file_handlers to the pack's stub manifest.
- **Relative paths behind proxies.** Subpath-hosted apps break on leading-slash asset paths.
- **`browserconfig` tile `src` is relative to the XML file**, not the page — a subtle, common bug.
- **Maskable ≠ regular icon.** Keep them as distinct manifest entries with `purpose: "maskable"`.
- **Badge and pinned-tab assets are monochrome** — don't replace them with the full-color icon.
- **Theme/tile/splash colors should match** the app's existing brand color, not the pack's defaults,
  unless told otherwise.
- **Don't forget the SW cache bump** — it's the #1 reason "the new icons didn't show up."
