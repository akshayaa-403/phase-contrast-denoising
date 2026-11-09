# Redesign the Phase-Contrast Clean-Up browser demo

## Your task

Redesign the browser demo in `docs/` into a workbench that a visitor understands
in ten seconds and can tune for ten minutes.

The demo is a static GitHub Pages site. It runs the classical half of the
pipeline live in JavaScript: Difference-of-Gaussians (DoG) halo suppression,
then CLAHE. It shows a precomputed residual U-Net output beside that, and it
reports image-quality metrics. The pipeline code is sound and verified. The
problems are layout, comparison, the honesty of the data shown, and the missing
workflow.

Work in the phases in section 9. After each phase, run that phase's checks and
continue. Stop only for something on the ask-first list in section 10.

---

## 1. Read these first

Before planning, read:

- `docs/index.html`, `docs/app.js`, `docs/cv.js`, `docs/styles.css`,
  `docs/theme.js`: the current demo.
- `docs/samples/manifest.json` and `scripts/make_demo_assets.py`: how samples
  and reference metrics reach the page.
- `scripts/verify_web_pipeline.py`: the parity check between `cv.js` and
  OpenCV. It must keep passing.
- `src/preprocessor.py`, `src/pipeline.py`, `src/report.py` (`_otsu_dice`),
  and `src/dataset.py` (`build_dataloaders` and its seed-42 held-out split).
- `README.md`, from "Browser demo" through "Accessibility". It explains why
  the current page is built the way it is.
- `configs/default_config.yaml`: the parameter defaults.

Then serve the current page with `python -m http.server -d docs 8000` and use
it at desktop and phone widths before you change anything.

---

## 2. Who visits, and what they need

1. **A hiring manager or reviewer arriving from the author's portfolio** (the
   case study links here). They stay under a minute. Without scrolling at
   1280×720, they need to see a large before/after of a microscope frame, one
   sentence on what the method did, and one honest headline number.
2. **A technical reviewer or biomedical engineer.** They know Python but are
   not computer-vision specialists, and they stay about ten minutes. They want
   to tune each parameter on a sample or on their own micrograph, inspect pixels
   up close, see whether segmentation improves, and take the settings back to
   the Python code.

Every design decision should serve one of these two visitors. Where they
conflict, the first screen serves visitor 1, and everything below the fold or
behind a click serves visitor 2.

---

## 3. What is wrong with the current page

These problems are the reason for the redesign. Fix all of them.

1. **The image scrolls away while you drag a slider.** The controls sit below
   the images and metrics, so on a laptop, and always on a phone, the picture is
   off-screen while you drag.
2. **The images are too small to show the halo.** There are five panels about
   190 px wide and no zoom, and the halo is only a few pixels wide.
3. **There is no direct before/after.** To compare, the eye has to jump between
   separate panels.
4. **The CLAHE clip slider almost never does anything, and the page does not
   say so.** `cv.js` and `src/preprocessor.py` both quantise to uint16 before
   CLAHE. OpenCV turns the clip limit into a pixel ceiling per grey level:
   `max(round(clip × tileArea / 65536), 1)`. On a 256 px sample with the default
   8×8 tiles, that ceiling is 1 for every value the slider allows, so the output
   is bit-identical. The ceiling only rises above 1 with very coarse tiles or
   large uploads. The author's case study documents this. It is a known bug and
   must not be hidden.
5. **Four of the six samples are training frames, and nothing says so.**
   `make_demo_assets.py` takes the first N files. The fair test set is the 20
   frames that `build_dataloaders` holds out (a seed-42 permutation, 20% of 100
   frames).
6. **The project's real result is not on the page.** The U-Net raises held-out
   PSNR a lot, but segmentation is the point. Mean Otsu Dice barely moves, yet
   the worst frame stops failing. The page shows no segmentation at all.
7. **The metrics table gives scores with no explanation.** The classical path's
   PSNR falls below the input's, which looks like a failure. The explanation is
   only in the README.
8. **Setting σ₂ ≤ σ₁ freezes the output.** The only sign is a warning below the
   sliders, which may be off-screen.
9. **Results cannot leave the page.** There is no image export, no way to carry
   tuned parameters back to `configs/default_config.yaml` or the CLI, and no
   shareable link.
10. **Compute runs on the main thread, and input formats are limited.** A large
    upload freezes the tab. Only formats the browser can decode will load, so
    the 16-bit TIFFs that microscopes produce are rejected or flattened to 8-bit.

---

## 4. Keep these. They are right.

- A static site with no build step, deployed from `docs/` by
  `.github/workflows/pages.yml`. Use relative URLs only, because the site is
  served under `/phase-contrast-denoising/`.
- `cv.js` as a verified port of `src/preprocessor.py`, UMD-wrapped so that Node
  and the verify script can load it.
- Samples driven by the manifest, and the "No sample images published yet"
  fallback.
- Uploads never leave the tab. No analytics, no trackers, and no third-party
  requests other than pinned libraries that earn their place (section 8).
- The theme behaviour in `theme.js`: follow the OS until the visitor chooses.
- The accessibility work already done: skip link, landmarks, visible
  `:focus-visible` rings, labelled controls with live `<output>` read-outs,
  table headers, an `aria-live` status, `prefers-reduced-motion`,
  `prefers-contrast: more`, 24 px minimum targets, and WCAG AA contrast in both
  themes.
- Nearest-neighbour rendering when magnifying scientific images, because
  smoothing invents detail that is not in the data. When minifying, average the
  pixels instead, so thumbnails do not alias.
- Honest labels. The U-Net output is precomputed and says so. Uploads have no
  ground truth, so they switch to no-reference measures that state what they
  can and cannot claim.
- British spelling, as in the README.

---

## 5. Requirements

Priorities:

- **Must (M)** ships in this pass.
- **Should (S)** ships once every Must is done and verified.
- **Could (C)** is only for spare time, and C3 also needs the owner's approval.

### 5.1 Layout

**M1 Workbench.** At 1024 px and wider, put a large sticky viewer on the left,
about two-thirds of the width. On the right, a scrolling sidebar holds these
sections in order: Image, 1 · Halo suppression (DoG), 2 · Local contrast
(CLAHE), Overlays, Measured, Export. At both 1280×720 and 1024×700, the viewer
and whichever control is being dragged must be visible together.

Below 1024 px, the viewer sits at the top and stays visible (sticky, about
45–55 svh). The sidebar sections become tabs or an accordion beneath it. On a
375×812 phone, a visitor dragging α must see the image change.

```
Desktop (≥1024 px)
┌───────────────────────────────────────────────────────────────────────┐
│ Phase-Contrast Clean-Up · one-sentence purpose            ◐  Source ↗ │
│ Held-out frames: <headline line read from the manifest>                │
├────────────────────────────────────────────────┬──────────────────────┤
│ [Input ▾]  live/precomputed ┃ [Hybrid U-Net ▾]  │ IMAGE                │
│                             ┃                   │ ▢▢▢▢▢▢▢▢▢▢ Upload…   │
│            viewer           ┃ ← divider         │ 1 · HALO (DoG)       │
│                             ┃                   │ σ₁ ──●────── [1.0]   │
│                             ┃                   │ σ₂ ─────●─── [8.0]   │
│                                                 │ α  ────●──── [0.60]  │
│ − 1:1 + fit │ split/grid │ hold for input │ mask│ 2 · CONTRAST (CLAHE) │
│ x 112  y 87 · input 0.412 · hybrid 0.091        │ OVERLAYS · MEASURED  │
│ ── line profile ─────────────────────────────── │ EXPORT               │
└────────────────────────────────────────────────┴──────────────────────┘

Phone (<1024 px)
┌─────────────────────────┐
│ title · one line        │
│ viewer (sticky)         │
│ toolbar                 │
├─────────────────────────┤
│ [Image][Tune][Measure]  │
│ [Export]                │
│ active tab content      │
└─────────────────────────┘
```

**M2 First screen.** At 1280×720, without scrolling, a visitor sees:

- the title;
- one sentence on what the page does (cut the current four-line tagline);
- the viewer showing a held-out frame;
- one headline line read from the manifest's held-out summary;
- the first controls.

An example headline is "Held-out frames: PSNR 11.46 → 20.26 dB with the U-Net;
lowest Otsu Dice 0.556 → 0.808". These figures come from the author's case
study, so use them to sanity-check your script. The page must read its numbers
from the manifest.

### 5.2 Viewer

**M3 Split comparison.** Show two stages in one frame, divided by a draggable
vertical line. Each side has its own stage picker. The stages are Input, After
DoG, Classical (DoG + CLAHE, live), Hybrid U-Net (precomputed), and Ground
truth (samples only). A label on each side names the stage and says whether it
is live or precomputed.

The divider has `role="slider"`. Arrow keys move it 1%, Shift+arrow 10%, and
Home/End jump to the edges. It can be dragged by touch, and it keeps its
position when the sample changes. The default view is Input on the left and
Hybrid U-Net on the right.

**M4 Grid toggle.** A grid view shows every stage side by side at the same zoom,
like the current overview. Clicking a stage in the grid puts it on the right
side of the split.

**M5 Zoom and pan,** kept in sync across every visible stage:

- Wheel or pinch zooms around the cursor, and dragging pans.
- Buttons: −, 1:1, +, fit. The current zoom level is shown.
- When the viewer has focus, `+`, `−`, `0` (fit) and `1` (1:1) set the zoom,
  and the arrow keys pan.
- Double-click toggles between fit and 2×.
- When magnifying, zoom in whole-number steps so pixels stay square.

**M6 Hold to compare.** While the visitor holds a button, or holds `\` with the
viewer focused, Input fills the whole frame.

**M7 The image being changed is always visible.** If a DoG or CLAHE control
moves while neither side shows a live stage, switch the right side to Classical
(live) and say so in the status region. This rule matters more than the
mechanism. If you find a cleaner way to meet it, use that.

**S1 Pixel read-out.** Hovering the viewer, or moving a keyboard crosshair,
shows x, y and that pixel's value in every visible stage, from 0 to 1 to three
decimal places.

**S2 Line profile.** Add a draggable horizontal scan line. Under the viewer, a
small inline-SVG chart plots intensity along that row for Input, the live stage,
Hybrid and Truth. This is the clearest way to see a halo: it shows up as an
overshoot on either side of each cell edge, and suppression flattens it. Tell
the lines apart by dash pattern and direct labels as well as by colour.

**S3 Derived views,** added to the stage pickers:

- "Removed by DoG": the input minus the DoG output, shown on a diverging colour
  map with mid-grey as 0.
- "Error vs truth": the absolute difference between a stage and the truth.
  Samples only.
- "U-Net residual": what the network subtracted. Precompute it in
  `make_demo_assets.py` as a PNG with mid-grey as 0. It is only defined for the
  classical output at the config defaults, so label it that way.

**S4 Scale guide.** While σ₁ or σ₂ is being dragged, draw circles of radius σ₁
and σ₂ over the image, so the visitor can compare the blur scales with the
sizes of the cells and the halo. Hide the circles after release, and keep them
static under reduced motion.

**S5 Pin and compare.** A "Pin" button stores the current classical output with
its parameters as an extra stage, for example "Pinned · σ₂ 12, α 0.8". It can
then be chosen in either picker, so two parameter settings can be compared
directly.

**C1 Lookup tables.** Offer grayscale (the default) and one perceptually uniform
false-colour map. They change the display only. The metrics never change.

**C2 Histogram** of the right-hand stage. When CLAHE is involved, draw the clip
ceiling on it.

### 5.3 Controls

**M8 Grouped by stage.** Under each group heading, one plain-language line says
what the stage does. For example: "Blur at two scales and subtract part of the
difference, flattening the bright ring around each cell."

Each control has:

- a label giving the symbol, its meaning and its unit ("σ₂ · halo width, px");
- a range slider and a number input, kept in sync;
- the config default marked on the track;
- its own reset;
- a step ten times larger with Shift+arrow.

Values load from `manifest.defaults` and reset to it.

**M9 σ₁ and σ₂ can never become invalid.** σ₂'s minimum follows σ₁ plus one
step. Raising σ₁ past σ₂ pushes σ₂ up with it, and a note beside σ₂ says so.
The output never freezes.

**M10 CLAHE precision.** Add a precision control with two options:

- "16-bit (as shipped)", the default. It matches the Python code and the input
  the U-Net was trained on.
- "8-bit (fixed)".

Under the clip control, show the ceiling the code computes, in pixels per grey
level, updating live. When that ceiling is 1 across the whole clip range at the
current tile size and image size, mark the control inactive. Add one sentence
explaining why, and link to the README section. Calculate this rather than
hard-coding it, because large uploads in 16-bit do get a working ceiling.

When 8-bit is selected, add a note by the Hybrid stage that the U-Net was
trained on 16-bit output.

**M11 Tiles.** Keep one control for a square grid.

**M12** Keep "Reset all to config defaults".

### 5.4 Measured

**M13 Per-stage read-out.** Replace the table with a compact read-out for Input,
Classical (live) and Hybrid. Give PSNR (dB), SSIM and Otsu Dice for each. Show
each stage's change from Input as a signed number plus a word, for example
"+8.8 dB, better", so the meaning never depends on colour alone. Highlight the
live row.

**M14 Port `_otsu_dice`.** Port `_otsu_dice` from `src/report.py` to `cv.js`
exactly: the same uint8 conversion, the same Otsu threshold and the same mask
comparison. Add it to the parity check.

**M15 Mask overlays.** Add toggles on the viewer for:

- the Otsu mask outline of the right-hand stage;
- the truth mask as a dashed outline.

The outlines must reach 3:1 contrast against the viewer background, and they
must differ in dash pattern as well as colour.

**M16 Short explanations.** Add one-sentence explanations that expand to a
little more. Cover:

- why Classical PSNR falls below Input's (CLAHE rewrites the tone curve, and
  PSNR compares absolute intensities);
- what Dice measures;
- why uploads get no-reference measures instead.

Take the wording from the README and keep each one under 40 words.

**M17 Python reference values.** Show "Measured in Python" values only while
every parameter equals its default for the active precision. Otherwise show
"sliders moved" instead of a number that no longer applies. The manifest holds
reference values for both precisions (section 9, Phase 1).

**M18 Held-out summary.** Below the fold, show a block with the number of
frames and the mean and worst PSNR and Dice for raw, classical and hybrid, all
read from the manifest.

**M19 Uploads.** For uploads, keep the no-reference measures (edge energy and
background variance) and add the Otsu foreground fraction before and after.
Keep the existing caveat.

**M20 Screen-reader announcements.** Numbers update visually while a control is
dragged. Announce them to screen readers once the drag ends, not on every frame.

**S6 Held-out dot plot.** Draw one dot per held-out frame in inline SVG, with a
line joining that frame's raw and hybrid values. Ring the current sample, and
let a click on a dot open that frame.

### 5.5 Samples and input

**M21 Held-out samples.** Use the 20 held-out frames as the samples, chosen by
the same code the trainer uses:

1. Move the split out of `build_dataloaders` into its own function, for example
   `split_pairs(pairs, val_split, seed)`.
2. Call that function from both `build_dataloaders` and `make_demo_assets.py`,
   so the two can never drift apart.
3. Training must not change. Add an assertion that the new function reproduces
   the old indices.

Ship all 20 frames, so that the held-out summary covers exactly the frames a
visitor can open. Measure the added weight; expect about 2–3 MB.

**M22 Sample tiles.** Each sample tile shows a thumbnail and its raw → hybrid
Dice. The tiles can be sorted with "lowest raw Dice first". The page opens on
the frame with the lowest raw Dice, labelled "worst raw frame" so that the
choice is visible. Find it from the manifest; do not hard-code it.

**M23 Input.** Accept images by drag and drop onto the viewer, by paste from the
clipboard, and through a file picker. Unsupported files get a clear inline
error. Convert to grayscale as the page does now.

**M24 Large images.** If an image's long edge is over 2048 px, offer to
downscale it or to process a centre crop. While a slider is moving, compute a
preview no larger than 512 px, then compute the full image on release. Show
progress for anything that takes over 200 ms.

**S7 16-bit images.** Decode 16-bit TIFF (uncompressed and LZW, single channel)
and 16-bit PNG, keeping 16-bit precision into the pipeline. Use a small pinned
library, loaded only when such a file arrives (section 8).

### 5.6 Compute

**M25 Web Worker.** Run the pipeline in a Web Worker. Load `cv.js` with
`importScripts('cv.js')`; the UMD wrapper already sets `self.PhaseCV`.

- Transfer buffers instead of copying them.
- Drop stale jobs: when newer parameters arrive, discard the older result.
- On a 256 px sample, the main thread must have no long task over 50 ms while a
  slider is dragged.

### 5.7 Export and sharing

**M26 Image export.** Download as PNG:

- the current right-hand stage at full resolution (8-bit);
- a side-by-side figure of the left and right stages, with labels and the
  parameters.

**M27 Copy parameters.** Copy the parameters in two forms:

- a YAML block using the exact keys under `preprocessor:` in
  `configs/default_config.yaml`;
- a ready-to-run `python main.py …` command.

If 8-bit precision is selected, add a note that the Python code does not
support it yet.

**M28 Shareable link.** Store the current view in the URL hash so a reviewer can
share it exactly: sample id, parameters, precision, both stages, divider
position and overlays. Invalid hash values fall back to the defaults. Uploads
are not encoded.

### 5.8 Stretch: needs the owner's approval

**C3 U-Net in the browser.**

1. Export the fp16 checkpoint to ONNX.
2. Run it with onnxruntime-web, using WebGPU with a WASM fallback.
3. Start it only from an opt-in button that states the download size.
4. Verify it against PyTorch at 40 dB or better on all 20 frames.
5. Feed it only the 16-bit classical output.

Proceed only if it runs in 3 s or less on a laptop with WebGPU and the owner
approves committing a file of about 60 MB. Otherwise skip it and report what
you measured.

---

## 6. Visual design direction

- **Character:** a calm scientific instrument. The image is the loudest thing on
  the page, and the rest of the interface stays quiet. No gradients, glass
  effects, glows or decorative illustration.
- **Viewer background:** a neutral dark grey in both themes, with its own tokens
  that do not change with the theme. Micrographs are read on dark, and a white
  surround changes how contrast is perceived.
- **Type:** match the author's portfolio, which visitors arrive from. Use IBM
  Plex Mono for labels, numbers, controls and code, and Libre Caslon Text for
  the few prose sentences. Load them from Google Fonts with `display=swap`,
  only in the weights used. Use tabular numerals for every number. If the fonts
  break the page-weight budget, keep system fonts and say so.
- **Colour:** keep the existing token system and the teal accent.
  - Every new colour is a token with a light and a dark value.
  - Every new colour is measured against AA: 4.5:1 for text, 3:1 for UI marks
    and overlays.
  - Give each stage one identity colour and use it everywhere: labels, overlays,
    the line profile and the dot plot. Always pair it with a second cue, such as
    a dash pattern or a label.
- **Spacing:** use a 4/8 px scale and one family of corner radii. Separate
  panels with space and hairlines, not cards nested inside cards.
- **Motion:** no animation when the image updates. UI state changes may
  transition for 150 ms or less, and not at all under reduced motion.

---

## 7. Writing on the page

- Results first, prose thin: the title and one sentence at the top, then the
  work.
- Every number on the page is either calculated live in the browser or read
  from the manifest that `make_demo_assets.py` writes. Never type a number into
  the HTML.
- Do not use the example "insight" sentence in `CLAUDE.md` §8.3 ("95% fidelity …
  98% accuracy"). It is a placeholder, not a measurement.
- Label where every image and number comes from: "live", "precomputed",
  "held-out", or "your image, no ground truth".
- Be exact about the U-Net: it runs offline, and the browser shows its saved
  output. (If C3 ships, say which one is on screen.)
- Write for a biomedical engineer. Explain σ, CLAHE and Dice once each, in plain
  words.

---

## 8. Constraints

- **Code:** vanilla HTML, CSS and JS (ES2020+; ES modules are allowed). No
  bundler, and no `npm install` needed to run or deploy. Draw charts in inline
  SVG or canvas, without a chart library.
- **Third-party code:** only where it saves real work, which here means TIFF
  decoding and, for C3, the ONNX runtime. Pin an exact version, load it from
  cdnjs or jsDelivr with Subresource Integrity, and load it only when needed.
- **Budgets:**
  - HTML, CSS, JS and fonts together under 300 KB uncompressed, not counting
    sample images;
  - the first sample visible within 1.5 s on a fast connection;
  - a median recompute under 30 ms for a 256 px image in Chrome on a laptop.
- **Browsers:** current and previous major versions of Chrome, Edge, Firefox and
  Safari. Handle touch and pen through Pointer Events.
- **Python:** do not change the default behaviour of `src/preprocessor.py` or
  the shipped model. You may add an opt-in parameter, for example
  `clahe_dtype="uint8"`, so that the scripts can produce 8-bit reference numbers.
- **Out of scope:** `app.py` (Gradio), training, and the portfolio repository.

---

## 9. Phases and checks

**Phase 0: baseline.**

1. Run `python scripts/verify_web_pipeline.py` and
   `python tests/test_preprocessor.py`, and record the results.
2. Take screenshots of the current page at 375×812, 1024×700, 1280×720 and
   1920×1080, in both themes. Use Playwright if it is installed; otherwise
   describe each view. Save them in a scratch folder, not the repo.
3. Write a ten-line audit that confirms or corrects section 3.

**Phase 1: correct data.**

Build M21–M22 (held-out samples), M14 (the Dice port) and the `cv.js` side of
M10 (the precision option). Extend the manifest with:

- a held-out summary;
- per-sample Dice;
- reference values for both precisions;
- the U-Net residual images, if you are doing S3.

Checks for this phase:

- The verify script covers both precisions and Dice: 40 dB or better for the
  pipeline, and Dice matching Python to three decimal places.
- A new script, `scripts/check_controls.js` (Node, no dependencies), runs every
  parameter across its range on two samples:
  - In 8-bit, it asserts that each parameter changes the output.
  - In 16-bit, at the default tiles on a 256 px sample, it asserts that clip has
    no effect. If someone fixes the bug, this test fails and reminds them to
    update the UI note. (The case study's own lesson was "Test that every
    control moves something.")
- Running `make_demo_assets.py` twice produces identical output.

**Phase 2: workbench and viewer.** M1–M12 and M25.

**Phase 3: measurement, overlays, input and output.** M13–M20, M23–M24 and
M26–M28.

**Phase 4: Shoulds, then Coulds.** Do them in this order of value: S2, S1, S3,
S4, S6, S5, S7, C1, C2. Do C3 only with the owner's approval.

**After every phase:**

- Run the scripted checks.
- Test by hand in a real browser at all four sizes and in both themes.
- Do a keyboard-only pass. Every control must be reachable and usable, focus
  must stay visible, and nothing may trap focus.
- Spot-check with a screen reader (NVDA or Narrator on Windows).
- Check reduced motion and high contrast.
- Serve from the repo root (`python -m http.server 8000`, then open `/docs/`) to
  catch any absolute URLs.

---

## 10. Ask the owner before you

- add a framework, bundler or build step;
- change default behaviour anywhere in `src/`, or retrain;
- commit any file over 5 MB;
- remove an existing feature, or a claim the page makes;
- touch anything outside `docs/`, `scripts/`, the split refactor in
  `src/dataset.py`, the opt-in preprocessor parameter, and `README.md`.

Do not commit or push. Leave all changes in the working tree.

---

## 11. Definition of done

- [ ] Every Must is done, or listed with the reason it is not.
- [ ] `verify_web_pipeline.py`, `check_controls.js` and `test_preprocessor.py`
      pass, with their output pasted into your final message.
- [ ] At 1280×720 the first screen shows the title, the one sentence, the viewer
      on a held-out frame, the headline line and the first controls.
- [ ] On a phone, dragging any control updates an image that is on screen.
- [ ] Every number on the page can be traced to a live calculation or to the
      manifest.
- [ ] The clip control's state is truthful in both precisions and at any image
      size.
- [ ] The README's "Browser demo" section is updated: what is new, how samples
      are chosen, and the precision control.
- [ ] Your final message lists:
  - the changes, by requirement ID;
  - what was skipped, and why;
  - the screenshot paths;
  - measured timings and page weight;
  - follow-ups for the portfolio repo. These are the demo video
    `public/assets/demos/phase-contrast-denoising.mp4` and its poster, the
    `public/assets/projects/phase-demo-metrics.webp` screenshot, and the video's
    alt text in `js/articles/phase-contrast-denoising.js`. All of them describe
    the old layout.
