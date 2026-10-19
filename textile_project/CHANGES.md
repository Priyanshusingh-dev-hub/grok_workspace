# Reconstruct

New command: `python -m textile reconstruct <img> --out <dir> --colors N [--scale 2] [--dpi 300] [--min-feature-mm 0.15] [--ai-upscale]`

## What it does

Two different jobs, on purpose.

* **Patch work** (flat ground, motifs, dots, stripes — anything with a hard boundary) is redrawn. A quality step runs first (Lanczos upscale, then a light bilateral denoise) so the tracer sees one edge instead of JPEG stairs and a brown fringe. Neighbours share that one edge. A round dot is redrawn as a circle. A thin stroke is kept, and thickened to `--min-feature-mm` if it would not print. Nothing measured on the source is deleted.
* **Spray / gradient** is not redrawn as flat patches. Ink amount comes from `gradients.solve_tones`, the same continuous overprint the existing `spray` command uses. A design with no real shading gets zero tone pixels.

`--ai-upscale` tries Real-ESRGAN and skips cleanly when it is not installed (no weights are bundled). Default is the classical boost, because an AI upscaler can invent petals that were not in the file.

`number`, `spray`, `auto`, and `--hybrid` are unchanged.

## Known limits

* Channel file names cannot contain `#` (mill rule). The PSD name looks like `INK-01-black 100C0E`. `report.json` lists `INK_01 black #100C0E`.
* Real-ESRGAN does nothing until `REALESRGAN_WEIGHTS` points at a model file.
* A photo is not a flat separation. On the floral and tree crops, colour error stays about 5–7 dE76 because eight inks cannot match JPEG texture. Flat plates themselves have std 0. The ≤3 target is for a real separation, not for a photograph.
* Patch labels are taken from the source, then redrawn at output size. The quality boost only slides the shared edge onto the cleaner boundary. It does not re-cluster the upscale, so a soft pixel cannot become a new island.
* This machine's test run cannot match the owner's 184 green tests: Python here is 3.10, so numpy 2.4 / the pinned opencv 5 and psd-tools 1.19 do not install. Failures in `phase1`, `paint` ("laal"), and PSD layer tests are from that older stack, not from reconstruct.
