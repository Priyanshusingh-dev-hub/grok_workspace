# ROADMAP: project ko kis taraf le jaana hai

Goal: ek **single command-line tool** (`textile`) jo user ke 4 kaam (CLAUDE.md me A–D) reliable tarike se kare, aur har baar same mill rules follow kare. `reference_code/` me prototypes hain jo kaam karte hain par alag-alag scripts hain, hardcoded paths aur thresholds ke saath.

## Target structure
```
textile/
  cli.py              # textile fill | repeat | tile | export ...
  io_utils.py         # load, DPI save, TIFF LZW, naming (no '#'), zip
  palette.py          # flat palette, k-means, stray color merge, nearest map
  fill_method1.py     # aligned majority-vote fill
  fill_method2.py     # structure/depth fill (misaligned refs)
  repeat_analyze.py   # repeat type detection
  tile.py             # deshear + crop search + seam quilting + upscale
  export.py           # final png/tif, colored channels, B/W seps, preview, report
  verify.py           # color count, channel overlap, DPI, seamless check
tests/
  samples/            # user ke test designs + expected stats
  test_*.py
```

## Phase 1: Common foundation (sabse pehle)
- [x] `export.py` + `verify.py`: abhi har script me ye logic copy-paste hai. Ek jagah lao.
- [x] `palette.py`: stray color merge function (threshold: < 0.05% pixels → nearest big color).
- [x] `cli.py` with `--size` (default 3535), `--dpi` (default 300), `--out`, `--name`.
- [x] Har command ke end me short **Hinglish report**: size, inch, DPI, colors + coverage %, verify pass/fail.

## Phase 2: Color fill (`textile fill`)
- [x] Method 1 ko module banao (`method1_colorfill.py` se).
- [x] Method 2 ko generalize karo:
  - [x] hardcoded paths aur thresholds hatao, args banao
  - [x] reference se palette auto (ground = sabse common color, motif = baaki)
  - [x] 2 se zyada colors, thodi misalignment ke saath: **Method 3** (`fill_method3.py`) bana. Reference ko
    tile-tile khiska kar line art par bithata hai (3 pass: 256/128/64 px tiles), phir Method 1 jaisa bharta hai.
    Floral ka reference 80 px tak moda: Method 1 87.4% sahi, Method 3 99.6%; seedhe reference par dono same.
    Depth wala tareeka (depth + local majority) bhi aazmaya: tree par Method 2 se sirf ~70% mila, chhod diya.
    **Abhi nahi hua:** 3+ rang wala *alag drawing* (motifs alag jagah). Iske liye asli sample chahiye.
- [x] `--method auto`: Method 1 → fail (alignment < 0.55 ya rangon ka farak > 6 points) → Method 3 →
  wo bhi fail → Method 2 (2 rang, report me saaf likha). Reference me 2 hi rang hon to seedha Method 2.
  Farak = har rang ka hissa reference vs output, kul farak ka aadha: floral 1.7, star 3.2, tree (Method 1) 13.7+.
- [x] Method 4 (line art ke gap band, Method 1 jaisa vote): AI line art ke liye. LoomLab app ka auto ab har tareeka
  aazmakar reference se score karta hai (`backend/app/core/filltrial.py`), trials ki table aur log ke saath.
- [x] Debug images: doubtful regions red me (Method 1 aur 3), side-by-side (reference | output | jo method nahi chuna).

## Phase 3: Repeat (`textile repeat` + `textile tile`)
- [x] `repeat_analyze.py` ko module banao, JSON output (type, W, H, drop, shear, mirror, avg_match).
  `textile repeat design.png [--out]`. AI floral par script jaise hi vectors: half-drop 502×316, drop 158, 5.1°.
- [x] `tile.py`: deshear → crop search (W, H ±8, position) → seam quilting → 3×3 preview → upscale (wrap pad) → export.
  Prototype ki settings (`--shear-ratio 0.0918 --w-range 496:514 --h-range 308:326`) par prototype jaisa hi tile
  (504×320, pixel-for-pixel). `--colors N` = flat rang + channels + verify; `--clean-ground 30` = ground pakka flat.
- [x] Rule: `tile` command andar se pehle `repeat` chalaye. Bina analysis tile kabhi na bane.
- [x] Panel print detection: agar strong vertical repeat hai par horizontal nahi, to "panel print" bolo aur tile mat kaato.
  Tree panel → "PANEL PRINT", tile STOP.

## Phase 4: Original designs (`textile make`)
- [x] Motif library: phool (layered petals), buti, patti, sprig, bel/vine (sine stem + spirals), dots, borders, haathi. PIL drawing, no AA.
- [x] Layout engine: half-drop grid, all-over scatter, vertical panel (center + side borders), wrap-safe drawing.
  Bikhre (scatter) motif bade motifs se door rehte hain; grid ko `shift` se aadha khiska kar doosra motif beech me.
- [x] Config file (JSON): palette, motif sizes, density, layout. User sirf config badle.
  `textile make examples/indigo_buti.json --out out/`. Examples: indigo buti (6 rang), floral all-over (7),
  haathi panel 18×9 inch (4). Har pixel ek rang, seamless, same config = same design (seed).

## Phase 5: Quality-of-life
- [x] Batch mode: `input/` folder ke saare pairs. `textile batch input --out output` (default `--method auto`),
  `NAME_lineart` + `NAME_ref`/`_colored`/`_reference`; ek kharab jodi baaki ko nahi rokti; `batch_summary.csv`.
  Windows: `run-textile-windows.bat` (textile_project\input → output).
- [x] Optional: vector trace per channel → re-raster at 300 DPI **bina AA**. Sirf jab user saaf edges maange. Kabhi Gaussian smoothing nahi.
  `textile vector design_final.png --out v/ [--eps 0.8] [--repeat]` + SVG. potrace offline nahi tha: OpenCV
  contours + approxPolyDP (kinara 0.8 px = 0.07 mm tak hilta hai), fillPoly LINE_8. Floral par 0.73% pixel badle;
  chhote dots (< 12 px) waise hi; `--repeat` wrap karke trace, jod saaf.
- [x] `textile edges` (user ne "kinare saaf karne wala tool" maanga): har rang ki outline ko apne saath-saath smooth
  karta hai (Gaussian image blur nahi), asli kone pakad ke rakhta hai (do seedhi baanhein = kona), patli line/chhote dot
  nahi hilte, bina anti-aliasing wapas draw, `--size`/`--dpi` se bareek grid par (1254 px -> 3535 px). Naya rang kabhi nahi.
  3 asli designs par match purane redraw jaisa ya thoda behtar (+0.1..+0.5), synthetic truth par ~8% kam galat pixel,
  par apne hi size (1x) par pixel grid hi limit hai: wahan fayda nahi. `--specks N` optional (default band: elephant par
  2 px ke tukde hatane se match 2.5 point gira, wo asli chhote motif the).
- [x] Optional 600 DPI export, par default kabhi nahi, aur warning ke saath (mill 300 DPI).
  `--dpi 600` = wahi inch, double pixel (7070), shuru aur ant me [CHETAVNI].

## Tests (har phase ke saath)
Samples me ye cases rakho aur expected stats check karo:
| Case | Expected |
|---|---|
| Floral peony/hibiscus (aligned) | Method 1, 9 colors, align ~0.96 |
| Floral, reference 80 px tak muda (test me banta hai) | auto → Method 3, align 0.53 → 0.95 |
| Tree panel cream/black (misaligned) | Method 2, 2 colors, cream ~31% |
| Star mandala (aligned, AI ref) | Method 1, 4 colors after stray merge |
| AI floral all-over | half-drop, ~502×316, shear ~5°, mirror nahi |

## Kya NAHI karna
- Boundary smoothing / Gaussian label smoothing.
- Colored image ka bilinear/bicubic resize.
- Mill ko 600 DPI bhejna.
- Repeat analysis ke bina tile kaatna.
- Brand/third-party designs ditto copy karna.
