# Textile Design Tool - Project Guide (Claude Code ke liye)

User se baat **Hinglish** (Roman Hindi + English) me karo, simple aur short. User textile mill ke liye kurta/suit fabric ke designs banata hai, zyada tar phone se kaam karta hai, aur AI tools se line art + colored reference generate karta hai.

Ye project ek **design pipeline** hai jisme 4 kaam hote hain:

| # | Kaam | Input | Output |
|---|---|---|---|
| A | **Color fill** | Line art + colored reference | Print-ready colored design + channels |
| B | **Repeat identify** | Koi all-over design | Repeat type (straight/half-drop/brick/mirror), size, jhukav |
| C | **Repeat unit nikaalna** | All-over design | Ek seamless repeat tile |
| D | **Original design banana** | Mood/colors/motif ka description | Naya seamless repeat ya panel design |

`ROADMAP.md` me likha hai ki kya kis order me banana hai. `reference_code/` me wo asli code hai jo user ke saath test aur approve ho chuka hai. **Naya tarika invent mat karo, pehle wahi logic use karo.**

---

## Mill ke pakke rules (sab kaamo par lagu)

1. **Default size 3535 × 3535 px @ 300 DPI = 11.78 inch.** Mill ka working format 300 DPI hai.
   - 600 DPI file mill ko mat bhejo. Unka software 7070 px ko 300 DPI maan le to design **double size (23.57 inch)** print ho jayega.
   - Formula: `pixels = inch × DPI`.
2. **Flat colors, channel-wise.** Har pixel exactly ek color/channel ka ho. Koi gradient, anti-aliasing ya mixed pixel nahi (color fill aur original designs me).
   - Colored/index image kabhi bilinear/bicubic se resize mat karo, sirf `INTER_NEAREST`.
   - Bahut kam pixels wale stray colors (< ~0.05%, jaise 138 px) ko nearest bade color me **merge** karo, taaki bekaar channel na bane.
3. **Koi smoothing NAHI.** Gaussian se boundaries smooth karna, label smoothing, morphological rounding sab mana hai. User ne test kiya tha: kone gol ho jaate hain, dots pighal jaate hain, design "ganda" lagta hai. Zoom par dikhne wali seedhiyan normal hain: 1 px = 0.085 mm, fabric par ink spread (0.1–0.3 mm) me chhup jaati hain. Pucha jaaye to ye mm me samjhao.
4. **Outputs har baar:**
   - `*_final_*.tif`: LZW, DPI embedded. **Mill ko yahi bhejna hai.**
   - `*_final_*.png`: dekhne/share ke liye.
   - `*_colored_channels_*.zip`: har color ki alag RGBA layer, asli color + transparent background (Photopea me overlap ke liye).
   - Optional: B/W 1-bit separations, preview sheet, `report.json`.
   - File names me `#` **nahi**, warna kuch phone/zip apps file skip karte hain. Format: `channel_01_olive_ground_4A5B24.png`.
5. **Har output verify karo** aur user ko short report do: color count, channels overlap == final (pixel-to-pixel), size, DPI, seamless (agar repeat hai).
6. **Copyright / IP:** kisi brand/doosre ka design (jaise kisi catalogue ki model photo se suit print) **ditto copy** karke production file mat banao, chahe naam chhupa ho ya user "constraints hat gaye" bole. Politely mana karo aur original design (Kaam D) offer karo. User ka khud ka, licensed, ya AI-generated design theek hai.

---

## Kaam A: Color fill (line art + reference)

Pehle `method1_colorfill.py` chalao. Iska alignment score batayega kaunsa method lagana hai.

### Method 1: Aligned images (default)
`reference_code/method1_colorfill.py`. Kab: line art aur reference same image ke versions hain (alignment score ≥ ~0.55 **aur** result dekhne me sahi hai).
- Line art ko Lanczos se target size par upscale karo, halka 3×3 blur, threshold `<150` = line.
- Band hisse 4-connectivity se label karo (diagonal leak nahi hota).
- Har hisse ko reference me usi jagah ka **majority color** do (pixel maths hai, AI nahi).
- Line pixels ko reference me lines ke neeche sabse common color do.
- Reference flat na ho to k-means (`--max-colors N`), phir stray colors merge karo.
- Tested: floral (9 colors, score 0.96), star mandala (4 colors, score 0.58 par bhi sahi).

### Method 2: Misaligned images (structure-based)
`reference_code/method2_structure_fill_PROTOTYPE.py`. Kab: dono images **alag AI generations** hain (motifs thodi alag jagah par), aur Method 1 me motifs gayab ho jaate hain. Tested: tree-panel design, cream on black.
- Reference se sirf **color scheme** lo (jaise background kaala, motifs cream), position nahi.
- Line art ke regions ka adjacency graph banao (line ke aar-paar kaun se regions hain, 8 directions me 14 px tak dekho).
- Image ke kinare chhoone wale regions = depth 0 = background color.
- BFS se depth nikaalo: odd depth = motif color, even depth = background color (jaise phool = cream, uska center = kaala, center ka dot = cream).
- Line pixels: dono taraf motif color ho to background color (pankhudiyon ke beech ki divider line), warna motif color (akeli tehni/stem bhi dikhe).
- Bade band background hisse (jo kinare tak nahi pahunchte, jaise border ki pattiyan aur triangles) reference se hint lekar background seed banao: area > 6000 aur reference-black > 0.8; side panels me area > 9000 aur > 0.55. **Chhote hisson par reference hint mat lagao**, warna pankhudiyan khokhli ho jaati hain.
- Abhi sirf **2 colors** (ground + motif) ke liye bana hai. 3+ colors ke liye depth/region-type ke hisaab se mapping chahiye, ROADMAP dekho.

### Method 3: Wahi drawing, reference thoda khiska (kitne bhi rang)
`textile/fill_method3.py`. Kab: reference wahi design hai par jagah-jagah thoda khiska/khincha hai (dobara
generate karne par aksar). Reference ko tile-tile khiska kar line art par bithata hai (sirf NEAREST, koi naya
rang nahi), phir Method 1 jaisa bharta hai. Alag drawing (motifs alag jagah, jaise tree) ko ye theek **nahi**
karta: wahan Method 2.

### Method 4: Method 1 + line art ke gap band (AI line art ke liye)
`textile/fill_method4.py`. Kab: line art aur reference same drawing hain (alignment achha) par AI line art ki
lines kahin-kahin tooti hain, aur ground ka rang gap se petal/patte me beh jaata hai (floral: poore patte
beige ho gaye). Lines ka mask band kiya jaata hai (dilate phir erode, ~1.5 source px), tab hisse ginte hain;
asli lines ko haath nahi lagta, band kiya hua hissa paas ke hisse ka rang leta hai, aur jo chhota hissa band
karne me nigal liya gaya wo apna vote rakhta hai. `--seal 0` = Method 1. Method 1 khud abhi bhi
reference_code jaisa byte-for-byte hai.

### Kaunsa method kab
1. `textile fill --method auto` chalao: Method 1 → jaanch (alignment < 0.55 ya rangon ka farak > 6 points =
   fail) → Method 4 → Method 3 → wo bhi fail → Method 2. Kyun badla, report me likha aata hai.
   (LoomLab app ka auto alag hai: wo har tareeka aazmakar reference se number deta hai, aur sirf-reference
   Reduce bhi ek umeedwar hai. Dekho root CLAUDE.md.)
2. Result ki preview (`*_compare.png`, jo method nahi chuna wo bhi saath me) **khud dekho**.
3. Kisi me galti ho to user ko batao kahan, aur debug image do.

---

## Kaam B: Repeat identify

`reference_code/repeat_analyze.py design.png`
- Patches ko template-match karke displacement vectors nikaalta hai aur DBSCAN se cluster karta hai.
- Batata hai: repeat type, full straight block size, half-drop unit + drop, mirror ya nahi, shear (jhukav).
- Tested: AI floral (black ground) → **half-drop**, block ~502×316 px, drop 158, ~5° jhukav, match ~0.70 (AI copies exact nahi hoti).
- **Panel/placement prints** (jaise kurta ka beech wala panel + side bel) all-over nahi hote. Unhe "panel print, straight vertical repeat" bolo, tile nikaalne ki koshish mat karo.
- User ko terms simple me samjhao: repeat / repeat unit / tile, straight, half-drop, brick, mirror, all-over print.

Command: `python -m textile repeat design.png` (JSON `--out` me). Panel print ko khud "PANEL PRINT" bolta hai.

**Galti jo ho chuki hai:** pehli baar repeat analysis kiye bina seedha tile kaat diya (620 px height, straight maan liya) aur result galat aaya. **Hamesha pehle Kaam B, phir Kaam C.**

## Kaam C: Repeat unit nikaalna

`reference_code/repeat_deshear_PROTOTYPE.py`, `repeat_tile_extract_PROTOTYPE.py`, `seam_quilt_lib.py`
1. Kaam B se W, H, shear lo.
2. Shear ho to affine se **deshear** karo (`x' = x - (shear/H)·y`).
3. Size W×H (±8 px search) aur crop position search karo jahan opposite kinare sabse zyada milte hain (MSE).
4. **Seam-cut (min-cost path)** se left-right aur top-bottom jodo. Seedha crop karne se jod dikhta hai.
5. Check: 3×3 tile karke dekho aur seams par zoom karo.
6. Upscale: wrap-padding ke saath Lanczos, halka unsharp. Background noise ho to pakke flat color se saaf karo.
7. Mill ke liye bolo: "is block ko **straight repeat** me lagao", kyunki half-drop block ke andar hi bana hai.
8. Source chhota ho (1024 px) to user ko batao ki upscale soft hoga, aur badi source image maango.

Command: `python -m textile tile design.png --out out/ [--colors 6] [--clean-ground 30]`: andar se pehle
repeat analysis, phir steps 2-7; `--size` = tile ki chaudai (default 3535). Panel print par ruk jaata hai.

## Kaam D: Original design banana

- PIL se procedural drawing (polygons, bezier, ellipses) **bina anti-aliasing**, flat palette (5–9 colors).
- Seamless: har element ko ±W, ±H offsets par 9 baar draw karo (wrap). Vertical-only panels ke liye sirf ±H.
- Tested examples: indigo buti half-drop (6 colors), elephant panel 18×9 inch (5 colors, haathi + bel side panels).
- Pehle preview khud dekho: density, motif size, ajeeb shapes. Phir iterate karo, user ko kam dikkat ho.
- "Inspired by X brand design" ho to colors, motifs aur layout sab badlo, look-alike mat banao.

Command: `python -m textile make config.json --out out/` (`textile/make.py`, `examples/*.json` se shuru karo).
Config: `size_px` (ya `size_inch`), `dpi`, `seed`, `palette` (naam: hex), `ground`, `repeat` (all-over/panel),
`layers` (layout + motif + size mm + colors). Sabse patli line 0.6 mm (`line_mm`), taaki chhape me na tute.

---

## Baaki commands (Phase 5)
- `python -m textile batch input --out output`: folder ke saare `NAME_lineart` + `NAME_ref` jode, auto method.
  Windows: repo me `run-textile-windows.bat` double-click (textile_project\input me files daalo).
- `python -m textile vector design_final.png --out v/`: kinaron ki seedhiyan seedhi + SVG. Sirf jab user maange.
- `python -m textile edges design_final.png --out e [--strength 1|2|3] [--specks N] [--repeat x|y|both]`: kinare saaf
  (user ke kehne par). Rule 3 (koi Gaussian smoothing nahi) yahan bhi hai: image blur nahi hota, outline apne saath-saath
  smooth hoti hai, kone (do seedhi baanhein) aur patli line/dot nahi hilte, bina AA wapas draw. Default `--size 3535`
  (1254 px ka design 3535 par bareek grid se banta hai). Apne hi size par fayda nahi: pixel grid limit hai. `--specks`
  default 0 (chhote tukde asli motif ho sakte hain). Chhote dots (< 0.2 mm) ka kaam LoomLab ka tiny-dot check hai.
- `python -m textile paint sketch.png --out o/ [--colors "1=cream, 4 7=laal" | file.csv] [--ref rangeen.png] [--seal N]`:
  sketch ke har band hisse ko number (`NAME_numbers.png`, chhote hisse neela dot + line), user ke bataye rang,
  har rang ek channel + poora package. Jis hisse ka rang na bataya ho: sabse paas wale bataye hisse ka rang, poora
  hissa ek rang (kabhi safed/ground apne aap nahi). `NAME_check.png` = bana design + numbers; sabse bade 12 hisse
  aur unka rang report me (rang "failta" dikhe to wahi CSV galti hoti hai: ground ke tukde ko motif rang).
  `--colors` kai baar: baad wala jeetta. `--ref` = usi design ka rangeen version, har hisse ka rang usse.
  Windows: `run-paint-windows.bat` (sketch + CSV / NAME_ref ek saath drag; `NAME_seal10.png` = seal 10).
- `python -m textile number rangeen.png --out o/`: rangeen design ke har ek-rang wale hisse ko number (sketch
  banwane ke liye map) + `NAME_colors.csv` + `NAME_flat.png`. Grain naapta hai: buna kapda / photo (grain >= 4)
  = daane saaf (median, patli/ragged chhitein ghul jaati, gol daane bachte); saaf digital design = koi median nahi,
  sirf kinare ka blend rang hatta (patla + do rangon ke beech) aur tooti outline ke tukde; bareek daane aur patli
  outline bachte. Saath me sketch: rang hata kar sirf lines (`NAME_sketch_seal0.png`, patli outline wale hisse
  seedhi kaali line, baaki hisson ke beech 2 px line) + `NAME_sketch_numbers.png`. Numbers wahi hain jo `textile paint`
  us sketch me khud ginta hai (seal 0, naam se), isliye `paint NAME_sketch_seal0.png --colors NAME_colors.csv` design
  wapas bana deta hai. Sudhaar (sab naape hue, paisley / buna kapda): (1) saaf design ke hisse uske apne size par,
  phir `edges` se smooth karke 3535 par (seedhiyan gayi, 2 min -> 45 s); (2) `--detail kam|normal|zyada` = 3 / 0.4 /
  0.1 sq mm se chhote hisse paas me (photo par 4x), lines se kate tukde bhi grey line me; (3) rang-rang ki seema grey
  (60) line, CSV `separators=fill` = print me paas ka rang, kaali outline kaali hi (95.5 -> 98.1% wapas match);
  (4) har run khud sketch + CSV se design wapas banakar % batata hai; (5) ek jaise hisse ek letter (`NAME_map.png`,
  CSV ka Group column); (6) `--line-mm` (default 0.17 = 2 px; grey line 2 px se patli nahi, tirchhi jagah tootti).
  Buna kapda (photo) wahi purana 3535 wala raasta + `edges` 1x smooth: 1066 -> 533 hisse, 97% match.
  `NAME_sketch_black.png` = sab kaala (dikhane ke liye); paint me `NAME_sketch_seal0.png` hi do. Windows: `run-number-windows.bat`.
  Separator line boundary ke beech (dono taraf 1 px): truth bench (3 `make` designs, 1000 px + JPEG 85 karke wapas)
  98.58 -> 99.05% asli se mel; baaki galti kinaron ke 2 px me (1000 px source ki seema). Numbers sheets poore size par
  (zoom par saaf, font sheet ke saath bada). `--hd` (ya naam me `_hd`) = 7070 px @ 300 DPI (23.6 inch), ~3 min.
  Line `--line-mm auto` (default): 0.17 / 0.35 / 0.5 mm teeno se sketch, har ek ko apni CSV se paint karke flat
  design se milata hai, sabse zyada match wala rakhta hai (0.05 tak barabar = moti); tulna report me, chuna hua
  `NAME_rangeen.png` (jaisa paint banayega). User ka paisley: 99.98 / 99.26 / 98.41 -> 0.17. Number diya to sirf wahi.
  Saaf design ke rang sirf 'solid' pixels se (3x3 me Lab range < 30): bhare design (user ka mor jaal, motif 5-15 px,
  kaali outline) par sab pixels se k-means ne mitti-grey diye, hara/gulabi/sunehra gaye. Blend ink sirf < 3% hissa
  (busy design ke asli rang bhi patle hote hain). 0.3% se kam ink paas wale me. WOVEN_GRAIN 4 -> 6 (busy saaf design
  4.4, buna kapda 8.2). Bench 98.93 -> 99.19. Mockup photo (design ke chaaron taraf kapda) khud crop karo.
  Line dono taraf barabar; jis hisse ka core (1.5 px) ya ek-tukda-pan line se toot-ta, us
  taraf sirf 1+1 px line (`_wide_line`): bench par koi hissa nahi gaya, mel 99.05 -> 98.93 (0.5 mm: 98.88).
  `shade_rims`: patla (<= 3 src px) lagbhag-kaala (L < 25) tukda jo apne se halke, dE < 20 wale hisse ki 1/4+ seema
  chhoota hai = uski shading, us hisse ka rang (user ke paisley ke navy patton ke kaale dash; bench par 0 nuksaan:
  dE akela nahi chalta, degraded floral ka hara/olive 18.4 vs kaala/navy 18.7). Sirf sabse gehre ink ke patle tukde
  'line' bante hain (cream nas ek hissa rehti hai). Full-size numbers: jagah na mile to dot ke paas safed halo par likh do.
  `NAME_sketch_bold.png` + `.svg` (`textile/curves.py`, `--bold-mm` default 0.5): wahi hisson ki outline, har ek ko
  `edges` ke `find_corners` + `smooth_outline` se smooth curve (gol gol rehta hai, do seedhi baanhon wala kona tez),
  gehri moti anti-aliased line, SVG kitna bhi zoom saaf. Patle hisse (< 1.2 line chaudai) ka safed core bacha rehta
  hai, bold line chhote petal/dot ko bhar nahi deti. Sirf dekhne / share / upar draw karne ke liye; `paint` ko
  `NAME_sketch_seal0.png` hi do (uske areas hi numbers hain). Curve: corners se tukde; jo tukda chord se 0.9 px se
  zyada nahi hilta wo seedhi line, baaki smoothing cubic spline (sine jaisi lehar, arc, spiral ek behti curve; ends
  corner par tike). `NAME_sketch_bold_numbers.png` = bold sketch par wahi numbers. `--bold-scale` (default 1 = 3535 px @ 300 DPI, mill jaisa,
  pixel-pixel wahi jo user ko pasand aaya tha; 2 = 7070 px @ 600 DPI wahi 11.78 inch, sirf dekhne ke liye): bold
  sketch + numbers khali canvas par curves se seedhe draw (image bada nahi), 4 = 14140 px; SVG design ki apni units me. `NAME_sketch_bold_rangeen.png` = bold sketch me design ke rang bhare (`curves.colour_fill`: har rang ka mask halka blur,
  bada karke sabse zyada wala jeetta = kinare curve jaise smooth, ek pixel ek rang; upar bold line multiply, safed halo nahi). Sirf dekhne ke liye, mill ko nahi. **Curves ek lagatar slope me** (`--smoothing` 0-100, default 70 = Photoshop brush ke "Smoothing" jaisa slider; andar `fair` sigma = smoothing/20 source px,
  ladder dekhi: 0% jhatke, 50% theek, 80% bahut smooth par shape wahi; `--fair` chhupa hua expert alias; `curves.fair`, `_prune_corners`, `CORNER_DEG`): (1) kona tabhi jab mod > 60 degree (pehle 45: halke mod bhi
  kone ban jaate the = jhatke); (2) do konon ke beech 4 source px se chhota tukda (notch / kaan) = shor, kona hata do (kamzor wala); (3) har kone-rahit tukda
  spline ke baad Gaussian se halka aasan (arc-length ke saath, sigma `--fair` source px; ant ke point + slope pakke: odd reflection): slope dheere badalti hai, chhote
  jog halke S ban jaate hain, radius tens px hai isliye gol shape sikudti nahi; (4) `dedup`: do parts ki saajhi seema ek hi baar kheenchi jaati hai (bade part se)
  — par sirf wahan chhodi jaati hai jahan 0.5 stroke ke andar doosri line chal rahi ho, warna nahi (patle part ke paar chhalang lagane wale lookup se gap na bane).
  Pehle har part ki apni smoothed copy = double line + crescent sliver. Hamare sikhe hue: tolerance-wale minimal-knot spline (Schneider jaisa) se curvature wobble
  kam nahi hua (metric wahi), tolerance badhane par shape se door gayi; Taubin bhi is wavelength par kamzor; Gaussian fairing + kone-pruning + ek-baar-stroke ne asli farak
  dikhaya. Bacha hua: source AI image ki apni ragged kinari / sliver-parts (patli 1-2 px bhaag) se bani asli ugliness ko ye hataa nahi sakta.
  `--circle` (default 0.9, 0 = band): jo outline apne best-fit circle se 90%+ milti hai (IoU) aur kisi point par radius se 12% (+1.5 px) se
  zyada nahi hilti wo pakka gol circle (`curves.circle_of`); daante wali / scalloped ring, square, ellipse, sheet ke kinare se kati outline
  waisi hi rehti hai (pehle sirf 90% IoU par daante mit rahe the). Sirf bold sketch; `NAME_sketch_seal0.png` (mill / paint) pixel wala hi rehta hai.
  User ka paisley: 566 outlines me 25 circle.
  `--polygons` (default 0.9, 0 = band): wahi soch triangle / rectangle / diamond / pentagon (3-5 seedhi bhujaen) ke liye (`curves.polygon_of`):
  approxPolyDP se kone, har bhuja apne outline points par least-squares line, padosi lines ka milan = naya kona; shape ko
  tabhi badalta hai jab polygon se IoU >= 0.9 AND koi outline point bhujaon se 3% sqrt(area) (+1.5 px) se door nahi AND har kona
  <= 150 degree khula (warna curve tukdon me kati hai). Isliye patta/petal/D-shape/ek bhuja phooli hui shape kabhi seedhi nahi
  hoti: design dekh kar hi faisla (shape khud polygon ho tabhi). Circle pehle try hota hai. Run ki report me gino: kitne circle /
  polygon / smooth rahe. Paisley: 25 circle, 15 polygon, 526 smooth.
  `--motifs` (default 0.9, 0 = band): oval (`curves.oval_of`, cv2.fitEllipse) aur patti / petal (`curves.leaf_of`: `find_corners` se
  theek 2 nok, har bhuja ek quadratic arc (control point least squares, nok tak arc-length), dono bhujaen 25% ke andar barabar jhuki
  ho to ek jaisi = symmetric petal). Wahi do jaanch (IoU >= 0.9 aur 3% sqrt(area) +1.5 px se door koi point nahi); ek nok wali boondh,
  dhaar wali shape, kati shape smooth curve hi rehti hai. Order: circle, polygon, oval, patti, baaki smooth. Paisley: 25 circle, 15
  polygon, 2 patti, 15 oval, 509 smooth (zyadatar patte bade-ragged hain, 90% clean nahi: wahi rehte hain). Chhote band blob par extra smoothing (`BLOB_SMOOTH`, 2.5) band hai (1.0): asli chhote dots chapte ho rahe the.
- `number` ab **rang plates bhi banata hai** (default): sketch (`seal0`) + CSV se wahi `paint` package `OUT/package/` me: har rang ki alag
  channel (`NAME_colored_channels_*.zip`, RGBA), B/W separations, mill ka `NAME_final_*.tif` (300 DPI), verify. `--no-package` = band.
  Saath me alag-alag images (`textile/stack.py`): `package/channels/` = har rang ki apni plate (RGBA, 3535 px @ 300 DPI), `package/stacked/` =
  `NAME_stack_01_of_07.png`... plate 1, 1+2, ... sab (safed par), har ek poora 3535 px @ 300 DPI, thumbnail sheet nahi; aakhri = final design.
  Photoshop file (`textile/psdout.py`, `package/NAME_layers_3535px_300dpi.psd`, `--no-psd` = band): har rang ki plate ek layer (naam `03 rust motif A83728`,
  neeche se upar press order), peeche transparent, upar chhupi hui GUIDE layer (bold sketch + numbers, Multiply). Canvas 3535 px, RGB 8 bit, 300 DPI
  file me likha. Layers on (guide band) = final design pixel-pixel (test); merged preview bhi RLE me (raw 37 MB tha, ab 11 MB file). psd-tools se likhta hai
  (backend ki hi dependency).
  `--merge-similar` (default band): chhota rang (< 1.5% hissa) jo bade rang ke dE < 12 ke andar ho = ek hi rang, uske pixel bade me (sketch, CSV,
  channels sab ek saath, kyunki parts banne se pehle). Wahi niyam jo `learn` ka `similar_inks` salah naapta hai. User ke paisley: navy 0.31% / 0.08% ->
  asli navy me, 6 -> 5 channels, 0.3% / 0.08% pixels ka rang badla. Default band: kisi design me asli chhota accent bhi paas ke rang jaisa ho sakta hai.
  Pehle sirf numbering + CSV banti thi aur user ko plates alag se chalani padti thi (bhool hui); ab ek command.
- **number ki safai v2 + speed** (naapa hua; rang ka kaam ab `number.flat_index` me alag, taaki akela naapa ja sake):
  (1) **patli tirchhi line nahi tootti** (`_rescued_lines`, `LINE_RESCUE` 10): 1 px tirchhi line 4-connected me tukde-tukde
  (har tukda 'speck') thi aur pighal jaati thi (user ka rust/purple design: cream patton ki gehri jhiriyan toot kar dash).
  Ab 8-connected 10+ px ki line, jiske chaaron taraf zyadatar (75%) EK hi rang ho (cream ke andar jhiri) = line, rehti
  hai; do alag rangon ke beech chalti patli patti (kinare ka AA fringe) pehle jaisa pighalti hai. Sirf saaf image (grain
  < 2, `LINE_RESCUE_GRAIN`): kapde ki photo (peacock mockup, grain 3.6) me dhaage bhi aisi line hain, wahan grey chhitein
  bachti thi. (2) **pighla pixel rang dekh kar jaata hai** (`_to_nearest_colour`): chhue hue rangon me jo uske asli rang
  ke sabse paas (pehle: jo bhi paas wala hissa). (3) **blend ink 3% se upar bhi** (`BLEND_RIM_ONLY` 0.8 / `BLEND_CORE`
  0.03 / `BLEND_EXACT_DE` 2.5, < 10%): jo ink 80%+ kinare par, 3% se kam pixel 3 px andar, aur do rangon ka ekdum mix
  (dE <= 2.5) = blend, screen nahi. AI ke naram kinare chaude blend banate: rust/purple ka maroon 3.6%, ginkgo ka silver
  5.8%, kaale sprigs ka grey 3.3% alag screen ban rahe the. Peacock jaal ke asli patle rang (teal/hara/sunehra) mix nahi
  (dE 4.6/10.6/13), uska cream-grey (dE 3.1, 78% kinara) sabse paas: isliye 2.5 aur 0.8. Blend ink palette se hat-ta
  bhi hai (pehle 0% wala khaali ink rehta tha). Naap: truth bench (3 `make` design + jhiri wala, AI jaisa kharab karke)
  98.87 -> 99.02% (harsh: 98.12 -> 98.33), haathi/jhiri design ki ink ginti ab sahi (5/4, 7/4 -> 4/4); user ke 25 asli
  design: asli image se mel 84.12 -> 86.55 (har ek behtar, koi kharab nahi), ink 4.96 -> 4.64 (03: 4->3, 06: 6->5, 10:
  3->2, paisley: 6->4, jaal: 3->2). Aazma kar chhoda: 'patli patti tabhi pighlao jab uska rang do padosiyon ke beech ho'
  (JPEG ke rang-shor se kinaron par gulabi halo bacha, bench -0.3/-0.7); sirf 8-connected area (kuch nahi badla).
  (4) **asli mel** (`design_match`, CIEDE2000 `names.delta_e2000` = backend wala, LoomLab Reduce ka paimana): flat design
  vs aapki image, report + log me. Sketch ka 99.9% sirf 'sketch + CSV = flat' batata hai; AI image par 85-93 achha,
  95+ mumkin nahi (shading/grain flat ink nahi). (5) **Speed** (ek 3535 px design ~4.4 min -> ~1.3 min; har badlaav ka
  output purane se milaya): `curves._strays` (oval/polygon/patti ki 'koi point door?' jaanch: kone ke paas wala point
  pakka paas, baaki sirf paas wali bhujaon se; 88 s -> 2 s, bold sketch byte-same, sirf 1e-15 ke dhaar par alag ho sakta),
  `_label_max` (ndimage.maximum ki jagah, 50x), `_wide_line` ek hi distance map teeno line motai ke liye + 1D unique
  (9 s -> 0.3 s har baar, same), `io_utils.unique_rgb` (np.unique axis=0 ki jagah packed int: colour_fill 15 -> 2.6 s,
  crisp/final/edges/vector me bhi, same), PSD ka preview khud (plates safed par; psd-tools ka composite 18 s -> 0,
  file byte-same), `paint._label_points` ek map par yaad (9 baar -> 2) + barabar gehre points me beech wala (number patti
  ke beech, scipy koi bhi chunta tha), `paint._shape` ka minAreaRect sirf outline par (9 s -> 0.3 s; box 1e-6 tak same,
  ek baar barabar-area wala doosra box: sirf group letter par asar).
- **number v3: patli lines ka rang, chhote asli rang, jud-waan ink** (truth bench v2 = purane 4 + gehre ground par cream
  bal-lines, cream par kaali bal-lines, chhote sunehre daane; normal aur bahut dhundhla (1000 px, blur 1.0); naya naap
  'thin' = patle hisson ke pixel sahi): (1) **`line_inks`**: AI image me 1-2 px line ground me ghul jaati hai, pixel line+ground
  ka MIX hota hai (RGB me, wahi jahan image bani) aur nearest ink ground ya beech wala ink (10 ke safed reshe gayab, cream par
  kaali bal-line sage ban kar pighal gayi). Pixel ke kisi disha (4, 1 ya 2 px door) ke dono taraf ek jaise ho aur pixel unse
  alag (`RIDGE_MIN` 20 RGB) = line; uska ink = dono taraf se pixel ki taraf ki kiran par pehla ink (`RIDGE_ANGLE` 0.15,
  pixel kam se kam 20% raste par: `RIDGE_ALPHA`). Kinara line nahi (dono taraf alag). Pixel ka apna ink pehle se kiran par ho
  to nahi chhoota (06 ke teen maroon me pehla maroon chura leta tha, lines cream ho gayi). Lab me kiya to peacock ki sunehri
  lines cream ban rahi thi (Lab me mix mud jaata hai): RGB. Blend ink hatne ke BAAD chalta hai (pehle: line ko blend ink mila,
  wo hata to line gayi). Aazma kar chhoda: 'line ke saath wali disha me dhalaan na ho' (jaal ki line kam moti, par 10 ke reshe
  aur peacock ki naseen bhi gayi). (2) **`SMALL_INK_CORE`/`SMALL_INK_DE`**: 0.3% se chhota ink rehta hai agar uske aadhe pixel 2+ px
  andar (pakke daane) aur har ink se dE2000 >= 20 (floral ke maroon centre 0.28% gayab hote the). Aisa chhota ink kisi blend ka
  'maa-baap' nahi banta. (3) dheela blend (dE 10-20, poora kinara) sirf jodi ke beech (0.2 < t < 0.8): sire ke paas = us sire ka
  shade (floral ki hari tehni, gulabi se olive t 0.90, blend maan kar hatai ja rahi thi). Aazma kar chhoda: 'blend dono
  maa-baap ko chhoota ho' (asli designs ke blend ek maa-baap + doosra blend chhoote hain: paisley 4 -> 5 ink). Blend hatana
  ab 3 baar tak (blend ki zanjeer). (4) **`_merge_twins`**: dE2000 < 3 (aankh ko ek) wale do ink ek (k-means complete-linkage
  ne do navy chhode: indigo 7 -> 6, hair_light 4 -> 3). Bench v2: normal mel 99.18 -> 99.36, sabse kamzor ink recall 60 -> 75,
  thin 23 -> 36%, galat ink-ginti 4 -> 1; dhundhla 98.41 -> 98.81, 44 -> 62, 16 -> 32%, 5 -> 3. User ke 30 asli design: ink
  ginti har ek ki wahi, pixel-mel -0.2 (line ko pura ink dena aadhe-mix pixel par kuch kaat-ta hai; isliye truth bench se
  jaancha). Nahi ho saka: bahut dhundhli image me 1 px ke sunehre daane (koi pixel sunehra bacha hi nahi) aur cream bal-line
  jinka rang kahin poora nahi dikhta (palette me cream hi nahi aata).
- **`--merge-shades` / `--no-merge-shades`** (`number` aur `final`; LoomLab aur CLI dono me default ON; band: `--no-merge-shades` ya app me tick hatao): AI tasveer ki
  shading (patti ki jad me gehra laal, sunehri patti ka narangi kinara) alag ink ban kar sakht kinare wale dhabbe banati thi.
  `merge_shade_inks`: chhota ink jo bade ink se dE2000 < 10 (`SHADE_DE`) ho AUR apne kinare ka 30%+ (`SHADE_TOUCH`) usi ko
  chhoota ho = uski shading, uske pixel bade ink me (bada apna rang rakhta hai; chhote se bada, zanjeer ant me main ink tak).
  Patli line wala ink (20% se kam pixel 0.35 mm andar: `SHADE_LINE_MM`/`SHADE_CORE`, jaise beige par cream jaali dE 8.8)
  kabhi nahi milta; jo motif chhoota hi nahi wo bhi nahi. 14 asli design: sirf 14 (laal-kaala floral: gehra laal -> laal,
  gehra narangi -> narangi, 7 -> 5 ink; mill size par charcoal -> kaala bhi, 8 -> 5) aur 02 (gulabi ka gehra dhabba, 5 -> 4)
  badle, baaki 12 waise ke waise; 06 ki maroon jodi (touch 28-29%) aur 05 (18%) nahi mili. Default ab ON (AI shading dhabbe rokne ke liye); asli do-rang
  motif ho to `--no-merge-shades` / app me tick hatao. Run ki report/`NAME_number.json` me `shades_merged`.
  `number` ab `NAME_number.json` bhi likhta hai (r ka saar: hisse, inks, mel, files ke naam) — LoomLab isi se padhta hai.
  `number` ka package wala hissa `cli.number_package` (paint + PSD + alag images) aur `final` ka `cli.fit_to` (crop) +
  `cli.final_package` (channels, TIF, SVG, verify, report) alag functions hain, output byte-same (snapshot se jaancha):
  LoomLab me operator rang badle (recolour/merge/apne inks) to app CSV ka HEX column badal kar yahi functions dobara
  chalata hai, taaki zip me bhi wahi rang aayein. Logic nahi badla, sirf code ek jagah.
- **Chhote objects ka alag raasta** (`number.draw_small`, `--small-px`, default 20, `0` = band; `number` aur `final`, LoomLab
  ka teesra tab bhi): user ne dekha ki bade objects wale design achhe bante hain par bhare design (chhote phool, daane, tehniyan)
  kharab. Wajah: ChatGPT ki ~1448 px tasveer 23.5 inch par sirf ~2.4 px/mm; 3 mm ka daana 7 px, uska aadha kinare ka mix.
  Purana raasta har tasveer-px ko ek ink deta (kinare ki jaankari phenk kar), phir `edges` se bada karke outline smooth karta:
  7 px ka star gol dhabba, patli tehni seedhiyon wale tukde. User ka idea: chhote ke liye alag, bade ke liye wahi. Ab: chhota
  object = ek-ink hissa (8-connected) `SMALL_PX` (20) tasveer-px se kam chauda (sqrt area) ya `SMALL_THIN` (2.5 px) tak patla,
  + 2 px ghera (`small_zone`); usme har ink ka har tasveer-px me hissa (`ink_coverage`: tasveer RGB me anti-aliased hai,
  kinare ka px = (1-t) a + t b, t = projection; chhote object ka px apne aas-paas ke ink (9x9) ke saamne) cubic se bade grid
  par, jis ink ka hissa sabse zyada wo jeeta. Bahar sab `edges` wala hi, pixel-pixel same (test). Zone me bhi: 2 tasveer-px
  se chhota tukda (`SMALL_SPECK`: AI texture, jaise 04 ki sage patti ki kaali dhaariyan) aur do SHADE rangon ki seema
  (`SMALL_SHADE_DE` 15: laal ↔ gehra laal) purane jaisa; ek rang ke shade wala chhota tukda zone me hi nahi. Rang wahi jo
  purana palette chunta hai, sirf shape badalti hai (rule 2 wahi: index image kabhi cubic se nahi, sirf har ink ka 'hissa'
  field; har px phir bhi ek hi ink). Naap (`small/truth.py` jaisa: 3 naqli bhare design, ~7600 objects har
  size ke, AI jaisa kharab, object IoU size ke hisaab se): 2.4x par 8-12 px 0.78 -> 0.88, 12-20 px 0.88 -> 0.94; 4.9x (mill
  sheet) 0.81 -> 0.91, 0.88 -> 0.96; 4-8 px 0.35 -> 0.41 (kitne mile 46 -> 47%); truth bench v2 teeno setting thoda behtar
  (99.36 -> 99.50, thin 36.0 -> 37.5); 14 asli design sab tasveer ke zyada paas (+0.1 se +1.5), koi ink nahi khoya; 14 mill
  size: patli tehniyan ek saaf line (pehle tukde), mel 90.6 -> 90.9. Samay: 3535 par +2-4 s, 7050 x 6210 par 34 -> 42 s.
  Aazma kar chhoda (naape hue): (1) px ko doosre ink ka mix maan kar dobara padhna (tan dhabba = sunehra daana): naqli
  design par 4-8 px daane 46 -> 67% mile, PAR asli 04 ki sage patti (navy-cream ke beech ka asli rang) phat gayi aur sirf
  patli sage dhaariyon wala ink poora gayab (purana 98% rakhta tha): rang se mix aur beech ka asli rang alag nahi hote;
  10 RGB ki margin se bhi sage phati. (2) 'mix-jaisa ink' (core < 10%, do inks ke beech) hata dena: wahi sage wala nuksaan.
  (3) har daane ka area uske kul rang se: 0-4 px me thoda fayda, 8-20 px objects bade bane. (4) shares par Gaussian 0.5:
  har naap me thoda nuksaan. Bacha hua: 2-3 px ke daane (tasveer me hain hi nahi), aur bhare design me kinare ke mix rang
  alag ink (bhure dhabbe) — ye rang ka sawaal hai, shape ka nahi; patli line ke asli rang ko bachate hue hi chhedna.
- **Export + verify tez** (har command inhi se file likhta/jaanchta hai): `export_package` channels ko `io_utils.ordered_map`
  se 4 thread par banata hai (PIL ka PNG/TIFF zlib/LZW aur numpy ka bada kaam GIL chhodta hai), final PNG + TIF saath me
  (har ek ki apni PIL image: ek image ko do thread save karein to `encoderinfo` mil jaata), thumbs ink-map ko ek baar NEAREST
  chhota karke (har channel ka poora safed sheet nahi), B/W seedha 1-bit (`Image.fromarray(~m)`, pehle int64 np.where).
  Zip me entries usi order me, har file ke bytes wahi (14 asli + 40-ink + 7050 px par milaya; zip ka apna time-stamp
  har run alag hota hi hai). `verify_package` har channel aur uska B/W ek saath, ek hi baar padhta hai (pehle channel do
  baar decode hota tha), rang `getcolors` se (ek ink = ek rang, warna poora array; nateeja wahi). 7050 x 6210 par:
  export 37 -> 7.6 s, verify 29 -> 8 s; `final` (14_red_black_floral, 23.5 x 20.7 in) 91 -> 34 s, PNG/TIF byte-same,
  RSS 1.8 -> 2.2 GB (4 channel ek saath memory me).
- **Tool ki yaaddasht** (`textile/learn.py`, `data/number-log.jsonl`, gitignored: user ki apni machine par badhti hai, `TEXTILE_LEARN_LOG`
  se jagah badal sakte ho): har `number` run ke baad (1) `Salah` = kya dikkat dikhi + kya karna hai (likhe hue niyam, har ek ka Hinglish
  upay: `similar_inks` (chhota ink bade ke dE < 12 aur < 1.5% hissa = shayad ek hi rang, CSV me ek karo), `tiny_parts`, `missed_numbers`,
  `match_low` (< 99%), `colours_at_limit` (--colors badhao), `busy` (> 1500 hisse), `woven`/`grainy`, `thick_line_won`), (2) `similar()` =
  pehle ke milte-julte design (grain, hisse/Mpx, ink ginti, bade ink ka hissa) aur unka verdict, (3) run log me. `python -m textile
  feedback NAME good|bad "note"` = user ki raay latest run par; `python -m textile learn` = runs, aam dikkatein, achhe/kharab runs ka saar.
  Ye neural network NAHI hai (kuch dozen designs par train karna sirf ratta lagwana hota: overfit): record + niyam + nearest-design.
  Jab ~50+ runs par raay (good/bad) jama ho jaye tab inhi features par chhota model (logistic regression / tree) fit ho sakta hai.
  Sirf salah aur note: koi run ka output isse nahi badalta; log fail ho (read-only disk) to run nahi rukta.
- **Parts se design** (`textile/stitch.py`): `python -m textile stitch PARTS --out o/ --name N [--grid 3x3] [--number --number-args ...]`;
  PARTS = folder, zip ya files. Jagah naam se (`r1c2`, `piece_1_2`, `x_1_2`; aakhri `_1536x1536` size tag chhod kar), warna naam-kram row-wise,
  grid ginti se (9 = 3x3). Har padosi jodi: agle part ki 6% patti pichhle me dhoondhna (template match >= 0.8), phir poore overlap par jaanch
  (grey farak < 12 aur seedhe kinare-se-kinare jod se behtar; repeat motif se jhootha milan na ho); na mile = kinare se kinare. Overlap me jod
  `tile` wale seam-cut se (sabse kam farak ki line), kabhi blend nahi (mixed rang nahi). Parts apne size par (overlap se kinare wale parts chhote
  hote hain); sirf 30%+ alag bade part scale. Report: har jod ka 'jump' (jod ke aar-paar rang farak) vs part ke andar ka; > max(18, 2.5x andar)
  = DIKHNE WALA JOD (alag-alag AI drawing). `textile split IMAGE --grid 3x3 --overlap 12` = ulta: overlap wale parts (har ek alag bada karne ke
  liye). User ke 9 x 1536 px parts (paisley_vine, kinare se kinare): 4608 px, koi jod nahi dikha. Sach: ye parts usi 980 px image ke bade kiye
  tukde hain, zoom par utne hi naram; asli fayda tabhi jab har part me sach me zyada detail ho (har hissa alag se bada AI se dobara bana).
- **Zoom par saaf** (`textile/crisp.py`, `python -m textile crisp design_final.png --out o --scale 2.8 --zoom 2`): pixel design zoom par dibbe ya (viewer
  smooth kare to) dhundhla dikhta hai. `crisp` har rang ki outline ko bold sketch wale hi curve-fit (`curves`: circle/oval/patti/polygon pakke, seedhi bhuja
  seedhi, kone tez, baaki ek behti curve) se dobara draw karta hai: SVG (kitna bhi zoom saaf) + `NAME_crisp_2x.png` (seedhe curve se draw, bina AA = har pixel
  ek flat rang, enlarged copy nahi). Sabse bada rang ground, baaki uske upar bade-se-chhote (rangon ke beech gap nahi). Report: pixel design se kitne % pixel
  alag (sirf kinare; d3: ~1.9%). Sirf dekhne / edit / share ke liye; mill ko pixel TIF hi. `textile vector` (purana) seedhe tukde banata hai (gol dot =
  aath-kon), `crisp` curve. Is me mile do sudhaar `curves` me bhi gaye (bold sketch ko bhi): (1) `fair` chhote band shape ko nigalta tha (sigma 8 px, dot radius 12:
  r ~8 reh jata) -> sigma <= 0.3 x radius (`FAIR_MAX_RADIUS`); (2) dot-size outline (perimeter < 50 src px) + <= 2 kone, koi 110 degree se tez nahi = ek gol curve
  (`SMALL_ROUND_PER`), warna gol dot teardrop ban jata; `leaf_of` ko tip-to-tip / chaudai >= 1.6 (`LEAF_MIN_ASPECT`).
- **Mill ke exact repeat size par** (`python -m textile final design.png --out o --inches 23.5x20.7 [--dpi 300] [--colors N] [--detail kam|normal|zyada] [--merge-similar] [--merge-shades|--no-merge-shades]`): pixels = inch x DPI
  (23.5 x 20.7 @ 300 = 7050 x 6210). Design ko repeat ke aspect par CROP karta hai (kheenchta kabhi nahi; seamless tha to jod toot jaata hai, bataata hai),
  phir `number` ka hi rang wala kadam (`flat_index`: ink chunna, blend/jud-waan ink hatana, patli lines, `edges.clean`) SEEDHA mill size par draw karta hai,
  phir `export_package` (channels zip, B/W, TIF 300 DPI), `vector.trace` se SVG, verify, aur report me `design_match` (asli picture se CIEDE2000 mel).
  Pehle `number` aadhe size par + `crisp` se 2x tha: 14_red_black_floral par 180 s / 2.2 GB, crisp ne petal ke dots aur cream ki patli daraarein kho di,
  aur uska scale zoom ko do baar ginta tha. Ab 91 s (tez export/verify ke baad 34 s, 2.2 GB RSS; flat_index khud 13 s), verify PASS,
  mel 90.6%, dots aur daraarein bachi.
  Ab `final` bhi layered PSD banata hai (`NAME_layers_7050x6210px_300dpi.psd`, har rang ek layer, `--no-psd` = band): export ke BAAD,
  taaki peak memory na badhe (7050 x 6210, 5 rang: +15 s, ~0.85 GB alag se; 12 rang: +35 s). Guide layer nahi: `number` ka bold sketch
  doosre size aur crop ka hai, kheench kar daalna galat jagah dikhata.
- **AI ke dobara banaye parts ki jaanch** (`python -m textile partscheck REFERENCE PARTS --plan NAME_parts_plan.json`): `split` ab overlap 15% (pehle 12), `--target 5000` batata
  hai AI se har part kitna maangna hai (5000 px = 1825 px part; 2048 / 1536 bhi chalta, `stitch` jodta, `final` exact size banata), aur `NAME_parts_plan.json` likhta hai (crop boxes).
  `partscheck` har AI part ko reference ke usi crop se milata hai (AI part reference ke scale par, INTER_AREA): (1) khisakna = phase correlation, > 1.2% chaudai;
  (2) naye rang = reference tile ke rang (coarse quantise, >= 0.4% hissa) se Lab 22 se door pixel > 6%; (3) detail = Canny kinaron ki lambai: < 0.80x kam, > 1.35x
  nayi. Naam wala 'DOBARA BANWAO' + wajah, exit code 2. Peacock par test: ek part ka rang badla (95% naye rang), ek ko dhundhla kiya (kinare 56%) -> dono pakde, baaki
  7 theek. Ye jaanch AI part ki sundarta nahi, sirf 'reference se mel' dekhti hai (zyada detail jo sahi ho wo 1.35x tak chalti).
- `--dpi 600`: sirf jab mill khud maange; tool chetavni deta hai.

## Photopea guide (user ke liye, puche to)
- New Project: 3535×3535, 300 DPI, Pixels/Inch, Transparent, RGB, 8 bit, sRGB. Ya seedha `channel_01` kholo, size apne aap set ho jayega.
- `File → Open & Place` se baaki channels add karo. Layers panel: `Window → Layers`. Order se farak nahi padta, channels overlap nahi karte.
- Kaam PNG me karo, mill ko TIFF bhejo. RGB/CMYK mill se confirm karna user ki zimmedari hai.
