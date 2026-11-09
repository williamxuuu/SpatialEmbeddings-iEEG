# How each electrode's region mix is worked out

Code: `spatialResolutionTesting/src/locemb/spatial/assign.py` and `atlases.py`. Numbers below are from the real Lite electrode positions (707 electrodes, 6 people).

## What goes in

- **The electrode's position:** x, y, z in millimetres in the standard "MNI" brain. Neuroprobe provides these, with each person's brain already warped onto that one standard brain.
- **A brain map (atlas):**
  - **Destrieux**, a 3D grid of 2 mm cubes covering the cortex (the outer layer of the brain). Each cube is tagged with one of 148 regions (74 per side), or nothing.
  - White matter and fluid are not tagged.
  - Where Destrieux has nothing, a second map (Harvard-Oxford, 1 mm cubes) fills in 14 deep structures: hippocampus, amygdala, thalamus, caudate, putamen, pallidum and accumbens, each on both sides.
  - That makes **162 possible regions**.

## Steps, for one electrode (main setting: 3 mm weighted)

1. **Place test points around the electrode.** Take every point on a 1 mm grid that lies within 9 mm of it: **3,071 points**.
2. **Give each point a weight by its distance** *d* from the electrode: weight = e^(−d² / (2 × 3²)), a bell curve with a 3 mm width.

   | Distance | 0 mm | 1 mm | 2 mm | 3.5 mm | 5 mm | 6 mm | 9 mm |
   |---|---|---|---|---|---|---|---|
   | Weight of one point | 1.00 | 0.95 | 0.80 | **0.51** | 0.25 | 0.14 | 0.01 |

   - Why 3 mm: McCarty et al. (eNeuro 2022) found that the signals of two sEEG contacts in grey matter resemble each other half as much about 3.6 mm apart. That's with the average-of-all-electrodes reference we also use.
   - Points farther out count less each, but there are many more of them. Only 29% of the total weight comes from within 3.5 mm, and 58% from within 5 mm.
3. **Tag each point with a region.**
   - Find which map cube the point falls in, and take that cube's region.
   - Try the cortex map first; if it has nothing there, try the deep-structure map. If neither does, the point is untagged.
   - Left vs. right for deep structures is decided by the sign of x (x ≥ 0 is right).
   - The points are 1 mm apart. Cortex cubes are 2 mm wide, so each is hit by about 8 points; deep-structure cubes are 1 mm, so one point each.
4. **Add up the weights of the tagged points in each region.** Untagged points are ignored: **shares are out of the tagged weight, not the total.**
5. **If no point within 9 mm is tagged**, the electrode gets "unknown".
6. **Turn the sums into shares** that add up to 1.
7. **Trim.** Always keep the biggest region, drop any other region under 5%, keep at most 8 regions, then rescale so the shares add up to 1 again.
8. **Make the list of known regions.**
   - Every region that appears (after trimming) for any **training** person goes on the list.
   - Test people never add regions.
   - Left and right are separate regions.
9. **Handle test people's regions that aren't on the list.**
   - **Default:** they become "unknown", one shared code. If an electrode has several such regions, their shares are added together on "unknown".
   - **`_nearest` rows only:** they become the nearest region that *is* on the list. The search goes outward from the electrode, up to 20 mm. If nothing is found, the electrode gets "unknown".
   - This changes only test people's electrodes.
10. **How the model uses it.**
    - Each region on the list, plus "unknown", has its own learned vector of 64 numbers.
    - The electrode's location vector is the share-weighted sum of its regions' vectors.
    - That vector is added to every piece of the electrode's signal before the transformer sees it.
    - With one region per electrode ("hard"), the single share is 1, so the vector is just that region's.
    - Nothing else about the model differs between settings, and the recorded signal itself is never changed. The distance weights only decide how much each region counts in the location description.

**"Hard" (one region per electrode), for comparison:**
- Take the region of the cube the electrode itself is in.
- If that cube is untagged, use the nearest tagged point within 5 mm.
- If there is none, use "unknown".
- Then steps 8–10 as above.

## Worked examples (subject 3, real positions)

**Electrode F3aOF6 at (−41.2, 25.7, −8.1).** Its own point is in the left orbital part of the inferior frontal gyrus. Of the 3,071 points, 1,037 are untagged.

| Region | Points | Summed weight | Share | After trimming |
|---|---|---|---|---|
| L inferior frontal gyrus, orbital part | 756 | 192.2 | 0.608 | **0.657** |
| L orbital gyri | 554 | 60.1 | 0.190 | **0.205** |
| L anterior circular sulcus of the insula | 220 | 40.3 | 0.128 | **0.138** |
| L lateral fissure, anterior horizontal part | 188 | 11.6 | 0.037 | dropped (< 5%) |
| 3 more regions | 316 | 11.8 | 0.038 in total | dropped |

**Electrode F3aOF7 at (−44.4, 24.5, −7.7)**, about 3 mm from the first. 1,581 of the 3,071 points are untagged.
- After trimming it keeps three regions: the same main region **0.826**, L orbital gyri **0.092**, and the triangular part of the inferior frontal gyrus **0.081**.
- With the old 5 mm ball, this electrode had been trimmed back to one region.

Over all 707 electrodes, 77% keep more than one region, and the main region keeps 0.75 of the weight on average.

## The other settings

- **`destrieux_sphere10` (much wider check):**
  - Every point within 10 mm (4,169 points) counts equally; there is no distance weighting.
  - Otherwise the same steps.
  - 92% of electrodes keep more than one region, and some hit the cap of 8.
- **`destrieux_gauss3_randnbr` (control):**
  - Do steps 1–9.
  - Keep each electrode's main region and all the shares.
  - Replace every other region with one picked at random from the whole list of known regions; it just can't be the main region.
  - The random picks don't have to be nearby, and two picks can land on the same region.
  - The picks are fixed by the run's seed.
- **`ho_prob_soft` (a second, different kind of mix):**
  - Uses the Harvard-Oxford probability maps (2 mm cubes). For each cube, they give the % of people in the atlas's sample in whom that spot was each region.
  - 48 cortex regions, split into left and right at x = 0 (96), plus 14 deep structures: **110 regions**.
  - Only the electrode's own cube is read (no ball, no blurring).
  - Those percentages are the shares; then steps 6–10.
  - If every percentage is 0 there, it uses the nearest point within 5 mm that has any.
- **`ho_prob_argmax`:** the same lookup, keeping only the most likely region, with share 1.
- **`coords`:** no regions.
  - x, y, z go through sine and cosine waves at 16 sizes, from 4 mm to 256 mm, giving 96 numbers.
  - A learned layer turns those into the same 64-number vector.
- **`none`:** a vector of zeros.

## Worth checking in review

These are choices I made or limits of the method, not settled facts.

1. **Shares ignore untagged points.** If the electrode sits in white matter and only a few cortex points are nearby, those few decide everything (F3aOF7: 1,581 of 3,071 points untagged). The alternative would be to count white matter as its own share.
2. **The 5% cut and the cap of 8 are my picks**, not from a paper. The 5% cut turns some near-border electrodes back into one region.
3. **The 3 mm width is anchored to a measurement, but loosely.**
   - McCarty's figure is the similarity between *two* contacts. That spreads wider than what one contact picks up; under a simple bell-curve model, about √2 wider, which would mean a width nearer 2 mm.
   - Their figure is for high-frequency activity, and our model uses 1–200 Hz.
   - Borders are also only as sharp as the map's 2 mm cubes.
4. **Destrieux covers the cortex only.** Many sEEG contacts are in white matter.
   - Hard gives "unknown" to electrodes with no tagged point within 5 mm: 2.1% of the training people's electrode weight.
   - 3 mm weighted reaches 9 mm, so fewer: 0.6%. The 10 mm ball: 0.4%.
   - So soft also places a few white-matter electrodes that hard can't.
5. **Deep structures come from a different map** (Harvard-Oxford). It is used point by point wherever Destrieux has nothing.
6. **Left/right is decided by the sign of x** for deep structures and for Harvard-Oxford. Electrodes right at the midline could land on the wrong side.
7. **Positions come from one standard brain.** Errors of a few mm in electrode location or in the warp change the mix, and a mix may help partly by absorbing those errors.
8. **The random-neighbour control picks from all regions**, not just nearby ones.
9. **The "nearest seen" control is mine**, not from a paper. It searches up to 20 mm.
10. **Code detail:** in the mix settings, the backup search for untagged electrodes reaches 5 mm (10 mm for the 10 mm ball). That is inside the area already searched, so it never adds anything. It's harmless, but it isn't a real backup.
