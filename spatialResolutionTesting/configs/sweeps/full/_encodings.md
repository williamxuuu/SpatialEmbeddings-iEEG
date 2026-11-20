Encodings shared by every full-run sweep (keep names identical across files:
scripts/analyze_full.py matches contrasts by these names).

| name | what | question it serves |
|---|---|---|
| none | no location | floor |
| electrode_id | one learned vector per electrode | identity tag, no anatomy / no sharing (control) |
| coords | Fourier xyz | continuous position |
| lobes, ho, destrieux, schaefer400 | hard atlas regions, coarse -> fine | Exp 1 resolution |
| grid20 | 20 mm cubes | resolution without anatomy |
| destrieux_shuffled | Destrieux codes shuffled among a person's electrodes | right location vs any consistent tag |
| destrieux_gauss3 | soft Destrieux, Gaussian sigma 3 mm | Exp 2 main soft setting |
| destrieux_gauss3_randnbr | gauss3 weights, random secondary regions | right neighbours vs spreading weight |
| ho_prob_soft / ho_prob_argmax | Harvard-Oxford probability mix vs its most likely region | second, independent soft |
| destrieux_nearest, destrieux_gauss3_nearest | cross-subject only: unseen regions -> nearest seen | soft vs hard at equal coverage |
