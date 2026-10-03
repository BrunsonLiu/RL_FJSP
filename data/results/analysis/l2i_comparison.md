# L2I vs. ours — per-instance comparison

| Inst | EF | L2I | Ours | Lit | L2I gap | Ours gap | Ours − L2I |
|---|---|---|---|---|---|---|---|
| mk01 | 57 | 60 | 42 | 40 | +50.00% | +5.00% | -18 |
| mk02 | 62 | 43 | 28 | 26 | +65.38% | +7.69% | -15 |
| mk03 | 331 | 257 | 204 | 204 | +25.98% | +0.00% | -53 |
| mk04 | 91 | 88 | 73 | 60 | +46.67% | +21.67% | -15 |
| mk05 | 220 | 228 | 176 | 172 | +32.56% | +2.33% | -52 |
| mk06 | 79 | 142 | 68 | 58 | +144.83% | +17.24% | -74 |
| mk07 | 204 | 199 | 143 | 139 | +43.17% | +2.88% | -56 |
| mk08 | 618 | 673 | 523 | 523 | +28.68% | +0.00% | -150 |
| mk09 | 433 | 580 | 332 | 307 | +88.93% | +8.14% | -248 |
| mk10 | 406 | 454 | 224 | 197 | +130.46% | +13.71% | -230 |
| mk11 | 706 | 795 | 619 | 615 | +29.27% | +0.65% | -176 |
| mk12 | 700 | 689 | 508 | 508 | +35.63% | +0.00% | -181 |
| mk13 | 622 | 773 | 416 | 430 | +79.77% | -3.26% | -357 |
| mk14 | 833 | 988 | 694 | 694 | +42.36% | +0.00% | -294 |
| mk15 | 549 | 721 | 370 | 341 | +111.44% | +8.50% | -351 |

**Mean gap:** L2I = 63.68%, ours = 5.64%
**Mean absolute diff:** ours − L2I = -151.3

## Notes

L2I is trained from scratch (random schedule init) with 5 seeds × 200 steps, REINFORCE, 10-D state, 4-op action space (reassign, swap_machine, swap_order, swap_order_across).
L2I does NOT use any pretrained RL initial schedule, ILS perturbation, or post-processing.
This is the most direct comparison of the L2I paradigm (operator selection RL) on FJSP.