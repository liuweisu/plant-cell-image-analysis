## Leaves: segmentation and disease severity (synthetic, n=30 per background)

| background | leaf IoU | lesion IoU | severity error (mean abs, % points) | without margin-lesion recovery |
| --- | --- | --- | --- | --- |
| paper | 0.975 | 0.898 | 0.92 | 4.98 |
| bench | 0.985 | 0.905 | 0.89 | 4.99 |
| soil | 0.953 | 0.570 | 4.16 | 4.16 |

## Nuclei: counting touching nuclei (synthetic, n=10, 40 nuclei each, IoU >= 0.5 matching)

| method | precision | recall | F1 | count error |
| --- | --- | --- | --- | --- |
| threshold + connected components | 0.82 | 0.49 | 0.61 | 16.2 |
| distance-transform watershed | 0.99 | 0.97 | 0.98 | 0.8 |

## Real image: scikit-image `human_mitosis` (512x512 px)

311 nuclei segmented; 19 flagged as unusually bright (candidate mitoses, red boxes).
