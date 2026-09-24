# TSRB

Experimental Setup. We evaluate on SUIM and TrashCan . SUIM uses a fixed 746/83/276 train/dev/controlledevaluation split from the official train val pool (TEST unused). TrashCan uses the Instance version, with a 5,450/615 train/dev split of the official train set and the 1,147-image official validation set; annotations are converted to 23 semantic classes with cross-class overlaps ignored. All methods use DeepLabV3-MobileNetV3-Large at 512 × 512 for 100 epochs with AdamW (lr 10−3 , weight decay 10−4 , effective batch 8). For each dataset, enhanced inputs are precomputed by a frozen UIEB-trained UIEC2 -Net RGB front-end with cross-feature compensation and bounded detail residual. KD methods share this cache, a frozen seed-0 same-architecture teacher trained on the same enhanced domain (38.60/28.52 mIoU on SUIM/TrashCan), and seed-specific random Student initializations. SUIM selects by dev mIoU, whereas TrashCan uses epoch 100. KD baselines use fixed settings without evaluation retuning (PixelKD/CWD: T = 4, weights 1 and 3; CIRKD/DIST follow their official implementations).



