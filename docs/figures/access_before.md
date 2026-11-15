# État initial : V, A400, E avant aménagement

## Tuiles réelles (moyennes par zone)

| split | zone | tuiles | A400_moy | A400_med | E_moy | E_indefini | V_moy |
|---|---|---|---|---|---|---|---|
| test | gueliz | 9 | 0.387 | 0.362 | 0.471 | 0 | 0.016 |
| train | mabrouka | 9 | 0.356 | 0.374 | 0.312 | 1 | 0.01 |
| train | massira | 9 | 0.136 | 0.177 | 0.548 | 0 | 0.0 |
| val | hivernage | 9 | 0.502 | 0.466 | 0.381 | 3 | 0.014 |

## Quartiers (4 tuiles disjointes ; E entre tuiles)

| zone | tuiles_disjointes | origines | A400 | E_tuiles | A400_min_tuile | A400_max_tuile |
|---|---|---|---|---|---|---|
| gueliz | 4 | 2311 | 0.418 | 0.133 | 0.0 | 0.867 |
| hivernage | 4 | 456 | 0.465 | 0.079 | 0.079 | 1.0 |
| mabrouka | 4 | 2093 | 0.352 | 0.393 | 0.0 | 0.607 |
| massira | 4 | 2458 | 0.151 | 0.76 | 0.0 | 0.24 |

### Détail par tuile

| tile_id | split | n_origins | A400_before | E_before | V_before | usable |
|---|---|---|---|---|---|---|
| gueliz_00 | test | 433 | 0.462 | 0.174 | 0.003 | True |
| gueliz_01 | test | 534 | 0.17 | 0.416 | 0.002 | True |
| gueliz_02 | test | 564 | 0.0 | 1.0 | 0.0 | False |
| gueliz_10 | test | 546 | 0.78 | 0.584 | 0.028 | True |
| gueliz_11 | test | 761 | 0.288 | 0.239 | 0.009 | True |
| gueliz_12 | test | 836 | 0.037 | 0.926 | 0.0 | False |
| gueliz_20 | test | 577 | 0.867 | 0.761 | 0.053 | True |
| gueliz_21 | test | 728 | 0.515 | 0.074 | 0.024 | True |
| gueliz_22 | test | 737 | 0.362 | 0.062 | 0.024 | True |
| hivernage_00 | val | 17 | 0.647 | nan | 0.0 | False |
| hivernage_01 | val | 95 | 0.179 | 0.353 | 0.0 | True |
| hivernage_02 | val | 140 | 0.079 | 0.545 | 0.0 | True |
| hivernage_10 | val | 18 | 0.667 | nan | 0.0 | False |
| hivernage_11 | val | 102 | 0.402 | 0.215 | 0.01 | True |
| hivernage_12 | val | 225 | 0.164 | 0.111 | 0.011 | True |
| hivernage_20 | val | 95 | 1.0 | nan | 0.002 | True |
| hivernage_21 | val | 48 | 0.917 | 0.937 | 0.019 | True |
| hivernage_22 | val | 204 | 0.466 | 0.126 | 0.085 | True |
| mabrouka_00 | train | 595 | 0.607 | 0.227 | 0.024 | True |
| mabrouka_01 | train | 250 | 0.216 | 0.477 | 0.02 | True |
| mabrouka_02 | train | 80 | 0.0 | nan | 0.0 | True |
| mabrouka_10 | train | 752 | 0.519 | 0.28 | 0.024 | True |
| mabrouka_11 | train | 580 | 0.531 | 0.178 | 0.02 | True |
| mabrouka_12 | train | 281 | 0.374 | 0.444 | 0.0 | True |
| mabrouka_20 | train | 978 | 0.273 | 0.178 | 0.001 | True |
| mabrouka_21 | train | 759 | 0.44 | 0.269 | 0.001 | True |
| mabrouka_22 | train | 440 | 0.245 | 0.444 | 0.0 | True |
| massira_00 | train | 276 | 0.007 | 0.977 | 0.0 | True |
| massira_01 | train | 372 | 0.005 | 0.977 | 0.0 | False |
| massira_02 | train | 316 | 0.0 | 1.0 | 0.0 | False |
| massira_10 | train | 406 | 0.219 | 0.192 | 0.0 | False |
| massira_11 | train | 590 | 0.236 | 0.192 | 0.0 | False |
| massira_12 | train | 870 | 0.205 | 0.685 | 0.0 | False |
| massira_20 | train | 734 | 0.131 | 0.192 | 0.0 | False |
| massira_21 | train | 1066 | 0.177 | 0.221 | 0.0 | True |
| massira_22 | train | 1132 | 0.24 | 0.493 | 0.003 | True |

## Tuiles synthétiques

| zone | tuiles | A400_moy | A400_med | E_moy | E_indefini | V_moy |
|---|---|---|---|---|---|---|
| synthetic | 600 | 0.443 | 0.431 | 0.422 | 0 | 0.013 |

| A400 | tuiles |
|---|---|
| 0–20 % | 129 |
| 20–40 % | 148 |
| 40–60 % | 135 |
| 60–80 % | 124 |
| 80–100 % | 64 |
