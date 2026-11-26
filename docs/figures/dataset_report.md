# Jeu de données d'entraînement

| split | tiles | pairs | patterns_mean | soft_minus_exact_mean | soft_minus_exact_abs_max | soft_exact_within_1pt |
|---|---|---|---|---|---|---|
| test | 7 | 32 | 7.5714 | -0.0 | 0.0 | 1.0 |
| test_synth | 44 | 468 | 78.1364 | 0.0025 | 0.0622 | 0.9038 |
| train | 500 | 5223 | 89.224 | 0.0034 | 0.2112 | 0.8943 |
| val | 5 | 36 | 11.0 | 0.0005 | 0.0049 | 1.0 |
| val_synth | 66 | 684 | 76.4545 | 0.0023 | 0.0963 | 0.9152 |

## Fidélité de l'A400 doux (perte de condition) sur les masques cibles

- écart moyen (doux − exact) : +0.0031
- |écart| médian : 0.0000 ; 95e centile : 0.0215 ; max : 0.2112
- part des cibles à ≤ 1 point : 89.8%

L'estimation pixel par pixel ne peut que SURestimer (un point d'accès par pixel et par
tronçon, contre un seul par polygone et par tronçon pour la métrique exacte).
