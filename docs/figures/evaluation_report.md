# Évaluation : modèles A/B/C, ablations et références

Métriques **exactes** : module indépendant, distances réseau (Dijkstra). E est toujours
lu avec A400. Ces indicateurs mesurent l'accès piéton à 400 m et la végétation ; ils ne
mesurent pas le bonheur des habitants.

Exécutions : A-full-s42, A-full-s43, A-full-s44, B-full-s42, B-full-s43, B-full-s44, C-full-s42, C-full-s43, C-full-s44, C-no_attention-s42, C-no_diversity-s42, C-no_modulation-s42, C-no_projection-s42. 3 propositions par problème (mêmes z pour tous les modèles) ; aléatoire : 10 tirages.

## Problèmes

| split | target_type | atteignable_glouton | problèmes |
|---|---|---|---|
| test | a posteriori | oui | 32 |
| test | inatteignable (+0.10) | non | 28 |
| test | inatteignable (+0.25) | non | 28 |
| test_synth | a posteriori | non | 12 |
| test_synth | a posteriori | oui | 456 |
| test_synth | inatteignable (+0.10) | non | 152 |
| test_synth | inatteignable (+0.25) | non | 152 |

« Atteignable » = atteint par le glouton. Le glouton n'est pas optimal : c'est une
estimation **par le bas** de ce qui est atteignable.

## Test synthétique (rapporté à part)

### Sorties retenues

| méthode | graines | |â−a| global | |â−a| atteignable | |â−a| non atteignable | A400 après | ΔA400 | E après | V après | valide | diversité | E défini | ms / carte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | 0.069 ± 0.006 | 0.037 ± 0.003 | 0.114 ± 0.010 | 86.5 ± 0.6 % | 39.1 ± 0.6 % | 0.721 ± 0.007 | 2.5 ± 0.2 % | 16.0 ± 10.6 % | 0.161 ± 0.058 | 100% | 10.1 |
| B | 3 | 0.066 ± 0.007 | 0.039 ± 0.004 | 0.104 ± 0.011 | 86.9 ± 1.1 % | 39.5 ± 1.1 % | 0.721 ± 0.019 | 2.5 ± 0.1 % | 18.6 ± 3.8 % | 0.110 ± 0.009 | 100% | 11.0 |
| C | 3 | 0.059 ± 0.002 | 0.034 ± 0.002 | 0.096 ± 0.004 | 87.7 ± 0.5 % | 40.3 ± 0.5 % | 0.738 ± 0.008 | 3.0 ± 0.0 % | 100.0 ± 0.0 % | 0.039 ± 0.003 | 100% | 12.9 |
| C – sans attention | 1 | 0.059 | 0.034 | 0.096 | 87.5 % | 40.1 % | 0.729 | 3.1 % | 100.0 % | 0.052 | 100% | 11.0 |
| C – sans diversité | 1 | 0.051 | 0.030 | 0.082 | 88.9 % | 41.5 % | 0.747 | 2.9 % | 100.0 % | 0.033 | 100% | 12.8 |
| C – sans modulation | 1 | 0.056 | 0.035 | 0.086 | 88.7 % | 41.3 % | 0.754 | 3.1 % | 100.0 % | 0.059 | 100% | 11.9 |
| C – sans projection | 1 | 0.062 | 0.033 | 0.103 | 87.2 % | 39.8 % | 0.735 | 2.7 % | 33.2 % | 0.085 | 100% | 12.8 |
| aléatoire | — | 0.109 | 0.071 | 0.164 | 81.2 % | 33.8 % | 0.608 | 3.7 % | 100.0 % | 0.585 | 100% | — |
| glouton | — | 0.040 | 0.017 | 0.074 | 89.7 % | 42.3 % | 0.788 | 3.2 % | 100.0 % | — | 100% | — |
| glouton ciblé | — | 0.038 | 0.013 | 0.075 | 89.3 % | 41.9 % | 0.781 | 3.2 % | 100.0 % | — | 100% | — |

Moyenne ± écart-type **entre graines** quand il y en a plusieurs. « valide » : dans R, budget respecté, ajouts adjacents au réseau.

### Violations des sorties brutes (avant projection)

| méthode | hors de R | budget dépassé | morceaux détachés | valide | px hors de R (moy.) |
|---|---|---|---|---|---|
| A | 0.159 | 0.005 | 0.793 | 0.16 | 0.358 |
| B | 0.147 | 0.013 | 0.748 | 0.186 | 0.249 |
| C | 0.0 | 0.13 | 0.504 | 0.424 | 0.0 |
| C – sans attention | 0.0 | 0.168 | 0.489 | 0.406 | 0.0 |
| C – sans diversité | 0.0 | 0.118 | 0.379 | 0.55 | 0.0 |
| C – sans modulation | 0.0 | 0.239 | 0.372 | 0.458 | 0.0 |
| C – sans projection | 0.119 | 0.012 | 0.604 | 0.332 | 0.183 |

### Effet de la projection (mêmes propositions, brutes vs projetées)

| méthode | valid (brute) | valid (projetée) | abs_err (brute) | abs_err (projetée) | A400 (brute) | A400 (projetée) | fill (brute) | fill (projetée) |
|---|---|---|---|---|---|---|---|---|
| A | 0.16 | 1.0 | 0.069 | 0.075 | 0.865 | 0.856 | 0.464 | 0.44 |
| B | 0.186 | 1.0 | 0.066 | 0.068 | 0.869 | 0.866 | 0.478 | 0.462 |
| C | 0.424 | 1.0 | 0.058 | 0.059 | 0.88 | 0.877 | 0.697 | 0.674 |
| C – sans attention | 0.406 | 1.0 | 0.055 | 0.059 | 0.881 | 0.875 | 0.749 | 0.723 |
| C – sans diversité | 0.55 | 1.0 | 0.051 | 0.051 | 0.89 | 0.889 | 0.648 | 0.633 |
| C – sans modulation | 0.458 | 1.0 | 0.054 | 0.056 | 0.89 | 0.887 | 0.778 | 0.729 |
| C – sans projection | 0.332 | 1.0 | 0.062 | 0.064 | 0.872 | 0.868 | 0.545 | 0.536 |

### Par budget

| b_frac | abs_err – A | abs_err – B | abs_err – C | abs_err – aléatoire | abs_err – glouton ciblé | A400 – A | A400 – B | A400 – C | A400 – aléatoire | A400 – glouton ciblé |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.115 | 0.116 | 0.09 | 0.168 | 0.054 | 0.783 | 0.782 | 0.812 | 0.712 | 0.843 |
| 0.02 | 0.071 | 0.062 | 0.062 | 0.122 | 0.039 | 0.861 | 0.872 | 0.874 | 0.797 | 0.892 |
| 0.03 | 0.048 | 0.047 | 0.047 | 0.09 | 0.032 | 0.901 | 0.904 | 0.901 | 0.846 | 0.912 |
| 0.05 | 0.039 | 0.038 | 0.036 | 0.055 | 0.029 | 0.915 | 0.919 | 0.92 | 0.893 | 0.924 |

## Test réel : Guéliz (test final)

### Sorties retenues

| méthode | graines | |â−a| global | |â−a| atteignable | |â−a| non atteignable | A400 après | ΔA400 | E après | V après | valide | diversité | E défini | ms / carte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | 0.110 ± 0.019 | 0.031 ± 0.035 | 0.156 ± 0.010 | 65.9 ± 3.4 % | 15.0 ± 3.4 % | 0.408 ± 0.061 | 2.5 ± 0.1 % | 6.8 ± 5.6 % | 0.136 ± 0.028 | 100% | 10.6 |
| B | 3 | 0.101 ± 0.008 | 0.017 ± 0.014 | 0.150 ± 0.006 | 64.8 ± 1.1 % | 13.9 ± 1.1 % | 0.382 ± 0.012 | 2.5 ± 0.0 % | 3.5 ± 4.2 % | 0.110 ± 0.043 | 100% | 11.1 |
| C | 3 | 0.108 ± 0.005 | 0.016 ± 0.006 | 0.161 ± 0.004 | 62.8 ± 0.5 % | 11.9 ± 0.5 % | 0.372 ± 0.008 | 2.5 ± 0.1 % | 100.0 ± 0.0 % | 0.039 ± 0.021 | 100% | 12.9 |
| C – sans attention | 1 | 0.109 | 0.016 | 0.162 | 62.8 % | 11.9 % | 0.371 | 2.4 % | 100.0 % | 0.027 | 100% | 11.0 |
| C – sans diversité | 1 | 0.105 | 0.011 | 0.159 | 63.1 % | 12.2 % | 0.372 | 2.5 % | 100.0 % | 0.012 | 100% | 12.8 |
| C – sans modulation | 1 | 0.107 | 0.016 | 0.159 | 62.9 % | 12.0 % | 0.370 | 2.4 % | 100.0 % | 0.096 | 100% | 11.9 |
| C – sans projection | 1 | 0.109 | 0.032 | 0.154 | 65.7 % | 14.8 % | 0.412 | 2.6 % | 11.0 % | 0.067 | 100% | 12.8 |
| aléatoire | — | 0.098 | 0.006 | 0.150 | 63.9 % | 13.0 % | 0.386 | 3.3 % | 100.0 % | 0.036 | 100% | — |
| glouton | — | 0.094 | 0.001 | 0.146 | 64.4 % | 13.5 % | 0.390 | 3.3 % | 100.0 % | — | 100% | — |
| glouton ciblé | — | 0.093 | 0.000 | 0.146 | 64.3 % | 13.4 % | 0.388 | 3.2 % | 100.0 % | — | 100% | — |

Moyenne ± écart-type **entre graines** quand il y en a plusieurs. « valide » : dans R, budget respecté, ajouts adjacents au réseau.

### Violations des sorties brutes (avant projection)

| méthode | hors de R | budget dépassé | morceaux détachés | valide | px hors de R (moy.) |
|---|---|---|---|---|---|
| A | 0.792 | 0.0 | 0.577 | 0.068 | 7.378 |
| B | 0.866 | 0.0 | 0.564 | 0.035 | 8.178 |
| C | 0.0 | 0.011 | 0.091 | 0.902 | 0.0 |
| C – sans attention | 0.0 | 0.0 | 0.288 | 0.712 | 0.0 |
| C – sans diversité | 0.0 | 0.0 | 0.152 | 0.848 | 0.0 |
| C – sans modulation | 0.0 | 0.0 | 0.023 | 0.977 | 0.0 |
| C – sans projection | 0.867 | 0.0 | 0.398 | 0.11 | 11.644 |

### Effet de la projection (mêmes propositions, brutes vs projetées)

| méthode | valid (brute) | valid (projetée) | abs_err (brute) | abs_err (projetée) | A400 (brute) | A400 (projetée) | fill (brute) | fill (projetée) |
|---|---|---|---|---|---|---|---|---|
| A | 0.068 | 1.0 | 0.11 | 0.103 | 0.659 | 0.634 | 0.153 | 0.13 |
| B | 0.035 | 1.0 | 0.101 | 0.098 | 0.648 | 0.639 | 0.152 | 0.128 |
| C | 0.902 | 1.0 | 0.108 | 0.108 | 0.629 | 0.628 | 0.15 | 0.147 |
| C – sans attention | 0.712 | 1.0 | 0.109 | 0.109 | 0.628 | 0.628 | 0.132 | 0.128 |
| C – sans diversité | 0.848 | 1.0 | 0.105 | 0.105 | 0.631 | 0.631 | 0.145 | 0.144 |
| C – sans modulation | 0.977 | 1.0 | 0.107 | 0.107 | 0.629 | 0.629 | 0.116 | 0.116 |
| C – sans projection | 0.11 | 1.0 | 0.109 | 0.104 | 0.657 | 0.633 | 0.169 | 0.142 |

### Par budget

| b_frac | abs_err – A | abs_err – B | abs_err – C | abs_err – aléatoire | abs_err – glouton ciblé | A400 – A | A400 – B | A400 – C | A400 – aléatoire | A400 – glouton ciblé |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.113 | 0.108 | 0.11 | 0.099 | 0.093 | 0.653 | 0.656 | 0.626 | 0.638 | 0.643 |
| 0.02 | 0.109 | 0.105 | 0.106 | 0.098 | 0.093 | 0.657 | 0.646 | 0.63 | 0.638 | 0.643 |
| 0.03 | 0.111 | 0.097 | 0.106 | 0.096 | 0.093 | 0.661 | 0.64 | 0.63 | 0.641 | 0.643 |
| 0.05 | 0.11 | 0.095 | 0.11 | 0.097 | 0.093 | 0.665 | 0.652 | 0.627 | 0.639 | 0.643 |
