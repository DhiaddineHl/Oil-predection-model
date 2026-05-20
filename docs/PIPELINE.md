# Documentation du pipeline de prédiction du prix du pétrole

Ce document décrit en détail les différentes phases du pipeline, le jeu de données construit, les opérations de prétraitement, les modèles utilisés et le processus de comparaison.

---

## 1. Vue d'ensemble du pipeline

Le projet est structuré en six phases principales, exécutées séquentiellement de la donnée brute jusqu'à la mise en service :

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Ingestion   │ ─▶ │  2. Fusion &     │ ─▶ │  3. Feature      │
│     (.xls/.xlsx)│    │     nettoyage    │    │     engineering  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  6. Déploiement │ ◀─ │  5. Comparaison  │ ◀─ │  4. Entraînement │
│  (API + Stream.)│    │     & analyse    │    │     des modèles  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

### Phase 1 — Ingestion des sources brutes

Trois fichiers Excel sont chargés depuis `data/raw/` (cf. `src/data/preprocessing.py`) :

| Fichier | Source | Granularité | Période | Moteur Excel |
|---|---|---|---|---|
| `RWTCd.xls` | EIA (U.S. Energy Information Administration) | Quotidienne | 1986 → présent | `xlrd` |
| `data_gpr_daily_recent.xls` | Caldara & Iacoviello — Geopolitical Risk Index | Quotidienne | 1985 → présent | `xlrd` |
| `Global_Policy_Uncertainty_Data.xlsx` | Davis (policyuncertainty.com) — GEPU | Mensuelle | 1997 → présent | `openpyxl` |

Chaque fonction de chargement (`load_wti`, `load_gpr`, `load_epu`) effectue :
- la suppression des colonnes en double (`df.loc[:, ~df.columns.duplicated()]`) — sans quoi `pd.to_datetime` plante avec *"cannot assemble with duplicate keys"* ;
- une détection robuste de la colonne `date` (sélection positionnelle plutôt que par nom) ;
- la conversion explicite des types numériques avec `errors="coerce"` (les valeurs non parsables deviennent `NaN`).

### Phase 2 — Fusion et alignement temporel

La fonction `merge_sources` réalise une fusion en trois étapes :

1. **Jointure interne** entre WTI et GPRD sur la colonne `date` (jours communs uniquement — élimine les week-ends et jours fériés non négociés).
2. **Jointure gauche** avec EPU sur `(Year, Month)` — l'EPU est mensuel donc la même valeur est reportée sur tous les jours du mois.
3. **Forward-fill** des colonnes `GEPU_current` et `GEPU_ppp` pour garantir aucun trou intra-mois.
4. **Filtrage temporel** : fenêtre effective fixée à `1997-01-01 → 2025-11-30` (intersection des disponibilités des trois sources).

### Phase 3 — Feature engineering

Calcul de variables explicatives à partir des séries brutes (cf. `add_features`).

### Phase 4 — Entraînement des modèles

Trois familles de modèles, entraînées indépendamment :
- **Baseline** : naïf (persistence) + Random Forest (scikit-learn).
- **Deep learning** : BiLSTM avec attention + CNN-GRU hybride (PyTorch).
- **Avancé** : Temporal Fusion Transformer simplifié (PyTorch + quantiles).

### Phase 5 — Évaluation et comparaison

Toutes les métriques sont agrégées par `src/evaluation/compare_models.py` qui produit un CSV, des graphiques comparatifs et un rapport Markdown.

### Phase 6 — Déploiement

- **API FastAPI** (`src/app/api.py`) sert tous les modèles via `/predict`, `/models`, `/health`.
- **Dashboard Streamlit** (`src/app/streamlit_app.py`) consomme l'API et offre prédiction live, analyse historique et comparaison.

---

## 2. Description du jeu de données construit

Après fusion et feature engineering, le fichier `data/processed/full_dataset.csv` contient les colonnes suivantes.

### 2.1 Colonnes de référence (non utilisées comme features)

| Colonne | Type | Justification |
|---|---|---|
| `date` | datetime | Clé temporelle pour le tri chronologique et les splits. **Exclue** des features pour éviter toute fuite via `dayofyear` ou index. |
| `target` | float | Prix WTI du **lendemain** (`WTI_price.shift(-1)`). C'est la variable à prédire. **Exclue** des features (sinon fuite triviale). |
| `Year`, `Month` | int | Clés de jointure mensuelle vers EPU. Conservées dans le CSV pour traçabilité mais **exclues** des features (redondantes avec `month` et avec les lags). |

### 2.2 Variables de prix WTI (signal principal)

| Colonne | Calcul | Justification |
|---|---|---|
| `WTI_price` | Brut (jour t) | Niveau actuel du prix — composante autorégressive de base. C'est aussi le pivot pour calculer les lags, la rolling et le log-return. |
| `WTI_lag_1, 2, 3` | `shift(1..3)` | Capturent la **mémoire courte** du marché — la majorité du signal d'auto-corrélation est concentrée sur les 1-3 jours précédents (cf. ACF). |
| `WTI_lag_5` | `shift(5)` | Une semaine de trading — capte la saisonnalité hebdomadaire. |
| `WTI_lag_10` | `shift(10)` | Deux semaines — horizon typique des rapports EIA hebdomadaires. |
| `WTI_lag_21` | `shift(21)` | Un mois de trading — capte l'effet "rapport mensuel OPEP / inventaires". |
| `WTI_roll_mean_7`, `WTI_roll_mean_30` | Moyenne mobile sur 7 / 30 jours **décalée de 1** | Lisse le bruit haute fréquence et donne au modèle un point d'ancrage de "régime de prix". Le `shift(1)` préalable garantit qu'on n'utilise pas le jour courant (anti-leakage). |
| `WTI_roll_std_7`, `WTI_roll_std_30` | Écart-type mobile (idem) | Proxy de **volatilité réalisée** — essentiel car la volatilité du pétrole est elle-même prédictive (effet ARCH). |
| `WTI_log_return` | `log(P_t / P_{t-1})` | Variation relative log — stationnaire, mieux conditionnée pour les réseaux de neurones que le niveau brut. |

### 2.3 Variables géopolitiques (GPRD — Caldara & Iacoviello)

| Colonne | Description | Justification |
|---|---|---|
| `GPRD` | Indice quotidien composite (Threat + Acts) | Capte le **risque géopolitique global** : guerres, attentats, sanctions. Le pétrole est très sensible à ce facteur (Moyen-Orient, Russie). |
| `GPRD_ACT` | Sous-composante "Acts" (événements réalisés) | Mesure l'occurrence d'actes — choc immédiat. |
| `GPRD_THREAT` | Sous-composante "Threat" (menaces) | Mesure l'anticipation — souvent un signal précurseur. La séparation Acts/Threat permet au modèle de pondérer différemment l'incertitude prospective vs. les chocs concrétisés. |
| `GPRD_MA7`, `GPRD_MA30` | Moyennes mobiles 7 / 30 jours (fournies par la source) | Lissage déjà calculé par les auteurs — capture la tendance géopolitique de fond, moins bruitée que le quotidien. |

### 2.4 Variables d'incertitude économique (GEPU — Davis)

| Colonne | Description | Justification |
|---|---|---|
| `GEPU_current` | Indice mensuel d'incertitude économique globale (poids PIB courants) | Capte l'incertitude **macroéconomique** (politiques monétaires, élections, réformes fiscales) — distincte du risque géopolitique. |
| `GEPU_ppp` | Variante pondérée par PIB en parité de pouvoir d'achat | Donne plus de poids aux économies émergentes (Chine, Inde) — utile car la demande pétrolière y croît plus vite. |

### 2.5 Variable d'interaction

| Colonne | Calcul | Justification |
|---|---|---|
| `GPRD_x_GEPU` | `GPRD × GEPU_current` | Capture les **effets non-linéaires conjoints** : une crise géopolitique a un impact plus fort quand l'incertitude économique est déjà élevée (ex : guerre + récession). Permet aux modèles linéaires (RF, baseline) d'accéder à cette interaction sans devoir l'apprendre. |

### 2.6 Variables calendaires

| Colonne | Description | Justification |
|---|---|---|
| `day_of_week` | 0 = lundi, …, 4 = vendredi | Capte les effets jour-de-la-semaine (volumes plus faibles le vendredi, "Monday effect"). |
| `month` | 1–12 | Saisonnalité : la demande de pétrole varie selon les saisons (chauffage en hiver, conduite en été). |

### 2.7 Variable cible

| Colonne | Calcul | Justification |
|---|---|---|
| `target` | `WTI_price.shift(-1)` | Prix de **clôture du lendemain**. Régression à 1 pas — cohérent avec un cas d'usage de trading intra-quotidien. |

---

## 3. Opérations de prétraitement

### 3.1 Nettoyage source-par-source

1. **Suppression des colonnes en double** après lecture Excel (problème fréquent avec les fichiers `.xls` exportés contenant des en-têtes redondants).
2. **Détection robuste de la colonne date** par index plutôt que par nom — résistant aux variations de casse (`DATE`, `Date`, `date`, `DAY`, `month`).
3. **Conversion typée** : `pd.to_datetime(..., errors="coerce")` et `pd.to_numeric(..., errors="coerce")` — les valeurs non parsables deviennent `NaN` puis sont supprimées.
4. **Suppression des lignes à `date` ou `WTI_price` manquant** — sécurité contre les en-têtes vides à la fin du fichier.

### 3.2 Alignement temporel

- **Jointure interne** WTI ⋈ GPRD : ne conserve que les jours présents dans les deux sources (jours ouvrables nord-américains).
- **Jointure gauche** vers EPU sur `(Year, Month)` + forward-fill : duplique la valeur mensuelle sur chaque jour du mois.
- **Fenêtre temporelle** : `1997-01-01 → 2025-11-30` — borne basse imposée par EPU (qui commence en janvier 1997), borne haute fixée pour reproductibilité.

### 3.3 Feature engineering anti-fuite

Toutes les statistiques mobiles utilisent **`shift(1)` avant le rolling** :

```python
out["WTI_roll_mean_7"] = out["WTI_price"].shift(1).rolling(window=7).mean()
```

Sans ce décalage, la moyenne du jour `t` inclurait `WTI_price[t]` lui-même → fuite d'information vers la cible. Cette précaution est critique en finance.

### 3.4 Suppression des lignes initiales invalides

Après l'ajout des lags et rollings, les **31 premières lignes** contiennent des `NaN` (lag maximum = 21, rolling maximum = 30). Elles sont écartées via `dropna` lors du build dataset.

### 3.5 Split chronologique (jamais de shuffle)

```
train : … → 2020-12-31      (≈ 24 ans, ~5 800 jours)
val   : 2021-01-01 → 2022-12-31  (~500 jours)
test  : 2023-01-01 → 2025-11-30  (~720 jours)
```

**Pourquoi pas de mélange aléatoire ?**
Une série temporelle a une dépendance temporelle — mélanger ferait fuiter de l'information du futur vers l'entraînement et donnerait des métriques artificiellement bonnes (le modèle "tricherait" en voyant des valeurs futures pendant l'entraînement).

### 3.6 Standardisation (StandardScaler)

- **Fit uniquement sur le train** — jamais sur val/test.
- Transformation appliquée à train, val et test avec les mêmes statistiques.
- Sérialisée via `joblib.dump` dans `models/scaler.pkl` pour rejouer la transformation à l'inférence (API).

**Pourquoi indispensable pour le deep learning ?** Les réseaux de neurones (LSTM, GRU, Transformer) convergent beaucoup plus rapidement et de manière stable quand les entrées ont une moyenne nulle et un écart-type unitaire. Le Random Forest n'en a pas besoin (insensible aux échelles), mais on garde le même pipeline pour cohérence.

### 3.7 Fenêtrage pour le deep learning

Une classe `TimeSeriesDataset` produit des fenêtres glissantes de longueur `seq_len=30` :

```
X[i] = features[i : i+30]    # forme (30, n_features)
y[i] = target[i+29]          # prix du lendemain de la dernière ligne
```

**Pourquoi `seq_len = 30` ?**
- C'est environ un mois de trading — assez long pour capter la mémoire moyen-terme (rapports mensuels, cycles de stocks).
- Assez court pour rester computationnellement raisonnable (la complexité de l'attention est O(seq_len²)).
- Cohérent entre tous les modèles deep et l'API — garantit qu'un même appel `/predict` est valide quel que soit le modèle.

---

## 4. Justification du choix des modèles

Le projet utilise une **stratégie de référence progressive** : chaque modèle doit battre le précédent pour justifier sa complexité. Trois niveaux sont implémentés.

### 4.1 Modèle 1 — Baseline (`src/models/baseline.py`)

#### Modèle naïf de persistance

> Prédiction : `y_pred[t+1] = WTI_price[t]`

**Pourquoi ?**
Les marchés financiers étant proches d'une **marche aléatoire** (hypothèse de marché efficient), prédire "demain = aujourd'hui" est un benchmark difficile à battre. Tout modèle qui ne fait **pas significativement mieux que le naïf** n'apporte aucune valeur métier.

#### Random Forest (500 arbres)

**Pourquoi ce choix ?**
- **Non-linéarité** : capture les interactions entre features (GPRD × prix, volatilité × incertitude) sans engineering manuel poussé.
- **Robustesse** : peu sensible aux outliers (les prix pétroliers en ont beaucoup — chocs COVID, guerre Ukraine).
- **Pas de scaling requis** : insensible aux échelles, donc moins de risque d'erreur.
- **Importance des variables** facilement interprétable.
- **Très bon baseline tabulaire** : sert de borne haute pour les approches non-temporelles. Si un modèle deep ne bat pas une RF qui ignore l'ordre temporel, sa complexité supplémentaire ne se justifie pas.

**Limite** : la RF traite chaque jour indépendamment — elle ne modélise pas explicitement la dynamique séquentielle (qu'on cherche à capturer par les modèles suivants).

### 4.2 Modèle 2 — Deep Learning (`src/models/deep_model.py`)

#### BiLSTM + Attention

```
Entrée (seq_len, n_features)
  → BiLSTM (2 couches, hidden=128, dropout=0.3)
  → AttentionPool (somme pondérée des pas de temps)
  → FC(256) → ReLU → Dropout(0.4)
  → FC(1)
```

**Pourquoi ?**
- **Bidirectionnel** : le LSTM lit la séquence dans les deux sens — utile en entraînement pour mieux contextualiser les features (par construction la cible reste future, donc pas de fuite).
- **Mémoire longue** : les portes du LSTM permettent de conserver une information sur plusieurs dizaines de pas (utile pour les chocs persistants type "guerre du Golfe").
- **Pooling par attention** : plutôt que de prendre uniquement le dernier état caché, on calcule une **somme pondérée** sur toute la séquence — laisse le modèle choisir les jours les plus pertinents (un choc géopolitique d'il y a 3 semaines peut être plus informatif que la veille).

#### CNN-GRU hybride

```
Entrée (seq_len, n_features)
  → Conv1D(64, kernel=3) → BN → ReLU
  → Conv1D(128, kernel=3) → BN → ReLU → MaxPool
  → GRU (2 couches, hidden=128)
  → FC(128) → ReLU → Dropout(0.3) → FC(1)
```

**Pourquoi ?**
- **Convolutions 1D** : extraient des **motifs locaux** (ex : pattern de "spike-de-volatilité-sur-3-jours") à très bas coût.
- **GRU** : plus léger que le LSTM (moins de paramètres) et souvent plus rapide à entraîner, avec des performances comparables sur des séquences courtes.
- **Approche complémentaire** au BiLSTM : si CNN-GRU surperforme, cela suggère que les motifs locaux dominent ; si BiLSTM surperforme, c'est que la mémoire longue est essentielle.

### 4.3 Modèle 3 — Avancé : Temporal Fusion Transformer (`src/models/advanced_model.py`)

```
Entrée (seq_len, n_features)
  → Variable Selection Network (softmax sur les features, par pas de temps)
  → Linear projection (d_model → lstm_hidden=256)
  → LSTM encoder (2 couches)
  → Multi-Head Self-Attention (8 têtes)
  → LayerNorm + résiduel
  → Gated Residual Network
  → Tête quantile (q10, q50, q90)
```

**Pourquoi le TFT ?**
- **Variable Selection Network (VSN)** : chaque pas de temps voit son propre poids softmax sur les features → fournit une **importance variable dans le temps**. Un modèle peut apprendre que "GPRD est très important en 2003" mais "EPU est plus important en 2020".
- **Self-attention multi-tête** : capte les **dépendances à longue portée** mieux que le LSTM seul (un événement d'il y a 30 jours peut directement influencer la prédiction sans devoir transiter par les 29 états intermédiaires).
- **Gated Residual Network (GRN)** : permet au réseau de **contourner les transformations inutiles** via une porte sigmoid — utile quand certaines composantes du signal sont déjà bien représentées.
- **Sortie quantile (q10/q50/q90)** : produit une **prédiction probabiliste** au lieu d'un point. En finance, savoir que "le prix sera entre 72$ et 81$ avec 80% de confiance" est souvent **plus actionnable** qu'une prédiction ponctuelle à 76.50$.
- **Loss pinball (quantile loss)** : pénalise asymétriquement les sous-estimations vs. surestimations selon le quantile → produit des intervalles bien calibrés.

**Trade-off** : le TFT est plus lourd à entraîner et plus coûteux à l'inférence. Il faut donc qu'il apporte un gain mesurable (sur MAE ou sur la couverture de l'intervalle de confiance) pour se justifier en production.

### 4.4 Tableau récapitulatif

| Modèle | Type | Paramètres ~ | Force principale | Faiblesse |
|---|---|---|---|---|
| Naïve | Heuristique | 0 | Imbattable si marché efficient | Pas d'information ajoutée |
| Random Forest | Ensemble d'arbres | ~500 × profondeur | Non-linéarités, robustesse, interprétable | Ignore l'ordre temporel |
| BiLSTM + Attn | Séquentiel récurrent | ~600 K | Mémoire longue + pondération temporelle | Lent sur longues séquences |
| CNN-GRU | Hybride local/séquentiel | ~250 K | Rapide, motifs locaux | Mémoire plus courte |
| TFT | Transformer + quantiles | ~1 M | Importance variable + incertitude | Coût compute, plus de tuning |

---

## 5. Processus de comparaison des modèles

Implémenté dans `src/evaluation/compare_models.py`.

### 5.1 Métriques calculées

Sur l'ensemble **test** (2023-01-01 → 2025-11-30), pour chaque modèle :

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Erreur moyenne en USD/baril — directement interprétable métier. |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Pénalise davantage les grosses erreurs — sensible aux chocs. |
| **MAPE** | `mean(|y - ŷ| / |y|) × 100` | Erreur relative en % — comparable entre régimes de prix. |
| **R²** | `1 - SS_res / SS_tot` | Part de variance expliquée. Un R² < 0 signifie "pire que la moyenne". |
| **Inference latency** | ms / échantillon | Mesurée à `batch_size=1` — critique pour le déploiement temps réel. |

Métriques additionnelles pour le **TFT** :
- **Pinball q10 / q90** : loss quantile sur les bornes basse et haute.
- **Coverage 80%** : fraction des observations réelles tombant dans `[q10, q90]`. Une bonne calibration donne ~80%.

### 5.2 Étapes du processus

1. **Chargement des résultats** : chaque script d'entraînement écrit un JSON dans `outputs/metrics/` (`baseline_metrics.json`, `deep_bilstm_metrics.json`, `deep_cnn_gru_metrics.json`, `advanced_tft_metrics.json`).
2. **Agrégation** : `compare_models.py` lit tous les JSON, les concatène en `pandas.DataFrame`, et trie par MAE croissante.
3. **Export tableau** : sauvegarde dans `outputs/metrics/model_comparison.csv` (consommé par le dashboard Streamlit).
4. **Re-prédictions sur test** : recharge chaque modèle et regénère les prédictions sur le **même** ensemble test → garantit une comparaison "pomme à pomme" alignée temporellement.
5. **Visualisations** :
   - **`model_comparison.png`** : barplot groupé MAE/RMSE par modèle — vue synthétique.
   - **`all_predictions_overlay.png`** : courbes superposées des prédictions de chaque modèle vs. réalité — vue qualitative.
   - **Pour le TFT** : bande d'incertitude (q10–q90) autour de q50.
6. **Rapport Markdown** : `outputs/model_analysis.md` rédige automatiquement :
   - Classement des modèles par MAE.
   - Gain relatif de chaque modèle vs. baseline naïve (en %).
   - Discussion qualitative : quel modèle gère le mieux les chocs (regarder RMSE), lequel est le plus stable (regarder R²), lequel est utilisable en production (regarder latency).

### 5.3 Critères de sélection du modèle final

Le modèle "champion" doit satisfaire **simultanément** :
1. **Performance** : MAE inférieure au baseline naïf d'au moins 5% (sinon la complexité n'est pas justifiée).
2. **Stabilité** : R² > 0.95 sur test (sans quoi le modèle "decroche" pendant les régimes de stress).
3. **Latence** : < 50 ms par prédiction sur CPU (cible production).
4. **Pour le TFT spécifiquement** : couverture 80% empirique entre 75% et 85% (bonne calibration probabiliste).

Si plusieurs modèles satisfont (1)-(3), on privilégie celui qui offre l'incertitude (TFT) pour son utilité opérationnelle, à moins que l'écart de MAE soit significatif (> 10%) en faveur d'un modèle plus simple.

### 5.4 Exposition des résultats

- **Onglet Streamlit "Model Comparison"** : affiche le CSV, les graphiques, et permet de basculer le modèle utilisé pour la prédiction live via le sidebar.
- **API `/models`** : retourne `metrics_summary` (le CSV sérialisé) pour que tout client puisse afficher les performances comparatives sans relancer l'évaluation.

---

## 6. Reproductibilité

Toutes les opérations stochastiques sont contrôlées :

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # si GPU
```

De plus :
- Les `DataLoader` PyTorch utilisent `shuffle=False` pour les séries temporelles (cohérence séquentielle).
- Le `StandardScaler` est sérialisé après fit → inférence reproductible.
- Toutes les graines, hyperparamètres et chemins sont injectés par argparse, jamais en dur dans le code.

---

## 7. Pour aller plus loin

Pistes d'amélioration non implémentées dans la version actuelle mais cohérentes avec le pipeline :

- **Walk-forward validation** : remplacer le split fixe par une validation glissante (multiple folds chronologiques) → métriques plus robustes.
- **Features externes additionnelles** : taux de change USD, prix du gaz naturel (Henry Hub), production OPEP+, inventaires hebdomadaires EIA.
- **Modèles d'ensemble** : moyenne pondérée des prédictions des 3 deep models — souvent réduit la variance.
- **Horizons multiples** : prédire t+1, t+5, t+21 simultanément avec une tête multi-output.
- **Détection de régime** : entraîner des modèles spécialisés par régime de volatilité (calme vs. crise) et router via un classifieur.
