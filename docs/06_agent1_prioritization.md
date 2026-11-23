# 06 — Agent 1 : Priorisation Intelligente des Conflits en Temps Réel (Random Forest)

## 1. Objectif et Rôle dans l'Architecture Multi-Agents

Dans un espace aérien dense comptant jusqu'à `250` drones simultanés, traiter tous les appareils avec le même niveau de priorité sature la bande passante de commande et dégrade la réactivité du contrôleur d'évitement. La baseline (`docs/03_baseline_evaluation.md`) se contentait de compter passivement les collisions (`< 2.0 m`) et conflits (`< 5.0 m`) sans hiérarchisation dynamique du risque.

L'**Agent 1** introduit une **classification supervisée en temps réel** qui attribue à chaque drone, à chaque pas de temps ($10\text{ Hz}$), un niveau d'urgence opérationnel (`P0 Critique`, `P1 Haute`, `P2 Normale`) afin d'ordonnancer les manœuvres d'évitement dans la boucle de contrôle.

---

## 2. Fondements Théoriques et Extraction des Variables

### 2.1. Partitionnement Spatial 3D (Grille de Voxels)

Afin d'éviter une recherche quadratique $O(N^2)$ entre les `250` drones, l'arène ($90\text{ m} \times 90\text{ m} \times 25\text{ m}$) est découpée en voxels 3D de dimension $10\text{ m} \times 10\text{ m} \times 5\text{ m}$. Pour chaque drone $i$, la recherche du plus proche voisin s'effectue uniquement dans son voxel courant et les 26 voxels adjacents, permettant d'extraire les caractéristiques sur l'ensemble des `43 573` enregistrements en **`1,29 s`**.

### 2.2. Variables d'Entrée ($X$)

Quatre descripteurs cinématiques et spatiaux sont calculés pour chaque drone à chaque frame :

1. **`d_min` (m)** : Distance euclidienne au plus proche voisin détecté (plafonnée à $20\text{ m}$).
2. **`v_close` (m/s)** : Vitesse de rapprochement positive projetée sur l'axe reliant les deux drones :
   $$v_{\text{close}} = \max\left(0, -\frac{(\mathbf{P}_j - \mathbf{P}_i) \cdot (\mathbf{V}_j - \mathbf{V}_i)}{\|\mathbf{P}_j - \mathbf{P}_i\| + \epsilon}\right)$$
3. **`voxel_density`** : Nombre total de drones présents simultanément dans le même voxel 3D (mesure de congestion locale).
4. **`v_norm` (m/s)** : Norme du vecteur vitesse propre du drone $\|\mathbf{V}_i\|$.

### 2.3. Cible à Prédire ($Y$) et Modèle Physique de Risque

Le niveau d'urgence combine la proximité spatiale et le **Time-To-Collision ($TTC = d_{\min} / v_{\text{close}}$)** :

| Classe  | Niveau d'Urgence    | Critère Physique                                                                       | Effectif Total (CSV) |
| :-----: | :------------------ | :------------------------------------------------------------------------------------- | :------------------: |
| **`0`** | **Normale (`P2`)**  | Hors zone de conflit immédiat                                                          | `14 084` (`32,3 %`)  |
| **`1`** | **Haute (`P1`)**    | $d_{\min} < 5{,}0\text{ m}$ ou ($d_{\min} < 8{,}0\text{ m}$ et $TTC < 2{,}5\text{ s}$) | `23 300` (`53,5 %`)  |
| **`2`** | **Critique (`P0`)** | $d_{\min} < 2{,}0\text{ m}$ ou ($d_{\min} < 5{,}0\text{ m}$ et $TTC < 1{,}0\text{ s}$) |  `6 189` (`14,2 %`)  |

### 2.4. Algorithme : Random Forest Classifieur

Implémenté dans `scripts/06_agent1_prioritization.py` avec `scikit-learn` (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=2`, `class_weight="balanced"`). L'option `class_weight="balanced"` pénalise davantage les erreurs sur la classe minoritaire `2: Critique` afin de maximiser le rappel (_Recall_) sur les situations de collision imminente.

---

## 3. Protocole Expérimental (Découpage Temporel Strict)

Pour éviter toute fuite de données temporelle entre des frames consécutives, le dataset (`251` frames) a été séparé **chronologiquement par `frame_id`** :

- **Entraînement (`Train` — 80 %)** : Les `200` premières frames (`42 092` échantillons).
- **Test (`Test` — 20 %)** : Les `51` dernières frames (`1 481` échantillons, phase de fin de mission avec densité réduite).

---

## 4. Résultats Quantitatifs (Jeu de Test)

Les performances enregistrées dans `results/agent1_prioritization_metrics.json` confirment la fiabilité du classifieur :

### 4.1. Métriques par Classe

| Classe                               |  Précision   | Rappel (_Recall_) |   F1-Score   | Support (Test) |
| :----------------------------------- | :----------: | :---------------: | :----------: | :------------: |
| **`0: Normale`**                     |   `0.9984`   |     `0.9992`      |   `0.9988`   |    `1 227`     |
| **`1: Haute`**                       |   `0.9871`   |     `0.9828`      |   `0.9849`   |     `233`      |
| **`2: Critique`**                    |   `0.9048`   |     `0.9048`      |   `0.9048`   |      `21`      |
| **Global (`Accuracy` / `F1-Macro`)** | **`0.9953`** |   **`0.9623`**    | **`0.9628`** |  **`1 481`**   |

### 4.2. Importance des Variables et Temps Réel

| Variable d'entrée                             | Poids dans la décision (`Feature Importance`) |
| :-------------------------------------------- | :-------------------------------------------: |
| **`d_min`** (Distance au plus proche voisin)  |           **`78,08 %`** (`0.7808`)            |
| **`v_close`** (Vitesse de rapprochement)      |           **`17,62 %`** (`0.1762`)            |
| **`voxel_density`** (Densité locale du voxel) |              `3,81 %` (`0.0381`)              |
| **`v_norm`** (Norme de vitesse propre)        |              `0,49 %` (`0.0049`)              |

- **Latence moyenne d'inférence par frame complète** : **`13,263 ms`** (compatible avec un cycle de contrôle à `10 Hz` / `100 ms`).

---

## 5. Analyse et Intégration

1. **Cohérence physique du modèle** : Les deux variables critiques de la cinématique d'évitement (`d_min` et `v_close`) représentent à elles seules **`95,70 %`** du pouvoir décisionnel du Random Forest, confirmant que le modèle a correctement appris la frontière de risque liée au $TTC$.
2. **Robustesse en fin de mission** : Bien que les `51` dernières frames du jeu de test ne comportent que `21` cas critiques (dispersion de la flotte en fin de trajectoire), le classifieur conserve un rappel et une précision de **`90,48 %`** sur la classe `P0 Critique` et une précision globale de **`99,53 %`**.
3. **Budget temps réel maîtrisé** : Combiné à l'Agent 3 (`1,9964 ms`), l'Agent 1 (`13,263 ms`) consomme environ `15,3 ms` par frame, laissant plus de `84 ms` disponibles pour l'Agent 2 (Cybersécurité) et le contrôleur Gazebo.

---

## 6. Artefacts Générés

- **Script d'entraînement et d'évaluation** : `scripts/06_agent1_prioritization.py`
- **Modèle entraîné sauvegardé** : `models/agent1_rf_prioritization.pkl`
- **Métriques JSON** : `results/agent1_prioritization_metrics.json`
