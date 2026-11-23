# 04 — Agent 2 : Cybersécurité et Détection de GPS Spoofing (Random Forest)

## 1. Objectif et Rôle dans l'Architecture Multi-Agents

Dans une flotte autonome de `250` drones, une attaque par usurpation de signal GPS (**GPS Spoofing**) injecte des coordonnées falsifiées afin de détourner un appareil ou de provoquer une collision en chaîne. Sans couche de vérification, le contrôleur d'évitement et l'Agent 1 prennent leurs décisions sur des positions corrompues.

L'**Agent 2** agit comme un **filtre de cybersécurité en temps réel** ($10\text{ Hz}$) : il analyse la cohérence entre la position GPS reçue et la cinématique inertielle propre du drone afin de détecter instantanément toute falsification (`is_spoofed = 1`), avant transmission des données aux modules de contrôle.

---

## 2. Fondements Théoriques et Extraction des Variables

### 2.1. Principe Physique : Cohérence GPS vs Cinématique Inertielle

Un drone réel ne peut pas se téléporter entre deux pas de temps $\Delta t = 0{,}1\text{ s}$. L'Agent 2 exploite le **résidu de vitesse** entre le déplacement GPS mesuré $\Delta \mathbf{P}_{\text{meas}}(t)$ et le déplacement attendu à partir de la vitesse inertielle $\mathbf{V}(t-1)$ :

$$\mathbf{r}(t) = \big(\mathbf{P}_{\text{meas}}(t) - \mathbf{P}_{\text{meas}}(t-1)\big) - \mathbf{V}(t-1)\,\Delta t$$

Pour garantir l'absence totale de fuite de données (_Zero Data Leakage_), **aucune coordonnée réelle (`tx`, `ty`, `tz`) ni distance réelle (`step_dist`) n'est utilisée** : seules les données observables en vol (`tx_measured`, `ty_measured`, `tz_measured`, `vx`, `vy`, `vz`, `v_norm`) alimentent le modèle.

### 2.2. Variables d'Entrée ($X$) — Détection d'Épisodes Complets

Une limite classique des détecteurs purement instantanés est qu'un spoofing à décalage constant ne génère un résidu important qu'à la première frame de l'attaque. Pour détecter **toute la durée de l'épisode** sans fausse alarme au retour à la normale, `11` descripteurs cinématiques et temporels sont calculés par drone (extraction complète sur `43 573` lignes en **`0,09 s`**) :

| Catégorie                                    | Variables                                                                                  | Rôle Physique                                                                                                                                                                   |
| :------------------------------------------- | :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Instantanées**                             | `step_dist_measured`, `velocity_residual`, `step_ratio`, `v_norm`                          | Détecte les sauts GPS brutaux et les incohérences immédiates de vitesse                                                                                                         |
| **Fenêtres glissantes**                      | `vel_res_roll_max_5`, `vel_res_roll_mean_5`, `vel_res_roll_max_15`, `vel_res_roll_mean_15` | Conserve la mémoire des anomalies cinématiques récentes sur $0{,}5\text{ s}$ et $1{,}5\text{ s}$                                                                                |
| **Dérive intégrée (_Dead-Reckoning Drift_)** | `drift_5`, `drift_10`, `drift_20`                                                          | Norme de la somme vectorielle $\left\|\sum \mathbf{r}(t)\right\|$ : reste élevée pendant tout un spoofing à offset constant et retombe à $\approx 0$ dès le retour à la normale |

### 2.3. Algorithme : Random Forest Classifieur

Implémenté dans `scripts/04_agent2_cybersecurity.py` avec `scikit-learn` (`n_estimators=200`, `max_depth=16`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`). La pondération équilibrée par sous-échantillon compense le fort déséquilibre des classes (`42 266` frames saines contre `1 307` frames spoofées, soit `3,0 %` d'attaques).

---

## 3. Protocole Expérimental (Split Strict par Drone)

Afin d'évaluer la capacité de généralisation sur des appareils inconnus et d'empêcher toute fuite temporelle entre frames voisines, le dataset a été séparé par **`drone_id`** via `GroupShuffleSplit` (`80 % / 20 %`) :

- **Entraînement (`Train`)** : `200` drones (`34 899` échantillons).
- **Test (`Test`)** : `50` drones jamais vus (`8 674` échantillons, dont `269` frames spoofées).

---

## 4. Résultats Quantitatifs (Jeu de Test)

Les performances enregistrées dans `results/agent2_spoofing_metrics.json` démontrent une détection quasi parfaite des attaques :

### 4.1. Métriques par Classe

| Classe                               |  Précision   | Rappel (_Recall_) |   F1-Score   | Support (Test) |
| :----------------------------------- | :----------: | :---------------: | :----------: | :------------: |
| **`0: Sain`**                        |   `0.9996`   |     `0.9998`      |   `0.9997`   |    `8 405`     |
| **`1: Spoofing`**                    | **`0.9925`** |   **`0.9888`**    | **`0.9907`** |     `269`      |
| **Global (`Accuracy` / `F1-Macro`)** | **`0.9994`** |   **`0.9943`**    | **`0.9952`** |  **`8 674`**   |

### 4.2. Matrice de Confusion et Indicateurs Globaux

| Indicateur                                     |       Valeur        |
| :--------------------------------------------- | :-----------------: |
| **Vrais Négatifs (`TN`)**                      |       `8 403`       |
| **Faux Positifs (`FP` — Fausses alarmes)**     | **`2`** (`0,024 %`) |
| **Faux Négatifs (`FN` — Attaques manquées)**   | **`3`** (`1,12 %`)  |
| **Vrais Positifs (`TP` — Attaques détectées)** |      **`266`**      |
| **ROC-AUC**                                    |    **`0.9999`**     |
| **Latence moyenne d'inférence / frame**        |   **`13,334 ms`**   |

---

## 5. Analyse et Intégration Multi-Agents

1. **Apport décisif de la dérive intégrée (`drift_5 / 10 / 20`)** : Par rapport à une approche purement instantanée (`F1 = 0.6054`, `194` faux positifs et `68` faux négatifs), l'ajout de la dérive vectorielle cumulée et des fenêtres glissantes fait progresser le F1-Score Spoofing à **`0.9907`** et divise les faux positifs par **`97`**.
2. **Fiabilité opérationnelle** : Avec seulement `2` fausses alarmes sur `8 405` mesures saines, l'Agent 2 évite les déclenchements intempestifs tout en interceptant **`98,88 %`** des frames falsifiées.
3. **Respect du budget temps réel (`10 Hz` / `100 ms`)** : L'ensemble de la chaîne IA — Agent 1 (`13,263 ms`), Agent 2 (`13,334 ms`) et Agent 3 (`1,996 ms`) — s'exécute en **`28,59 ms` par frame**, laissant plus de **`71 ms`** de marge pour le contrôleur de vol et la simulation Gazebo.

---

## 6. Artefacts Générés

- **Script d'entraînement et d'évaluation** : `scripts/04_agent2_cybersecurity.py`
- **Modèle entraîné sauvegardé** : `models/agent2_rf_spoofing.pkl`
- **Métriques JSON** : `results/agent2_spoofing_metrics.json`
