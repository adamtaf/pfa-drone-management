# Évaluation de la Baseline (Modèle de Référence Sans Agents IA)

- **Script associé** : `scripts/03_baseline_simulation.py`
- **Données d'entrée** : `data/processed/fleet_telemetry_processed.csv`
- **Sortie générée** : `results/baseline_metrics.json`

---

## 1. Objectif et Rôle Scientifique de la Baseline

Dans le cadre de ce Projet de Fin d'Année (PFA), la **Baseline** constitue le système de contrôle aérien de référence (« naïf »), dépourvu d'intelligence artificielle, de filtrage cybersécurité et d'algorithme d'évitement de collision.

Son rôle est double :

1. **Établir un étalon de mesure quantitatif** sur le jeu de données synchronisé afin de démontrer objectivement le gain apporté par l'ajout progressif des trois agents autonomes :
   - **Agent 2 (Cybersécurité)** : détection et filtrage du _GPS Spoofing_.
   - **Agent 3 (Prédiction LSTM)** : reconstruction non linéaire des trajectoires lors des coupures réseau (_Blackout_).
   - **Agent 1 (Priorisation des données)** : évaluation du risque spatial via la grille 3D et ordonnancement prioritaire des paquets télémétriques critiques.
2. **Valider l'architecture temps réel du moteur spatial** (Grille 3D par voxels) sur une flotte dense avant l'intégration des réseaux de neurones.

---

## 2. Structure Temporelle et Dynamique du Jeu de Données

La simulation s'appuie sur l'horloge globale du dataset `fleet_telemetry_processed.csv` échantillonnée à **$10\text{ Hz}$** ($\Delta t = 0{,}1\text{ s}$) :

| Caractéristique du Dataset           |         Valeur Exacte         | Interprétation Physique                                                                         |
| :----------------------------------- | :---------------------------: | :---------------------------------------------------------------------------------------------- |
| **Nombre total d'enregistrements**   |         **`43 573`**          | Couples uniques `(frame_id, drone_id)` évalués par la boucle.                                   |
| **Taille de la flotte (`drone_id`)** |       **`250` drones**        | Numérotés de `1` à `250`, décollant simultanément à `frame_id = 0`.                             |
| **Horloge globale (`frame_id`)**     |    **`251` pas de temps**     | Indexés de `0` à `250`, soit une durée maximale de **$25{,}0\text{ s}$**.                       |
| **Durée de vol par drone**           |   **`100` à `251` frames**    | Durées variables ($10{,}0\text{ s}$ à $25{,}1\text{ s}$ ; moyenne de $174{,}29\text{ frames}$). |
| **Injections GPS Spoofing**          | **`1 307` frames (`3,00 %`)** | Paquets télémétriques altérés (`is_spoofed = 1`).                                               |
| **Coupures réseau (Blackout)**       |  **`500` frames (`1,15 %`)**  | Pertes de signal continues (`is_blackout = 1`).                                                 |

### Gestion des durées de vol hétérogènes

Tous les drones décollent à `frame_id = 0`, mais leurs atterrissages sont échelonnés entre `frame_id = 99` (vol le plus court de $100$ frames) et `frame_id = 250` (vol le plus long de $251$ frames). À chaque pas de temps `frame_id`, le moteur filtre uniquement les drones actifs à cet instant : dès qu'un drone termine sa mission, il est retiré de la grille spatiale 3D.

---

## 3. Architecture et Théorie Implémentée dans la Baseline

Le comportement du programme `scripts/03_baseline_simulation.py` repose sur quatre règles déterministes :

### A. Navigation Aveugle (Absence d'Agent 1)

Les drones suivent strictement leurs trajectoires nominales $(tx, ty, tz)$ sans interaction mutuelle. Aucune commande de déviation n'est calculée lorsque deux appareils entrent dans la même zone aérienne.

### B. Confiance Aveugle aux Capteurs (Absence d'Agent 2)

Lorsque le signal télémétrique est actif (`is_blackout = 0`), la baseline accepte sans contrôle de vraisemblance cinématique les coordonnées mesurées $(tx_{\text{measured}}, ty_{\text{measured}}, tz_{\text{measured}})$. Toute falsification par **GPS Spoofing** (`is_spoofed = 1`) corrompt directement l'état perçu par le système de contrôle.

### C. Extrapolation Cinématique Linéaire en Blackout (Absence d'Agent 3)

Lorsqu'un drone subit une perte de communication (`is_blackout = 1`), la baseline estime sa position courante par **extrapolation linéaire du premier ordre** (mouvement rectiligne uniforme) à partir du dernier vecteur vitesse valide enregistré $\mathbf{V}_{t_0} = (V_x, V_y, V_z)$ :

$$\hat{\mathbf{P}}_t = \hat{\mathbf{P}}_{t-1} + \mathbf{V}_{t_0} \cdot \Delta t \quad \text{avec } \Delta t = 0{,}1\text{ s}$$

Ce modèle linéaire suppose une accélération nulle ($\mathbf{a} = \mathbf{0}$) et ne peut donc pas suivre les virages ou changements d'altitude effectués durant la coupure réseau.

### D. Hachage Spatial et Détection de Voisinage 3D ($\mathcal{O}(1)$)

L'espace aérien de $90\text{ m} \times 90\text{ m} \times 25\text{ m}$ ($X, Y \in [-45, 45]\text{ m}$, $Z \in [0, 25]\text{ m}$) est discrétisé en voxels de $10\text{ m} \times 10\text{ m} \times 5\text{ m}$. Chaque drone actif est indexé dans sa cellule $(c_x, c_y, c_z)$, et les distances euclidiennes sont évaluées uniquement face aux drones présents dans les $3 \times 3 \times 3 = 27$ voxels adjacents, ramenant la complexité de recherche de $\mathcal{O}(N^2)$ à $\mathcal{O}(1)$.

---

## 4. Définition Mathématique des Indicateurs Clés de Performance (KPIs)

| Métrique (KPI)                        | Définition Mathématique / Technique                                                                                                | Rôle dans l'Évaluation                                                               |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **Collisions critiques (`< 2.0 m`)**  | Nombre total de paires $(i, j)$ par pas de temps vérifiant $\|\mathbf{P}_i - \mathbf{P}_j\|_2 < 2{,}0\text{ m}$                    | Quantifie les impacts physiques directs en l'absence d'évitement.                    |
| **Conflits de proximité (`< 5.0 m`)** | Nombre total de paires $(i, j)$ par pas de temps vérifiant $2{,}0\text{ m} \le \|\mathbf{P}_i - \mathbf{P}_j\|_2 < 5{,}0\text{ m}$ | Mesure les intrusions dans la bulle de sécurité réglementaire.                       |
| **Taux de détection Spoofing (%)**    | $\frac{\text{Attaques détectées}}{\text{Total frames } (\text{is\_spoofed} = 1)} \times 100$                                       | Évalue la capacité de filtrage cybersécurité ($0\%$ par définition en baseline).     |
| **RMSE Localisation Spoofing (m)**    | $\sqrt{\frac{1}{N_{\text{spoof}}} \sum \|\mathbf{P}_{\text{réel}} - \mathbf{P}_{\text{perçu}}\|_2^2}$                              | Mesure l'erreur moyenne de positionnement induite par le piratage GPS.               |
| **RMSE Prédiction Blackout (m)**      | $\sqrt{\frac{1}{N_{\text{blackout}}} \sum \|\mathbf{P}_{\text{réel}} - \hat{\mathbf{P}}_{\text{linéaire}}\|_2^2}$                  | Évalue la dérive de l'extrapolation linéaire lors des pertes de signal.              |
| **Latence moyenne par pas (ms)**      | Temps CPU moyen pour traiter l'ensemble des drones actifs d'un `frame_id`                                                          | Vérifie le respect de la contrainte temps réel ($< 100\text{ ms}$ à $10\text{ Hz}$). |

---

## 5. Résultats Expérimentaux Obtenus

L'exécution de `scripts/03_baseline_simulation.py` sur l'intégralité des **251 pas de temps** ($43\,573$ enregistrements télémétriques) a produit les métriques de référence suivantes (`results/baseline_metrics.json`) :

| Métrique                                  | Valeur Mesurée (Baseline) |
| :---------------------------------------- | :-----------------------: |
| **Pas de temps évalués (`frame_id`)**     |           `251`           |
| **Enregistrements télémétriques traités** |         `43 573`          |
| **Collisions critiques (`< 2.0 m`)**      |        **`2 350`**        |
| **Conflits de proximité (`< 5.0 m`)**     |       **`22 044`**        |
| **Taux de détection GPS Spoofing**        |        **`0.0 %`**        |
| **RMSE Localisation sous Spoofing**       |      **`14.2744 m`**      |
| **RMSE Prédiction sous Blackout**         |      **`2.8425 m`**       |
| **Latence moyenne par pas de temps**      |      **`3.0797 ms`**      |

---

## 6. Analyse Critique et Justification de l'Architecture Multi-Agents

1. **Saturation de l'espace aérien (`2 350` collisions et `22 044` conflits)** :
   La circulation simultanée de jusqu'à 250 drones dans un volume confiné de $90\text{ m} \times 90\text{ m} \times 25\text{ m}$ sans régulation génère une densité d'intersections critique. Ce résultat démontre la nécessité de **l'Agent 1 (Évitement & Régulation 3D)** pour réallouer dynamiquement les vecteurs vitesse et altitudes.
2. **Vulnérabilité aux cyberattaques GPS (`14.2744 m` d'erreur)** :
   Sans mécanisme de détection, une attaque par _GPS Spoofing_ déplace la position perçue de plus de $14{,}27\text{ m}$ en moyenne — soit plus de **7 fois le rayon de collision** ($2{,}0\text{ m}$). Cela rend tout évitement impossible tant que la donnée n'est pas assainie par **l'Agent 2 (Anti-Spoofing cinématique)** à l'aide du seuil de rupture physique ($1{,}11\text{ m}$).
3. **Dérive de l'extrapolation linéaire en blackout (`2.8425 m` d'erreur)** :
   Lorsqu'un drone perd la connexion pendant $0{,}5\text{ s}$ à $1{,}0\text{ s}$, l'approximation linéaire rectiligne accumule une erreur quadratique moyenne de $2{,}84\text{ m}$, dépassant le seuil critique de collision ($2{,}0\text{ m}$). L'intégration de **l'Agent 3 (Réseau récurrent LSTM)** permettra de modéliser la courbure des trajectoires pour maintenir l'erreur sous le seuil de sécurité.
4. **Marge de calcul temps réel (`3.08 ms` / pas de temps)** :
   Avec un temps de traitement moyen de **$3{,}08\text{ ms}$** par pas de temps pour gérer l'indexation spatiale 3D de toute la flotte (contre une période d'échantillonnage de $100\text{ ms}$ à $10\text{ Hz}$), le système dispose de plus de **$96\text{ ms}$** de budget CPU par cycle pour exécuter les trois agents IA en temps réel.
