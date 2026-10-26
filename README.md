# LPSR v0.2 — amélioration multi-images de plaques d'immatriculation (×2 / ×4)

Outil hors ligne qui produit, à partir d'une **séquence d'images** d'un même véhicule (idéalement 10), une **image améliorée** de la plaque pour chaque image, agrandie ×2 (par défaut) ou ×4. Le modèle combine l'information de **toutes les images** de la séquence.

> ⚠️ **Les images améliorées sont une aide visuelle à l'enquête et ne constituent pas une preuve.** Se référer systématiquement aux images originales.

## Ce qui change par rapport à la v0.1

La v0.1 (Real‑ESRGAN ×2, modèle générique) **est retirée**. Elle rend les plaques plus nettes en apparence, mais **invente des caractères** : un caractère peu lisible est remplacé par un autre, affiché nettement (par exemple un « B » rendu comme un « 6 »), et une plaque illisible peut devenir un texte net mais faux. Sur les séquences de test, elle dégrade la lisibilité par rapport à un simple agrandissement. Son script reste disponible dans `legacy/` et sous l'étiquette `v0.1`, pour mémoire uniquement.

La v0.2 a un autre objectif : être **fidèle** aux images de la caméra.

- **Elle combine les 10 images.** Le détail est ainsi récupéré à partir de ce que la caméra a réellement enregistré, plutôt qu'inventé à partir d'une seule image.
- **Elle a été entraînée sur de vraies paires basse / haute résolution** (20 000 séquences de plaques réelles filmées par des caméras de surveillance). Elle a ensuite été adaptée aux caméras françaises : chaque image doit pouvoir être reproduite à partir des autres.
- **Quand l'information manque, l'image reste floue** au lieu d'afficher un caractère inventé : l'incertitude reste visible.

## Résultats mesurés

Évaluation sur 71 séquences réelles de test, jamais utilisées pendant le développement (aucune plaque commune avec l'entraînement). La lisibilité est mesurée par un lecteur automatique de texte, utilisé seulement pour l'évaluation : il ne fait pas partie de l'outil.

| Méthode (×2) | Probabilité du bon caractère¹ | Caractères gagnés / perdus vs bicubique | Plaques lues exactement² |
|---|---|---|---|
| Bicubique ×2 (référence) | 0,074 | — | 19,7 % |
| v0.1 Real‑ESRGAN ×2 | 0,021 | 4,4 % / 31,2 % | 1,4 % |
| **v0.2 ×2** | **0,119** | **17,1 % / 5,0 %** | **21,1 %** |

¹ Moyenne géométrique, par caractère, de la probabilité que le lecteur attribue au bon caractère. ² Vote du lecteur sur les 10 images de la séquence.

- **Gain significatif.** Par rapport au bicubique, la v0.2 rend le bon caractère plus probable : intervalle de confiance à 95 % de la différence de log‑probabilité par caractère [+0,37 ; +0,59]. La v0.1 est significativement moins bonne : [−1,45 ; −1,11].
- **Environ trois fois plus de caractères gagnés que perdus.**
- **Pas davantage d'erreurs « sûres d'elles »** : aucune hausse significative des caractères faux affichés nettement.
- **Le modèle ×4 a la même qualité que le ×2.** Sur le test : probabilité du bon caractère 0,121 ; 22,5 % de plaques lues exactement ; 17,6 % de caractères gagnés contre 4,7 % perdus. L'écart avec le ×2 n'est pas significatif ([−0,01 ; +0,05]). Il produit une image plus grande sans ajouter d'information, pour l'affichage.

**Limite.** Les plaques font en moyenne environ 50 pixels de large et sont compressées (vidéo). Une grande partie de l'information est absente des images : le gain est réel mais progressif, et beaucoup de plaques restent illisibles.

## Installation

1. **Installer les dépendances** (une seule fois) :

   ```bash
   pip install -r requirements.txt
   ```

2. **Télécharger les poids** depuis la page *Releases* du dépôt (version `v0.2`) et les placer à côté de `enhance_sequence.py` :
   - `weights_lpsr_v0.2_x2.pth` (×2, 35 Mo)
   - `weights_lpsr_v0.2_x4.pth` (×4, 37 Mo)

3. **Vérifier les fichiers** avec `SHA256SUMS` (fourni avec la release) :

   ```bash
   sha256sum -c SHA256SUMS
   ```

Aucune connexion Internet n'est nécessaire ensuite. Une carte graphique est recommandée ; sur processeur (`--device cpu`), compter environ 5 secondes par image.

## Utilisation

```bash
python enhance_sequence.py --input dossier_sequence --output resultats              # x2 (par défaut)
python enhance_sequence.py --input dossier_sequence --output resultats --scale 4    # x4
python enhance_sequence.py --input dossier_sequence --output resultats --device cpu # sans carte graphique
```

Le dossier d'entrée contient :
- **soit** les images complètes et un fichier `detections.json` : `[{"frame": "000000.png", "license_plate_coordinates": [x1, y1, x2, y2]}, ...]`. Vérifier que les coordonnées correspondent bien aux images : sur certaines caméras, les images avaient été rognées en haut sans que les coordonnées soient corrigées ;
- **soit** uniquement des recadrages de la plaque **avec une marge** (plaque ≈ 2/3 centraux en largeur et 1/2 central en hauteur).

Sortie :

| Fichier | Contenu |
|---|---|
| `plaque_XX_originale.png` | recadrage original de l'image XX |
| `plaque_XX_bicubique_xS.png` | agrandissement classique (référence neutre) |
| `plaque_XX_amelioree_xS.png` | amélioration multi-images |
| `planche.png` | planche comparative avec l'avertissement (désactivable : `--no_sheet`) |

## Contenu du dépôt

- `enhance_sequence.py` – outil en ligne de commande
- `lpsr/model.py` – définition du réseau
- `requirements.txt` – dépendances Python
- `legacy/srmodel_enhance_v0.1.py` – ancienne version (retirée)

## Détails techniques

- **Réseau :**
  - encodeur par image (4 blocs à attention de canal) ;
  - alignement déformable sur l'image de référence ;
  - fusion temporelle pondérée par la similarité avec la référence ;
  - reconstruction (5 groupes × 8 blocs) puis agrandissement ×2 ou ×4.

  96 canaux, 8,8 M paramètres. Chaque image de la séquence sert tour à tour de référence.
- **Entraînement :**
  - paires réelles basse / haute résolution du jeu d'entraînement de la compétition ICPR 2026 *Low‑Resolution License Plate Recognition* (plaques brésiliennes et Mercosur), avec une perte de reconstruction tolérante aux petits décalages et une perte de lisibilité (lecteur figé) ;
  - adaptation aux caméras françaises par auto‑supervision **sans étiquette** : l'image restaurée, recalée et réduite, doit reproduire une image retirée de la séquence.
- **Pourquoi l'adaptation française est indispensable** : entraîné sur les seules données ICPR, le modèle redessine les plaques françaises « à la brésilienne » et invente des caractères.
- **Évaluation** : 474 séquences françaises réelles, réparties par plaque (aucune plaque commune entre entraînement, validation et test). Les réglages ont été choisis sur la validation ; le test n'a servi qu'à la mesure finale.

## Licence

Code : MIT. Le modèle a été entraîné en partie sur les données ICPR 2026 LRLPR : vérifier leurs conditions d'utilisation avant tout usage opérationnel.
