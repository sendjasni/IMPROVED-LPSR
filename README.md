# Amélioration de Plaques d'Immatriculation par Super‑Résolution ×2

Ce dépôt contient un outil **prêt à l'emploi** pour améliorer la lisibilité des plaques d'immatriculation à l'aide d'un modèle de super‑résolution **Real‑ESRGAN ×2**.  
Il a été conçu pour fonctionner **hors ligne** une fois les dépendances installées, et ne nécessite **aucune connexion Internet** après la première configuration.

## Pourquoi Real‑ESRGAN ?

**Real‑ESRGAN** (Real‑World Enhanced Super‑Resolution Generative Adversarial Network) est un modèle de super‑résolution **spécifiquement entraîné pour restaurer des images dégradées de manière réaliste** (flou de bougé, bruit, compression JPEG, faible résolution).  
Contrairement à des modèles classiques (bicubique, EDSR) ou à des modèles entraînés uniquement sur des dégradations synthétiques, Real‑ESRGAN a été massivement pré‑entraîné sur des millions d’images réelles, ce qui lui permet de :

- **Supprimer le flou** de mouvement et le flou optique,
- **Atténuer le bruit** de capteur,
- **Réduire les artefacts de compression** JPEG,
- **Restituer des détails nets** sans introduire d’hallucinations gênantes.

Nous avons sélectionné la **version ×2** car elle offre le meilleur compromis entre **gain de netteté** et **préservation de la structure** des caractères de la plaque.  
Le résultat est une image **deux fois plus grande** (largeur et hauteur doublées) avec des caractères **nettement plus lisibles**, même lorsque la plaque d’origine est sévèrement dégradée.

> ⚠️ **Avertissement :** Ce modèle est susceptible d’évoluer. Les poids fournis (`weights_x2_v0.1_v0.1.pth`) peuvent être mis à jour dans le futur pour améliorer les performances ou s’adapter à de nouvelles conditions.

## Contenu du dépôt

- `srmodel_enhance_v0.1.py` – Script Python autonome
- `weights_x2_v0.1.pth` – Poids du modèle.
- `requirements.txt` – Liste des dépendances Python

## Installation

1. **Cloner le dépôt** ou copier les trois fichiers dans un même dossier.
2. **Installer les dépendances** (une seule fois) :

```bash
pip install -r requirements.txt
```

*Les dépendances sont : `torch`, `realesrgan`, `opencv-python`, `pillow`, `tqdm`*

3. **Vérifier que le fichier de poids** `weights_x2_v0.1.pth` est bien présent dans le dossier.  
   S’il est absent, le script proposera de le télécharger automatiquement (connexion Internet requise ce jour‑là uniquement).

## Utilisation

### Améliorer une seule image

```bash
python srmodel_enhance_v0.1.py --input plaque.jpg
```

L’image améliorée sera sauvegardée sous le nom `plaque_enhanced.png` dans le même dossier.

### Améliorer toutes les images d’un dossier

```bash
python srmodel_enhance_v0.1.py --input ./dossier_plaques --output ./plaques_ameliorees
```

Toutes les images `.png`, `.jpg`, `.jpeg`, `.bmp` seront traitées et sauvegardées dans le dossier de sortie (créé automatiquement).

### Utiliser le CPU (si aucune carte graphique compatible CUDA n’est disponible)

```bash
python srmodel_enhance_v0.1.py --input image.jpg --device cpu
```


## 📋 Détails techniques

- **Architecture** : RRDBNet (Residual‑in‑Residual Dense Block Network) à 23 blocs, configurée pour un facteur d’échelle ×2.
- **Entrée acceptée** : toute image couleur (BGR) de taille quelconque.
- **Sortie** : image RGB 8 bits, deux fois plus grande (largeur ×2, hauteur ×2), sauvegardée au format PNG (par défaut).
- **Taille mémoire** : pour des plaques d’immatriculation typiques (quelques centaines de pixels de large), le modèle utilise moins de 2 Go de mémoire GPU.
- **Vitesse** : quelques centaines de millisecondes par image sur un GPU moderne.

## ❓ FAQ

**Q : Puis‑je utiliser cet outil sur des images qui ne sont pas des plaques d’immatriculation ?**  
R : Oui, le modèle fonctionne sur tout type d’image, mais il a été optimisé pour restaurer des détails fins comme du texte. Les résultats peuvent varier selon la nature de l’image.

**Q : Le script modifie‑t‑il les images originales ?**  
R : Non, les images originales ne sont jamais écrasées. Les résultats sont sauvegardés avec un suffixe `_enhanced` ou dans un dossier séparé.

**Q : Pourquoi ×2 plutôt que ×4 ?**  
R : Le facteur ×2 offre le meilleur équilibre entre amélioration de la netteté et absence d’artefacts. Un agrandissement ×4 peut introduire des hallucinations ou des déformations sur des caractères de petite taille.

**Q : Puis‑je intégrer ce script dans une application plus large ?**  
R : Absolument. La fonction `load_esrgan_x2()` et `enhance_image()` sont documentées et peuvent être importées comme n’importe quel module Python.

## Licence

Ce script est distribué sous licence MIT.  