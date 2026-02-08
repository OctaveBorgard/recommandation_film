# Movie Analysis & Recommendation System

Ce projet est une plateforme d'analyse cinématographique exploitant le Deep Learning et le NLP pour classifier des films et générer des recommandations personnalisées.

## Fonctionnalités

Le système s'articule autour de quatre piliers majeurs :

1.  **Prédiction de Genre** :
    * **Via l'Affiche** : Analyse visuelle de l'affiche pour identifier le genre (Action, Comédie, Horreur, etc.).
    * **Via le Plot** : Analyse textuelle du synopsis pour déterminer le genre.
2.  **Validation d'Affiche** : Une méthode qui permet de vérifier si une image soumise est bien une affiche de film.
3.  **Moteur de Recommandation** : Suggère des films similaires en se basant sur une description textuelle (synopsis) fournie par l'utilisateur. (On propose les deux films le plus similaire celon leur synopsis et les deux films les plus similaire en comparant avec les affiches car la similarité entre un text et une affiche est toujours plus faible qu'entre deux textes)
4.  **Trouver son film** : Prend une photo par la caméra et donne le film dont l'affiche ressemble le plus !

---

## Installation et Lancement Rapide

Le projet est entièrement containerisé pour éviter les problèmes de dépendances locales.

### Prérequis
* Docker installé sur votre machine.

### Exécution
Pour construire l'environnement et lancer l'application d'un seul coup, exécutez la commande suivante dans votre terminal :

```bash
docker compose up --build
```
Une fois le déploiement terminé, l'interface sera accessible sur votre navigateur à l'adresse http://localhost:7860/

Structure du Dépôt
* app/ : Code source de l'application et de l'interface.

* models/ : Modèles pré-entraînés pour la classification d'images et le NLP.

* training/ : Fichier utilisé pour entrainer les modèles.

* content/ : Fichier vide nécessaire pour la création des images docker.

* docker-compose.yml : Orchestration des services.


## Tutoriel d'utilisation

