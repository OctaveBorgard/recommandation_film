# Movie Analysis & Recommendation System

Développé par : Do Quoc-Bao, Nguyen Thi-Bao-Ngoc, Sidibe Mata-Dramane, Borgard Octave

Ce projet est une plateforme d'analyse cinématographique exploitant le Deep Learning et le NLP pour classifier des films et générer des recommandations personnalisées.

A noter que la partie 1 (développée sur le main car correspondant au code de base du projet) a été développée en TP, donc nous avons tous participeé à son développement contrairement à ce que les commits de cette partie peuvent laisser penser.

## Fonctionnalités

Le système s'articule autour de quatre piliers majeurs :

1.  **Prédiction de Genre** :
    * **Via l'Affiche** : Analyse visuelle de l'affiche pour identifier le genre (Action, Comédie, Horreur, etc.).
    * **Via le Plot** : Analyse textuelle du synopsis pour déterminer le genre.
2.  **Validation d'Affiche** : Une méthode qui permet de vérifier si une image soumise est bien une affiche de film.
3.  **Moteur de Recommandation** : Suggère des films similaires en se basant sur une description textuelle (synopsis) fournie par l'utilisateur. (On propose les deux films le plus similaire celon leur synopsis et les deux films les plus similaire en comparant avec les affiches car la similarité entre un text et une affiche est toujours plus faible qu'entre deux textes)
4.  **Trouver son film** : Prendre une photo par la caméra et donne le film dont l'affiche ressemble le plus ! (Ne marche pas sur cloud malheureusement mais fonctionne très bien en local)

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

Pour trouver son film à partir d'un selfie il faut d'abord cliquer sur le petit appareil photo avant de cliquer sur "Search from Photo".
