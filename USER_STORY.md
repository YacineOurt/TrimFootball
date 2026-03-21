## Le contexte réel

Samedi soir, 22h30. On vient de perdre 1-0 contre Dunkerque. Le coach est frustré, il veut une réunion demain à 10h avec des éléments concrets. En parallèle, vendredi prochain on joue contre Martigues et j'ai leur dernier match filmé en Veo aussi.

J'ai donc **deux urgences** qui reviennent chaque semaine : débriefer notre match, et préparer l'adversaire suivant.

---

## Scénario 1 — Le débrief (samedi soir / dimanche matin)

**Ce que je fais aujourd'hui manuellement** : je regarde la vidéo 2-3 fois, je note des timestamps sur un Google Sheet, je fais des captures d'écran, je monte un PowerPoint à la main. Ça me prend 3 à 4 heures.

**Ce que j'aimerais qu'il se passe avec TrimFootball :**

J'importe le MP4. Le logiciel mouline — je m'en fiche si ça prend 20 minutes, je vais manger. Quand je reviens, je veux voir **un résumé du match découpé automatiquement**, pas juste un dashboard statique. Concrètement :

**Étape 1 — La vue d'ensemble rapide.** Un écran qui me dit en 30 secondes ce qui s'est passé structurellement : "1ère mi-temps : vous étiez en 4-4-2 bloc bas (hauteur moyenne 35m), compacité correcte (550 m²). 2ème mi-temps : bloc qui monte à 45m après le but encaissé, compacité qui se dégrade à partir de la 65e." C'est ça que le coach veut entendre demain matin — les grandes tendances, pas des courbes brutes.

**Étape 2 — Les moments clés.** Le logiciel devrait me flagger automatiquement les séquences où quelque chose de tactiquement significatif se passe : un décrochage soudain de la compacité (l'équipe s'est étirée), un changement de hauteur de bloc brutal, un pressing collectif réussi ou raté. Je veux une timeline avec des marqueurs cliquables : "55:12 — compacité explose (+400 m² en 3 secondes)" et quand je clique, ça m'amène à la vidéo à ce moment-là.

**Étape 3 — L'extraction pour la réunion.** Je sélectionne 4-5 séquences, et je veux pouvoir exporter soit des clips vidéo avec les données superposées (positions + enveloppes convexes), soit des visuels statiques propres que je colle dans mon PowerPoint ou que j'envoie directement sur le groupe WhatsApp du staff.

---

## Scénario 2 — La préparation adversaire (lundi-jeudi)

C'est là que l'accès aux vidéos Veo des autres équipes prend toute sa valeur.

**Ce que je veux :**

J'importe les 3 derniers matchs de Martigues. Le logiciel les analyse et me donne un **profil tactique agrégé** : formation dominante en possession et hors possession, hauteur du bloc moyenne, compacité, les zones où ils occupent le plus le terrain. Pas match par match — une synthèse.

Ensuite je veux pouvoir creuser : "Quand Martigues perd le ballon, combien de temps ils mettent à se replacer ? Est-ce qu'ils pressent haut ou ils reculent ?" Ce sont des questions de transition, et c'est là que la segmentation par phase de jeu devient indispensable.

Et enfin le livrable : une fiche adversaire de 2-3 pages que je peux imprimer et donner au coach, avec les schémas de position, les chiffres clés, et les zones de vulnérabilité identifiées.

---

## Scénario 3 — Le suivi sur la durée

Ça c'est le bonus mais ça change la vie : après 10 matchs, pouvoir dire "notre compacité moyenne en 1ère mi-temps s'améliore depuis 5 matchs" ou "on a un vrai problème de hauteur de bloc dans les 15 dernières minutes". Le coach adore ce genre de tendances parce que ça valide (ou invalide) son travail à l'entraînement.

---

## Ce qui compte vraiment pour moi en tant qu'utilisateur

Si je devais résumer en une phrase ce que je veux : **ne pas être data scientist, mais avoir les réponses d'un data scientist**. Je ne veux pas interpréter des courbes brutes. Je veux que le logiciel fasse le travail d'interprétation de base et me laisse affiner. Le dashboard actuel avec les courbes temps réel c'est bien pour un ingénieur, mais moi j'ai besoin de conclusions et de séquences vidéo associées.

Le lien **donnée → vidéo** est non-négociable. Une métrique sans pouvoir voir la séquence correspondante, ça ne sert à rien dans mon métier. Et l'**export propre** aussi — au final, tout ce que je produis finit dans un PowerPoint ou un clip vidéo. Si l'outil ne m'aide pas à sortir ces livrables, je retourne faire mes captures d'écran à la main.
