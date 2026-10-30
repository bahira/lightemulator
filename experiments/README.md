# experiments/ — prototypes autonomes (hors app)

Code daté de la phase d'exploration, **non maintenu**, gardé parce qu'il documente des
résultats cités dans les rapports. Aucun de ces fichiers n'est importé par `src/` ni par la CI.

| Fichier | Rôle | État |
|---|---|---|
| `spear_api_server.py` | serveur HTTP minimal exposant les kernels SPEAR | prototype |
| `spear_gui_chat.py` | chat Tkinter branché sur le serveur | prototype |
| `spear_full_implementation.py` | implémentation de référence (numpy) des kernels | référence |
| `player_softplus.py`, `PlayerSoftplus.spec` | banc de test du player (softplus) + spec PyInstaller | prototype |

Cible à terme : soit promotion dans `tools/` (si ça devient utile), soit suppression.
Rien ici ne doit être cité comme « livré ».
