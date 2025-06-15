# LLM Builder - Infra manuelle avec Axolotl

## 🔧 Instructions

### 1. Build de l'image Docker
```bash
docker compose build
```

### 2. Lancer l'entraînement
```bash
docker compose run trainer \
    axolotl config/mistral-lora.yaml
```

### 📁 Répertoires utiles :
- `datasets/` : contient les jeux de données au format jsonl
- `config/` : fichiers yaml de configuration Axolotl
- `output/` : checkpoints, logs, adapters