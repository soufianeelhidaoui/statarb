# 🤖 Trading Automatique - Configuration Simple

## ✅ CONFIGURATION ACTIVE

### 🕐 Cron Jobs Installés
```bash
# Trading automation - Version simple (session IBKR persistante)
20 9 * * 1-5 cd /Users/soufianeelhidaoui/Projects/statarb && python3 automation/trading_scheduler.py preopen
50 15 * * 1-5 cd /Users/soufianeelhidaoui/Projects/statarb && python3 automation/trading_scheduler.py evening
30 16 * * 1-5 cd /Users/soufianeelhidaoui/Projects/statarb && python3 automation/trading_scheduler.py evening
0 18 * * 1-5 cd /Users/soufianeelhidaoui/Projects/statarb && python3 automation/trading_scheduler.py summary
```

### ⚡ Paramètres Énergie macOS
```bash
# Configuration appliquée via pmset
sleep 0 → Le Mac ne se met jamais en veille complète
standby 0 → Pas de veille profonde
networkoversleep 1 → Le réseau reste actif
displaysleep 10 → L'écran s'éteint après 10 min (économie batterie)
```

## 🎯 Fonctionnement

### `trading_scheduler.py` - Simple et efficace
- **Reçoit** : phase de trading depuis cron
- **Exécute** : `scripts/run_daily.py` avec les bons paramètres
- **Gère** : environnement Python, PYTHONPATH, répertoire de travail

### Phases de Trading
- **`preopen`** (9h20) → Vérifications + exécution ordres
- **`evening`** (15h50 & 16h30) → Ingestion + décisions
- **`summary`** (18h00) → Rapports + email

## 🔧 Usage Manuel (Debug)
```bash
cd /Users/soufianeelhidaoui/Projects/statarb
python3 automation/trading_scheduler.py preopen
python3 automation/trading_scheduler.py evening --day 2025-09-19
python3 automation/trading_scheduler.py summary
```

## 🚀 Workflow Quotidien Simple

### **8h30 - Setup Manuel (5 minutes)**
1. **Lancer IB Gateway** (plus stable que TWS)
2. **Se connecter** avec login/password/2FA
3. **Vérifier API** : Configuration → API → Enable Socket Clients
4. **Laisser ouvert** toute la journée

### **9h20-18h00 - Automatique**
- **9h20** → Ordres preopen
- **15h50** → Ingestion + décisions evening  
- **16h30** → Ingestion + décisions evening
- **18h00** → Rapports + email summary

## ✅ Résultat Final

- **MacBook fermé + branché** = Trading automatique fonctionne
- **Une seule connexion manuelle** le matin suffit
- **Session IBKR persistante** = Pas de reconnexion
- **Système simple** = Moins de points de défaillance

## 🛡️ Sécurité

- **Mode paper** activé par défaut (`trading.mode: paper`)
- **Pas d'argent réel** tant que mode paper
- **Logs** dans les rapports générés