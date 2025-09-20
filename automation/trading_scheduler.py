#!/usr/bin/env python3
"""
Trading Scheduler - Exécution des phases de trading aux bons horaires de marché
"""
import subprocess
import sys
from pathlib import Path

def run_trading_phase(phase: str, day: str = None):
    """Exécute une phase de trading"""
    
    # Aller à la racine du projet
    project_dir = Path(__file__).parent.parent
    
    # Utiliser l'environnement Python du projet
    python_exe = project_dir / ".venv" / "bin" / "python"
    
    # Construire la commande
    cmd = [str(python_exe), "scripts/run_daily.py", phase]
    if day:
        cmd.extend(["--day", day])
    
    print(f"[TRADING] Exécution {phase.upper()}: {' '.join(cmd)}")
    
    # Définir PYTHONPATH et exécuter
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_dir)
    
    result = subprocess.run(cmd, cwd=project_dir, env=env)
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Usage: trading_scheduler.py {evening|preopen|summary} [--day YYYY-MM-DD]")
        print("")
        print("Phases de trading:")
        print("  evening  - Ingestion + décisions (fin de marché)")
        print("  preopen  - Vérifications + exécution (pré-ouverture)")
        print("  summary  - Rapport + email (soir)")
        return 1
    
    phase = sys.argv[1]
    day = None
    
    if len(sys.argv) > 3 and sys.argv[2] == "--day":
        day = sys.argv[3]
    
    if phase not in ["evening", "preopen", "summary"]:
        print("❌ Phase inconnue. Utiliser: evening, preopen ou summary")
        return 1
    
    return run_trading_phase(phase, day)

if __name__ == "__main__":
    sys.exit(main())