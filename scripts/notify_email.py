#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse
import pandas as pd
import logging
from src.config import load_params

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# utilise ton sender existant
from src.notify_email import load_email_config, send_email


def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _get_market_context() -> dict:
    """Contexte de marché pour l'email"""
    now = datetime.now(timezone.utc)
    market_hours = 9.5 <= now.astimezone().hour <= 16  # Approximation US market
    
    return {
        "timestamp": now.astimezone().strftime('%Y-%m-%d %H:%M %Z'),
        "market_open": market_hours,
        "day_of_week": now.astimezone().strftime('%A'),
        "is_weekend": now.weekday() >= 5
    }


def _generate_css() -> str:
    """CSS moderne pour l'email"""
    return """
    <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1a1a1a;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }
    .container {
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px 32px;
        text-align: center;
    }
    .header h1 {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .header .subtitle {
        font-size: 16px;
        opacity: 0.9;
        font-weight: 400;
    }
    .content {
        padding: 32px;
    }
    .alert {
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .alert-success { background: #d4edda; border-left: 4px solid #28a745; color: #155724; }
    .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; color: #856404; }
    .alert-info { background: #d1ecf1; border-left: 4px solid #17a2b8; color: #0c5460; }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 32px;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-value.enter { color: #28a745; }
    .metric-value.exit { color: #dc3545; }
    .metric-value.hold { color: #6c757d; }
    .metric-value.orders { color: #007bff; }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section {
        margin-bottom: 32px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #495057;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 8px;
    }
    .pairs-list {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .pair-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .pair-header {
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 8px;
        color: #495057;
    }
    .pair-details {
        font-size: 14px;
        color: #6c757d;
    }
    .pair-action {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .action-buy { background: #d4edda; color: #155724; }
    .action-sell { background: #f8d7da; color: #721c24; }
    
    /* Table container with horizontal scroll */
    .table-container {
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 16px;
    }
    table {
        width: 100%;
        min-width: 700px; /* Force horizontal scroll on mobile */
        border-collapse: collapse;
        font-size: 14px;
    }
    th, td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
        white-space: nowrap;
    }
    th {
        background: #f8f9fa;
        font-weight: 600;
        color: #495057;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    tr:hover {
        background: #f8f9fa;
    }
    .verdict-enter { color: #28a745; font-weight: 600; }
    .verdict-exit { color: #dc3545; font-weight: 600; }
    .verdict-hold { color: #6c757d; font-weight: 500; }
    
    /* Mobile responsive adjustments */
    @media screen and (max-width: 768px) {
        .content { padding: 20px; }
        .header { padding: 20px; }
        
        .table-container {
            -webkit-overflow-scrolling: touch;
            margin: 16px -20px 0 -20px; /* Extend to container edges */
            border-radius: 0;
            border-left: none;
            border-right: none;
        }
        
        .table-container:before {
            content: "← Faites défiler horizontalement pour voir toutes les colonnes →";
            display: block;
            text-align: center;
            font-size: 11px;
            color: #6c757d;
            font-style: italic;
            padding: 8px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        table {
            min-width: 800px; /* Wider on mobile for better readability */
        }
        
        th, td {
            padding: 10px 12px;
            font-size: 13px;
        }
        
        /* Adjust metrics grid */
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .pairs-list {
            grid-template-columns: 1fr;
        }
    }
    
    @media screen and (max-width: 480px) {
        .metrics-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
    }
    .footer {
        background: #f8f9fa;
        padding: 20px 32px;
        border-top: 1px solid #dee2e6;
        text-align: center;
        font-size: 12px;
        color: #6c757d;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-paper { background: #d1ecf1; color: #0c5460; }
    .badge-live { background: #f8d7da; color: #721c24; }
    .no-data {
        text-align: center;
        padding: 40px;
        color: #6c757d;
        font-style: italic;
    }
    </style>
    """


def _format_number(value: float, decimals: int = 2) -> str:
    """Format numbers with proper locale"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def _generate_summary_section(dec: pd.DataFrame, orders: pd.DataFrame, context: dict) -> str:
    """Section résumé sophistiquée"""
    n_enter = int((dec["verdict"] == "ENTER").sum()) if "verdict" in dec.columns else 0
    n_exit = int((dec["verdict"] == "EXIT").sum()) if "verdict" in dec.columns else 0
    n_hold = int((dec["verdict"] == "HOLD").sum()) if "verdict" in dec.columns else 0
    n_orders = len(orders)

    # Calcul notionnel total si disponible
    total_notional = 0
    if not orders.empty and "qty_a" in orders.columns and "price_a" in orders.columns:
        total_notional = (orders["qty_a"] * orders["price_a"] + orders["qty_b"] * orders["price_b"]).sum()

    # Alert basée sur le contexte
    alert_class = "alert-info"
    alert_icon = "ℹ️"
    alert_msg = f"Analyse complétée pour {context['day_of_week']}"
    
    if context["is_weekend"]:
        alert_class = "alert-warning"
        alert_icon = "⏰"
        alert_msg = "Weekend - Pas de trading prévu"
    elif n_orders > 0:
        alert_class = "alert-success"
        alert_icon = "🎯"
        alert_msg = f"{n_orders} ordre(s) prêt(s) pour l'ouverture"

    return f"""
    <div class="alert {alert_class}">
        <span style="font-size: 20px;">{alert_icon}</span>
        <span>{alert_msg}</span>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value enter">{n_enter}</div>
            <div class="metric-label">Entrées</div>
        </div>
        <div class="metric-card">
            <div class="metric-value exit">{n_exit}</div>
            <div class="metric-label">Sorties</div>
        </div>
        <div class="metric-card">
            <div class="metric-value hold">{n_hold}</div>
            <div class="metric-label">Attente</div>
        </div>
        <div class="metric-card">
            <div class="metric-value orders">{n_orders}</div>
            <div class="metric-label">Ordres</div>
        </div>
    </div>
    
    {f'<p style="text-align: center; font-size: 16px; margin-bottom: 20px;"><strong>Exposition totale:</strong> ${_format_number(total_notional, 0)}</p>' if total_notional > 0 else ''}
    """


def _generate_pairs_section(orders: pd.DataFrame) -> str:
    """Section paires avec cartes élégantes"""
    if orders.empty:
        return '<div class="no-data">Aucune paire à trader aujourd\'hui 📊</div>'

    cards_html = []
    need_cols = {"a", "b", "side_a", "qty_a", "side_b", "qty_b", "action"}
    
    if need_cols.issubset(orders.columns):
        for _, r in orders.iterrows():
            side_a_class = "action-buy" if "BUY" in str(r["side_a"]) else "action-sell"
            side_b_class = "action-buy" if "BUY" in str(r["side_b"]) else "action-sell"
            
            cards_html.append(f"""
            <div class="pair-card">
                <div class="pair-header">{r['a']} ↔ {r['b']}</div>
                <div class="pair-details">
                    Action: <strong>{r.get('action', 'N/A')}</strong><br>
                    Prix estimés: ${_format_number(r.get('price_a', 0), 2)} / ${_format_number(r.get('price_b', 0), 2)}
                </div>
                <div class="pair-action {side_a_class}">{r['side_a']}: {int(r['qty_a'])}</div>
                <div class="pair-action {side_b_class}">{r['side_b']}: {int(r['qty_b'])}</div>
            </div>
            """)

    return f'<div class="pairs-list">{"".join(cards_html)}</div>'


def _generate_decisions_table(dec: pd.DataFrame) -> str:
    """Table des décisions avec scroll horizontal sur mobile"""
    if dec.empty:
        return '<div class="no-data">Aucune décision disponible</div>'

    # Sélectionner les colonnes importantes
    display_cols = []
    col_labels = {
        "a": "A", "b": "B", "verdict": "Verdict", "action": "Action",
        "z_last": "Z-Score", "hl": "Half Life", "beta": "Beta", "pval": "P-Value", "reason": "Reason"
    }
    
    for col in ["a", "b", "verdict", "action", "z_last", "hl", "beta", "pval", "reason"]:
        if col in dec.columns:
            display_cols.append(col)

    if not display_cols:
        return '<div class="no-data">Données de décisions incomplètes</div>'

    # Construire le HTML de la table
    headers = "".join([f"<th>{col_labels.get(col, col.replace('_', ' ').title())}</th>" for col in display_cols])
    
    rows = []
    for _, row in dec.iterrows():
        cells = []
        for col in display_cols:
            value = row[col]
            cell_class = ""
            
            if col == "verdict":
                if value == "ENTER":
                    cell_class = "verdict-enter"
                elif value == "EXIT":
                    cell_class = "verdict-exit"
                else:
                    cell_class = "verdict-hold"
                cells.append(f'<td class="{cell_class}">{value}</td>')
            elif col in ["z_last", "hl", "beta", "pval"] and pd.notna(value):
                cells.append(f'<td>{_format_number(float(value), 3)}</td>')
            elif col == "reason":
                # Truncate long reasons with tooltip
                display_value = str(value)[:25] + "..." if len(str(value)) > 25 else str(value)
                cells.append(f'<td title="{value if pd.notna(value) else ""}">{display_value if pd.notna(value) else "N/A"}</td>')
            else:
                cells.append(f'<td>{value if pd.notna(value) else "N/A"}</td>')
        
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-container">
        <table>
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Chemin du bundle de la journée")
    ap.add_argument("--subject-date", default=None, help="Date à afficher (YYYY-MM-DD). Par défaut: aujourd'hui.")
    ap.add_argument("--dry-run", action="store_true", help="N'envoie pas l'email, affiche le HTML.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Email notification for bundle: {args.bundle}")

    # Charger la config pour le contexte
    try:
        params = load_params()
        mode = params.get("trading", {}).get("mode", "paper").upper()
        source = params.get("data", {}).get("source", "yahoo").upper()
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        mode, source = "PAPER", "YAHOO"

    bundle = Path(args.bundle)
    if not bundle.exists():
        logger.error(f"Bundle not found: {bundle}")
        raise SystemExit(f"Bundle introuvable: {bundle}")

    # Charger les données
    dec = _safe_csv(bundle / "decisions.csv")
    orders = _safe_csv(bundle / "orders.csv")
    
    logger.debug(f"Loaded {len(dec)} decisions, {len(orders)} orders")

    # Contexte marché
    context = _get_market_context()
    
    # Badge mode
    mode_badge_class = "badge-live" if mode == "LIVE" else "badge-paper"
    mode_badge = f'<span class="badge {mode_badge_class}">{mode}</span>'

    # Construire l'email HTML
    html_body = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>StatArb Daily Report</title>
        {_generate_css()}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 StatArb Report</h1>
                <div class="subtitle">
                    {context['timestamp']} • {source}/{mode_badge}
                </div>
            </div>
            
            <div class="content">
                {_generate_summary_section(dec, orders, context)}
                
                <div class="section">
                    <h2 class="section-title">🎯 Paires à Trader</h2>
                    {_generate_pairs_section(orders)}
                </div>
                
                <div class="section">
                    <h2 class="section-title">📊 Détail des Décisions</h2>
                    {_generate_decisions_table(dec)}
                </div>
            </div>
            
            <div class="footer">
                <p>StatArb System • Généré automatiquement • 
                Bundle: <code>{bundle.name}</code></p>
                <p style="margin-top: 8px; opacity: 0.7;">
                    {"🔴 Marché fermé" if context["is_weekend"] or not context["market_open"] else "🟢 Heures de marché"}
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    # Sujet enrichi
    date_for_subject = args.subject_date or datetime.now().strftime("%Y-%m-%d")
    n_orders = len(orders)
    subject_emoji = "🎯" if n_orders > 0 else "📊"
    subject = f"{subject_emoji} StatArb {date_for_subject} • {n_orders} ordres • {source}/{mode}"

    if args.dry_run:
        logger.info("DRY-RUN mode - displaying HTML instead of sending email")
        print(html_body)
        return 0

    try:
        cfg = load_email_config("config/email.yaml")
        success = send_email(subject, html_body, cfg)
        
        if success:
            logger.info(f"Email sent successfully to {len(cfg.get('recipients', []))} recipients")
            return 0
        else:
            logger.warning("Email sending failed or disabled in configuration")
            return 1
            
    except Exception as e:
        logger.error(f"Email notification failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())