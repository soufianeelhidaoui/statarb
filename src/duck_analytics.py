from pathlib import Path
import duckdb

def write_scored_to_duckdb(parquet_path: Path, mode: str = "replace") -> str:
    """
    mode:
      - 'replace' : remplace totalement la table pairs_scored par le contenu du parquet (recommandé)
      - 'append_by_name': tente d'aligner les colonnes par nom et ajoute si possible (historique quotidien)
    """
    db = Path("reports") / "analytics.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db.as_posix())

    if mode == "replace":
        con.execute("CREATE OR REPLACE TABLE pairs_scored AS SELECT * FROM read_parquet(?)", [parquet_path.as_posix()])
        con.close()
        return db.as_posix()

    con.execute("CREATE TABLE IF NOT EXISTS pairs_scored AS SELECT * FROM read_parquet(?)", [parquet_path.as_posix()])

    cols = [r[1] for r in con.execute("PRAGMA table_info('pairs_scored')").fetchall()]  # name at index 1

    parquet_cols = [r[0] for r in con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [parquet_path.as_posix()]).description]
    selects = []
    for c in cols:
        if c in parquet_cols:
            selects.append(f'"{c}"')
        else:
            selects.append(f'NULL AS "{c}"')

    sql = f"INSERT INTO pairs_scored SELECT {', '.join(selects)} FROM read_parquet(?)"
    con.execute(sql, [parquet_path.as_posix()])
    con.close()
    return db.as_posix()
