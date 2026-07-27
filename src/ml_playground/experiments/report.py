from pathlib import Path

import pandas as pd


def results_to_csv(results, path):
    rows = []
    for r in results:
        row = {"model": r.get("name", r.get("model", "?"))}
        if "params" in r and r["params"]:
            for k, v in r["params"].items():
                row[f"param_{k}"] = v
        if "metrics" in r:
            for k, v in r["metrics"].items():
                row[f"metric_{k}"] = v
        if "error" in r:
            row["error"] = r["error"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)
