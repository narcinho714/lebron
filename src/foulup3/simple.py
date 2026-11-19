"""Simple raw test: win % when fouling vs defending, by seconds left, split into two eras."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import wilson, newcombe_rd
from .config import PROCESSED, REPORTS

EDGES = [0, 4, 8, 12, 16, 20, 24]
MID = [2, 6, 10, 14, 18, 22]
ERAS = [("1996-97 to 2012-13", lambda s: s <= 2012), ("2013-14 to 2024-25", lambda s: s >= 2013)]
BLUE, ORANGE, INK, GRID = "#2a6fb0", "#d9731a", "#1f2933", "#e3e7eb"

def table(df):
    rows = []
    for lab, m in zip(["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"], MID):
        pass
    df = df.assign(b=pd.cut(df.t_entry, EDGES, labels=MID, right=False))
    for m in MID + ["All"]:
        d = df if m == "All" else df[df.b == m]
        f, n = d[d.foul == 1], d[d.foul == 0]
        if len(f) == 0 or len(n) == 0: continue
        kf, kn = f.leader_win.sum(), n.leader_win.sum()
        rd, lo, hi = newcombe_rd(kf, len(f), kn, len(n))
        flo, fhi = wilson(kf, len(f)); nlo, nhi = wilson(kn, len(n))
        rows.append(dict(sec=m, n_foul=len(f), n_defend=len(n), win_foul=kf/len(f), win_defend=kn/len(n),
                         foul_lo=flo, foul_hi=fhi, def_lo=nlo, def_hi=nhi, diff=rd*100, lo=lo*100, hi=hi*100))
    return pd.DataFrame(rows)

def run():
    s = pd.read_parquet(PROCESSED / "states.parquet")
    s = s[s.season_type.isin(["RS", "PO"])] if "season_type" in s else s
    out = REPORTS / "simple"; out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    allt = []
    for j, (name, f) in enumerate(ERAS):
        t = table(s[f(s.season)]); t.insert(0, "era", name); allt.append(t)
        b = t[t.sec != "All"]; x = b.sec.astype(float).values
        a = ax[0, j]
        a.fill_between(x, b.foul_lo*100, b.foul_hi*100, color=ORANGE, alpha=.15)
        a.fill_between(x, b.def_lo*100, b.def_hi*100, color=BLUE, alpha=.15)
        a.plot(x, b.win_foul*100, color=ORANGE, lw=2, marker="o", label="Foul")
        a.plot(x, b.win_defend*100, color=BLUE, lw=2, marker="o", label="Defend")
        al = t[t.sec == "All"].iloc[0]
        a.set_title(f"{name}\nfoul n={int(al.n_foul)}, defend n={int(al.n_defend)}", color=INK, fontsize=11)
        a.set_ylabel("Leader win % (regulation + OT)", color=INK); a.set_ylim(60, 101); a.legend(loc="lower left", frameon=False)
        a = ax[1, j]
        a.bar(x, b["diff"], width=2.6, color=[ORANGE if v > 0 else BLUE for v in b["diff"]])
        a.errorbar(x, b["diff"], yerr=[b["diff"]-b.lo, b.hi-b["diff"]], fmt="none", ecolor=INK, capsize=3, lw=1)
        a.axhline(0, color=INK, lw=1)
        a.set_ylim(-30, 25)
        a.set_xlabel("Seconds remaining when situation began (4s bins, midpoint)", color=INK)
        a.set_ylabel("Foul - Defend (pp)", color=INK)
        a.set_title(f"Pooled: {al['diff']:+.1f} pp  (95% CI {al.lo:+.1f} to {al.hi:+.1f})   orange = foul better", color=INK, fontsize=10)
        a.set_xticks(MID)
    for a in ax.ravel():
        a.grid(color=GRID, lw=.8); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("Up 3, opponent has the ball: foul vs defend (raw win %, no adjustment)", color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "foul_vs_defend.png", dpi=150)
    pd.concat(allt).round(3).to_csv(out / "foul_vs_defend.csv", index=False)
    print(pd.concat(allt).round(3).to_string(index=False))

if __name__ == "__main__":
    run()
