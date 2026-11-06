import pandas as pd

df = pd.read_csv("transactions.csv", sep=None, engine="python")
df.columns = [c.strip() for c in df.columns]

df["STATUS"] = df["STATUS"].astype(str).str.strip()
df["CONTRACTOR"] = df["CONTRACTOR"].astype(str).str.strip()

def parse_sum(s):
    s = str(s).strip().replace("\xa0", "").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

df["SUM"] = df["SUM"].map(parse_sum)

ok = df[df["STATUS"].str.upper() == "OK"]

print("Топ-3 крупнейших OK-платежей:")
print(ok.nlargest(3, "SUM")[["CONTRACTOR", "SUM"]].to_string(index=False))
print()

umbrella_total = ok.loc[
    ok["CONTRACTOR"].str.lower().str.contains("umbrella", na=False),
    "SUM"
].sum()

print(f"Полная сумма OK-платежей в Umbrella, Inc.: {umbrella_total:.2f}")
