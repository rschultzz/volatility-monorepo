import os, sys
from dotenv import load_dotenv
load_dotenv()
import psycopg
url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://","postgresql://")
def q(sql, params=None):
    with psycopg.connect(url) as c:
        c.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        c.execute("SET statement_timeout = '600s'")
        cur = c.execute(sql, params)
        cols = [d.name for d in cur.description]
        return cols, cur.fetchall()
def show(sql, params=None):
    cols, rows = q(sql, params)
    print(" | ".join(cols))
    for r in rows: print(" | ".join(str(x) for x in r))
    print()
if __name__ == "__main__":
    show(sys.stdin.read())
