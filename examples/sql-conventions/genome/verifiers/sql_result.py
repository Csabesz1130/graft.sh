#!/usr/bin/env python3
"""A command verifier: the reply's SQL must return what the gold SQL returns on the shop fixture.

graft passes {"reply", "messages", "reference", "case"} as JSON on stdin and runs this in the folder that
holds graft.toml. Exit 0 means the reply passes; anything else fails it, and the last line printed is the reason.

    python3 genome/verifiers/sql_result.py [--names amount_eur[,other]] [--gold SQL] [--seed PATH]

The gold query is the case's reference unless --gold gives one. Both queries run on a fresh in-memory SQLite
database built from the seed file. The results must have the same number of columns and the same rows:
in order when the gold query has a top-level ORDER BY, as a multiset otherwise. Numbers may differ by 0.005.
--names lists column names that carry a convention: where the gold result has such a column, the reply must
have a column of that name in the same position. Other column names are free.

The reply is untrusted. It must be one SELECT (or WITH ... SELECT); an authorizer allows nothing but reads
(and no read of SQLite's own sqlite_* tables), a progress handler stops runaway queries, and no file database is
ever opened.
"""
import argparse
import json
import re
import sqlite3
import sys

TOLERANCE = 0.005
MAX_OPS = 20_000_000          # SQLite virtual-machine steps before a query is stopped
MAX_ROWS = 5_000
WRITE_WORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "DETACH",
               "PRAGMA", "VACUUM", "REINDEX", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "ANALYZE"}
DENIED_FUNCTIONS = {"load_extension", "random", "randomblob", "readfile", "writefile", "edit", "fts3_tokenizer"}
READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
NOT_WRITES = {sqlite3.SQLITE_PRAGMA: "PRAGMA", sqlite3.SQLITE_ATTACH: "ATTACH", sqlite3.SQLITE_DETACH: "DETACH",
              sqlite3.SQLITE_TRANSACTION: "transaction control", sqlite3.SQLITE_SAVEPOINT: "transaction control",
              sqlite3.SQLITE_ANALYZE: "ANALYZE"}


class Fail(Exception):
    """The reply does not pass; the message is the reason."""


def mask(sql: str) -> str:
    """The SQL with comments blanked and the inside of every quoted string or identifier blanked, same length.
    What is left is structure: keywords, parentheses and semicolons that really are SQL syntax."""
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " * (j - i))
            i = j
        elif ch in "'\"`[":
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close != "]" and j + 1 < n and sql[j + 1] == close:   # a doubled quote is an escaped quote
                        j += 2
                        continue
                    break
                j += 1
            out.append(ch + " " * (min(j, n) - i - 1) + (close if j < n else ""))
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def extract_sql(text: str) -> str:
    """The one statement in a reply: out of its first code fence if it has one, without a trailing semicolon."""
    fence = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)(?:```|\Z)", text, re.DOTALL)
    sql = (fence.group(1) if fence else text).strip()
    if not sql:
        raise Fail("empty reply: no SQL")
    masked = mask(sql)
    first = masked.find(";")
    if first >= 0:
        if masked[first + 1:].replace(";", "").strip():
            raise Fail("not a single SELECT: more than one statement")
        sql, masked = sql[:first].rstrip(), masked[:first].rstrip()
    word = re.match(r"\s*\(*\s*([A-Za-z_]+)", masked)
    keyword = word.group(1).upper() if word else ""
    if keyword in WRITE_WORDS:
        raise Fail(f"write statement refused: {keyword} is not a read-only query")
    if keyword not in ("SELECT", "WITH", "VALUES"):
        raise Fail("not a single SELECT: the reply must be one SQLite query and nothing else")
    if (re.search(r"'now'", sql, re.IGNORECASE)
            or re.search(r"\bcurrent_(date|time|timestamp)\b", masked, re.IGNORECASE)
            or re.search(r"\b(date|time|datetime|julianday|unixepoch)\s*\(\s*\)", masked, re.IGNORECASE)
            or re.search(r"\bstrftime\s*\(\s*'\s*'\s*\)", masked, re.IGNORECASE)):
        raise Fail("query reads the wall clock; today is the fixed date in the system message")
    return sql


def ordered(sql: str) -> bool:
    """Whether the statement has an ORDER BY of its own, outside every pair of parentheses."""
    depth, top = 0, []
    for ch in mask(sql):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        top.append(ch if depth == 0 and ch != ")" else " ")
    return re.search(r"\bORDER\s+BY\b", "".join(top), re.IGNORECASE) is not None


def database(seed: str) -> sqlite3.Connection:
    """A fresh in-memory copy of the fixture that can only be read."""
    conn = sqlite3.connect(":memory:")
    with open(seed, encoding="utf-8") as handle:
        conn.executescript(handle.read())
    conn.commit()
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
    conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    return conn


def run(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    denied: list[tuple[int, str]] = []
    steps = [0]

    def authorize(action, arg1, arg2, _db, _source):
        if action == sqlite3.SQLITE_READ and (arg1 or "").lower().startswith("sqlite_"):
            denied.append((action, f"reads {arg1}"))
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in DENIED_FUNCTIONS:
            denied.append((action, f"calls {arg2}()"))
            return sqlite3.SQLITE_DENY
        if action in READ_ACTIONS:
            return sqlite3.SQLITE_OK
        denied.append((action, NOT_WRITES.get(action, "")))
        return sqlite3.SQLITE_DENY

    def progress():
        steps[0] += 1000
        return 1 if steps[0] > MAX_OPS else 0

    conn.set_authorizer(authorize)
    conn.set_progress_handler(progress, 1000)
    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            raise Fail("not a single SELECT: the statement returns no result")
        rows = cursor.fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            raise Fail(f"query returns more than {MAX_ROWS} rows")
        return [column[0] for column in cursor.description], rows
    except sqlite3.Warning as err:              # "You can only execute one statement at a time."
        raise Fail("not a single SELECT: more than one statement") from err
    except sqlite3.Error as err:
        if denied:
            action, what = denied[0]
            if action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION) or what:
                raise Fail(f"statement refused: {what} is not allowed") from err
            raise Fail("write statement refused: only reads are allowed") from err
        if steps[0] > MAX_OPS or "interrupt" in str(err).lower():
            raise Fail("query stopped: it ran too long") from err
        raise Fail(f"SQL error: {' '.join(str(err).split())}") from err
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers):
        return abs(a - b) <= TOLERANCE + 1e-9
    return type(a) is type(b) and a == b


def same_row(a: tuple, b: tuple) -> bool:
    return all(same(x, y) for x, y in zip(a, b, strict=True))


def show(row: tuple) -> str:
    text = "(" + ", ".join("NULL" if v is None else repr(v) for v in row) + ")"
    return text if len(text) <= 120 else text[:117] + "..."


def compare(got_names, got, gold_names, gold, in_order: bool, pinned: list[str]) -> None:
    if len(got_names) != len(gold_names):
        raise Fail(f"columns differ: got {len(got_names)}, expected {len(gold_names)}")
    for position, name in enumerate(gold_names):
        if name.lower() in pinned and got_names[position].lower() != name.lower():
            raise Fail(f"column {position + 1} must be named {name}, got {got_names[position]}")
    if len(got) != len(gold):
        raise Fail(f"rows differ: got {len(got)}, expected {len(gold)}")
    left = list(got)
    missing = []
    for row in gold:
        for index, candidate in enumerate(left):
            if same_row(row, candidate):
                del left[index]
                break
        else:
            missing.append(row)
    if missing:
        raise Fail(f"rows differ: {len(missing)} of {len(gold)} expected rows are missing, "
                   f"e.g. expected {show(missing[0])}, got {show(left[0])}")
    if in_order:
        for number, (mine, theirs) in enumerate(zip(got, gold, strict=True), 1):
            if not same_row(mine, theirs):
                raise Fail(f"row order differs at row {number}: got {show(mine)}, expected {show(theirs)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", help="the gold SQL; default: the case's reference")
    parser.add_argument("--names", default="",
                        help="comma-separated column names the reply must use where the gold does")
    parser.add_argument("--seed", default="genome/fixture/seed.sql")
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    gold_text = args.gold or ((payload.get("reference") or {}).get("content") or "")
    if not gold_text.strip():
        print("verifier error: the case has no reference SQL and no --gold was given")
        return 2
    try:
        gold_sql = extract_sql(gold_text)
        gold_names, gold = run(database(args.seed), gold_sql)
    except Fail as err:
        print(f"verifier error: the gold query does not run ({err})")
        return 2
    try:
        sql = extract_sql((payload.get("reply") or {}).get("content") or "")
        names, rows = run(database(args.seed), sql)
        compare(names, rows, gold_names, gold, ordered(gold_sql),
                [name.strip().lower() for name in args.names.split(",") if name.strip()])
    except Fail as err:
        print(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
