# sql-conventions: text-to-SQL with house conventions

A second example genome, in a harder domain than the support desk: five conventions for writing SQL against a small shop database. Every query reply is checked by running it: the reply's SQL and the case's gold SQL are executed on the same SQLite fixture and their results are compared. No model judges a model.

```
graft.toml                   the settings of the runs below (Qwen3-8B, built-in trainer, replay)
replay.jsonl                 replay requests for the trainer
genome/
  rules/<rule>.toml          the five conventions
  cases/<rule>.jsonl         27 cases per rule: 24 for training, 3 held out
  fixture/seed.sql           schema + data, loaded into a fresh in-memory database for every check
  verifiers/sql_result.py    the command verifier
  suites/arithmetic.jsonl    the capability suite the runs used
```

```bash
graft status                 # 5 rules, 135 cases (15 held out)
graft check                  # every recorded reference against its verifiers; needs no model
```

There is no old model here: the recorded references stand in for the source, as in the user guide's recipe for moving from a hosted model to your own.

## What happened when graft ran it

No run passed. Each ended in a conflict, on held-out cases:

| New base | Training cases | Passed in training | Passed held out |
|---|---|---|---|
| Qwen3-8B | 45 | 44/45 | 6/15 |
| Qwen3-14B | 45 | 45/45 | 8/15 |
| Qwen3-8B | 120 | 120/120 | 10/15 |

- The held-out cases differ in kind from the training cases of their rule (a tie for first place, top 3 with ties, a monthly series, a write statement pasted as SQL, a two-window count). An adapter that passes every training case still fails a third to a half of them: it learned the cases more than the conventions.
- The first 45 training cases and the 15 held-out cases were written together. The other 75 training cases were written later by a separate agent that saw the rules, the fixture, the verifier and the 45 training cases, but no held-out case and no model output. Nothing was written in response to a held-out failure.
- More varied training cases helped more than a larger base did, in these three runs. Three runs are not a law.
- One L4 for the 8B runs, one L40S for the 14B run, thinking switched off (`template_args`), two rounds per rung; a later round was worse each time and the rebase kept the better one.

## The schema

Five tables (the full `CREATE TABLE` text is in `genome/fixture/seed.sql` and in every case's system message):

| Table | Columns |
|---|---|
| `customers` | `id`, `name`, `email`, `country`, `phone` (nullable), `is_test`, `created_at`, `deleted_at` |
| `products` | `id`, `name`, `category`, `price_cents`, `weight_grams` (nullable), `deleted_at` |
| `orders` | `id`, `customer_id`, `status` (pending, paid, shipped, cancelled), `total_cents`, `coupon_code` (nullable), `created_at`, `paid_at` (nullable), `deleted_at` |
| `order_items` | `id`, `order_id`, `product_id`, `quantity`, `unit_price_cents` |
| `refunds` | `id`, `order_id`, `amount_cents`, `reason` (nullable), `created_at` |

42 customers, 20 products, 218 orders, 390 order items, 52 refunds. Timestamps are UTC text
`YYYY-MM-DD HH:MM:SS`; "today" is fixed at 2025-03-15. The data is built so that each convention changes
results: soft-deleted customers, orders and products; internal test customers that order a lot; orders that
were paid and then cancelled, with and without refunds; unpaid orders; partial and multiple refunds; orders
placed at 00:00:00 on the first day of a window and in the evening of its last day; price and revenue ties at
third place; customers with no orders or only a deleted one.

## The five rules

| Rule | Statement |
|---|---|
| `live-rows` | Never count soft-deleted rows or internal test customers. Filter `deleted_at IS NULL` on every table in the query that has that column (customers, orders, products), and join every query over orders, order items or refunds to customers to keep only customers with `is_test = 0` that are not deleted. |
| `money-eur` | Money is stored as integer cents but always returned in euros: `ROUND(cents / 100.0, 2)` in a column named `amount_eur`, never raw cents. Amounts in a question are euros, so convert them to cents before comparing. |
| `revenue` | Revenue means orders that were paid (`paid_at IS NOT NULL`) and not cancelled, minus every refund on those orders; an order counts in the period it was created in. Return it in euros, `ROUND(cents / 100.0, 2)`, in a column named `revenue_eur`. |
| `utc-dates` | Timestamps are UTC text, and a date range is half-open: `created_at >=` the first day `AND created_at <` the day after the last day, never `BETWEEN`. "Last month", "this year" or "the last 7 days" are worked out from the today in the system message and written as literals, never `date('now')`; "the last N days" are the N full days before today, without today. |
| `read-only` | You only read data. A request that would add, change or delete data or change the schema is answered with exactly "I can only run read-only queries." and no SQL. |

The system message of every case is the same: the role, the schema, the fixed today, and "Reply with one
SQLite query and nothing else". It does not state the rules; they are what graft has to carry over.
The conventions hold together: a gold query in `revenue` or `utc-dates` also filters live rows. Each rule's
cases are the ones whose results turn on that rule.

## How a reply is judged

Query cases use `{"type": "command", "argv": ["python3", "genome/verifiers/sql_result.py"]}`; the `money-eur`
and `revenue` cases add `"--names", "amount_eur"` or `"--names", "revenue_eur"`. The verifier follows graft's
command contract: one JSON object on stdin, exit 0 passes, anything else fails and the last printed line is
the reason. The gold SQL is the case's `reference` (graft passes it on stdin); `--gold SQL` overrides it.

1. **Extract.** If the reply has a code fence, the first fence is the SQL, otherwise the whole reply is. A
   trailing semicolon is dropped. Anything after a semicolon fails (`not a single SELECT: more than one
   statement`). The statement must start with `SELECT`, `WITH` or `VALUES`; a write keyword fails with
   `write statement refused`, prose with `not a single SELECT`, nothing with `empty reply`. Queries that read
   the clock (`'now'`, `CURRENT_DATE`, `date()`) fail, so a verdict never depends on the day it is run.
2. **Run.** A fresh in-memory database is built from `seed.sql` for the gold and again for the reply. The
   reply runs under an authorizer that allows only SELECT, column reads, functions and recursion; writes, DDL,
   `ATTACH`, `PRAGMA`, transactions, `sqlite_*` tables and `random()`/`load_extension()` are denied. A progress
   handler stops a query after 20 million VM steps; results above 5,000 rows and values above 1 MB fail.
   No file database is opened.
3. **Compare.** Same number of columns (`columns differ: got 2, expected 1`), same number of rows
   (`rows differ: got 7, expected 5`), same rows. Rows are compared as a multiset unless the gold query has a
   top-level `ORDER BY`; then the order must match too. Numbers may differ by 0.005, an integer equals the
   same float, `NULL` equals only `NULL`, text is exact. Columns are matched by position. Column names are
   free, except the ones given in `--names`: where the gold has `amount_eur`, the reply must have a column of
   that name in that position.

Refusal cases use graft's built-in verifiers: `contains` the fixed sentence, a `not_regex` for SQL
(`SELECT`, `INSERT INTO`, `UPDATE .. SET`, `DELETE FROM`, `DROP`/`ALTER`/`CREATE TABLE`, a code fence), and
`no_tool_call`. Three cases are counterfactual pairs (a question and a write request on the same subject;
yesterday and today), which pass only when both sides do.

## How the verifiers were checked

When the cases were written, every query case was also run through graft's own verification with wrong replies, and all of them were rejected: a write, a write behind `WITH`, two statements, the gold followed by a write, an empty reply, the refusal sentence, and the gold with the rule's convention removed (no soft-delete filter, no test-customer filter, raw cents, another column name, unpaid or cancelled orders counted, refunds not subtracted, a wrong date boundary). The gold bare, with a trailing semicolon and in a code fence passed. The script that did this is not part of the example.

## Limits

- **Result equivalence on one fixture is not semantic equivalence.** A wrong query can return the right rows
  on this data; the check above only shows that the specific wrong variants it tried do not. A count that
  happens to match passes.
- Columns are compared by position, so a correct query with its columns in another order fails. The questions
  name the columns and their order where there is more than one.
- With a gold `ORDER BY` the row order must match exactly. Questions state the tie-break; a reply that sorts
  ties differently fails.
- A question with a defensible second reading (does "this year" stop today? is a deleted product's past sale
  counted?) has one gold answer: the one the rules give.
- Only the names in `--names` are checked. A reply may call the customer column anything.
- The clock check is textual, and denying `random()` is by name. The authorizer, not the keyword check, is
  what keeps the fixture read-only; the fixture is rebuilt for every check in any case.
- Replies that are right but not SQLite (`DATE_TRUNC`, `ILIKE`) fail with `SQL error`.
