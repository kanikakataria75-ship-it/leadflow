# Deploying LeadFlow

Everything in the repo is ready. These are the steps only you can do, because
they need your logins.

Read **"Before you start"** first — two of those points change what you should
expect from the free tier.

---

## Before you start

**1. There is no git repo yet.** Vercel and Render both deploy from one. This is
step 0 below.

**2. Render's free tier sleeps.** The web service spins down after ~15 minutes
idle and takes 30–60s to wake. For a terminal one person opens a few times a day
that is an annoyance, not a blocker — but it is why **the daily scan does not run
on Render.** It runs in GitHub Actions and writes straight to Postgres, so the
forward record accumulates whether or not the web service is awake.

**3. SQLite could not stay.** The scan store was a local file
(`cache/aes_scans.sqlite`). Render's filesystem is ephemeral, so that file is
deleted on every deploy and every spin-down — the forward record would reset to
zero and still look healthy. The store now speaks both dialects: SQLite locally
when `DATABASE_URL` is unset, Postgres when it is set.

**4. I could not test the Postgres path.** No Postgres was available on this
machine. The SQL translation is unit-tested and the SQLite path is regression-
tested, but the first real proof is step 2.3, which you run. Do not skip it.

**5. The price cache is not in the repo.** `cache/` is 270 MB and rebuildable
from the vendors, so it is gitignored. GitHub Actions caches it between runs, so
only the first scan is slow.

---

## Step 0 — Put it in git

```bash
cd "C:/Users/kanik/Desktop/TRADING AI CLAUDE"
git init -b main
git add -A
git status --short | head -30
```

Check that list before committing. You should see source, `.env.example`,
`render.yaml`, `vercel.json`, `.github/workflows/`, and exactly two files under
`nifty_swing_bot/results/` (`leadflow_frozen_results.json` and
`leadflow_series.json` — the Backtest and Book pages 503 without them).

You should **not** see `.env`, anything under `cache/`, or `node_modules/`.

```bash
git commit -m "LeadFlow: terminal, auth, Postgres store, scheduled scan"
```

Then create an empty **private** repo on GitHub and push:

```bash
git remote add origin https://github.com/<you>/leadflow.git
git push -u origin main
```

Keep it private. The repo contains your frozen strategy configuration and your
research write-ups.

---

## Step 1 — Generate your secrets

```bash
python -c "import secrets; print('SESSION_SECRET=' + secrets.token_urlsafe(48))"
python -c "import secrets; print('APP_PASSWORD=' + secrets.token_urlsafe(18))"
```

Keep both in your password manager. You will paste them into two places
(Render, and GitHub Secrets for `DATABASE_URL` only).

---

## Step 2 — Postgres (Neon)

Neon's free tier is 0.5 GB and does not expire. Supabase works equally well; the
only difference is the URL format.

**2.1** Sign up at <https://neon.tech> → **Create project** → name it `leadflow`,
pick the region closest to you.

**2.2** On the project dashboard, copy the **connection string**. It looks like:

```
postgresql://user:password@ep-xxx-yyy.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
```

**2.3 — Do not skip this.** Prove the store works against it before anything
depends on it:

```bash
# Windows PowerShell
$env:DATABASE_URL="postgresql://...your string..."
python -m nifty_swing_bot.aes_scanner.dbcheck
```

Expected final line:

```
RESULT      : PASS - the store works against this database
```

It creates the schema, round-trips a row through every table the scanner writes,
exercises the one query whose SQL differs between dialects, and deletes what it
inserted. **If it does not say PASS, stop and send me the output** — deploying
past a failure here means losing the forward record silently.

**2.4 (optional) Carry over your existing local record.** You currently have 2
scans and 0 forward entries, so there is almost nothing to move and starting
clean is reasonable. If you would rather keep them, tell me and I will write the
copy script — it is a small job but not one to improvise.

---

## Step 3 — Backend on Render

**3.1** Sign up at <https://render.com> with your GitHub account.

**3.2** **New → Web Service** → connect the `leadflow` repo.

**3.3** Render will read `render.yaml`. Confirm it shows:

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn nifty_swing_bot.api.app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Plan | Free |

**3.4** Under **Environment**, add these four. They are marked `sync: false` in
the blueprint precisely so they are never committed:

| Key | Value |
|---|---|
| `APP_PASSWORD` | from step 1 |
| `SESSION_SECRET` | from step 1 |
| `DATABASE_URL` | from step 2.2 |
| `ANTHROPIC_API_KEY` | only if you want the LLM insight layer; optional |

**3.5** Deploy, then check:

```
https://<your-service>.onrender.com/api/health
```

You want:

```json
{ "config_locked": true, "auth_configured": true, "database": "postgres", ... }
```

- `auth_configured: false` → `APP_PASSWORD` did not get set. The gate **fails
  closed**, so this means locked out, never wide open.
- `database: "sqlite (ephemeral on most hosts)"` → `DATABASE_URL` did not get
  set. Fix before the first scan.
- A service that **won't start at all** is usually the config guard doing its
  job — check the logs for `LockedConfigError`, which means the deployed code
  does not match the freeze.

Copy your service URL.

---

## Step 4 — Frontend on Vercel

**4.1** Edit `vercel.json` and replace the placeholder with your Render URL:

```json
{ "source": "/api/:path*", "destination": "https://<your-service>.onrender.com/api/:path*" }
```

Commit and push.

> **Why the rewrite matters.** Vercel proxies `/api/*` to Render server-side, so
> the browser only ever talks to one origin. That keeps the session cookie
> first-party, which is what lets it stay `HttpOnly; SameSite=Lax` — the safest
> setting. If you instead point the frontend straight at the Render domain, the
> cookie becomes cross-site and needs `SameSite=None; Secure` plus a CORS
> allow-list. Don't; use the rewrite.

**4.2** Sign up at <https://vercel.com> with GitHub → **Add New → Project** →
import `leadflow`.

**4.3** Vercel reads `vercel.json`. Confirm:

| Field | Value |
|---|---|
| Framework preset | Other |
| Build command | `cd nifty_swing_bot/ui && npm install && npm run build` |
| Output directory | `nifty_swing_bot/ui/dist` |

No environment variables are needed here — the frontend holds no secrets.

**4.4** Deploy, open the URL. You should get the **login screen**. Sign in with
`APP_PASSWORD`. First load may take a minute while Render cold-starts.

---

## Step 5 — The daily scan

**5.1** In your GitHub repo: **Settings → Secrets and variables → Actions → New
repository secret**:

| Name | Value |
|---|---|
| `DATABASE_URL` | same string as step 2.2 |
| `ANTHROPIC_API_KEY` | optional |

**5.2** Go to the **Actions** tab, enable workflows if prompted, pick
**daily-scan**, and click **Run workflow** to test it now rather than waiting
for the schedule.

The first run takes ~10–20 minutes because it builds the price cache from
scratch. Later runs reuse the cache and take 2–4 minutes.

**5.3** The schedule is `30 12 * * 1-5` — 12:30 UTC = **18:00 IST, weekdays**,
after the NSE close. Cron is always UTC and GitHub does not observe DST, so this
drifts by an hour relative to any local clock that does. Edit the cron line if
you want it elsewhere.

**5.4** After a successful run, check the Forward Record page. The scan writes to
Postgres, so the terminal picks it up with no redeploy.

---

## What's protecting what

| Concern | Where it lives |
|---|---|
| Password gate on every route | `api/auth.py` + middleware in `api/app.py` |
| Session cookie | HMAC-signed, `HttpOnly`, `SameSite=Lax`, 7-day expiry |
| Secrets | Env vars only; `.env` gitignored; `.env.example` has placeholders |
| Rate limiting | `api/ratelimit.py` — 120 reads/min, **2 scans/hour**, 8 logins/15min |
| Frozen-config guard | `assert_live_config()` at import, so a drifted deploy never binds |
| Forward record | Postgres, outside the ephemeral filesystem |

Verified locally: unauthenticated requests to `/api/results`, `/api/config` and
`/api/series/*` all return 401; login with the wrong password returns 401; 7 bad
logins against a limit of 4 produced 3× 429; repeated scan triggers against a
limit of 1 produced 429s; and a deliberately drifted `risk_per_trade_pct` made
`create_app()` refuse to start with `LockedConfigError`.

---

## Local development after these changes

The gate applies locally too, so set a password or you will be locked out of
your own machine:

```bash
# .env in the project root (gitignored)
APP_PASSWORD=whatever-you-like
SESSION_SECRET=any-long-string
COOKIE_SECURE=0        # required for http://localhost
```

Leave `DATABASE_URL` **unset** locally and the store keeps using the SQLite file,
so your research workflow is unchanged.

---

## Honest limits of this setup

- **Cold starts.** Render free sleeps after 15 min idle. Unavoidable without
  paying; a keep-alive ping would violate their free-tier terms.
- **Rate limiting is per-process and in-memory.** It resets when the host
  restarts. Correct for one instance and one user; it would need Redis to mean
  anything across replicas.
- **One password, no rotation, no 2FA.** That is what you asked for and it fits a
  single-user tool. If this ever has a second user, it needs real accounts.
- **The Postgres path is unverified until you run step 2.3.** Said twice on
  purpose.
- **Neon's free tier suspends an idle database** after a few minutes. It wakes on
  the next connection, so the first query after a quiet spell is slow. It does
  not lose data.
