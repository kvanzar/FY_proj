# How It Works (Plain-English Architecture)

This explains what the system actually does and how the code is put
together, without assuming you already know the codebase. For the formal
requirements list, see `prd.md`. For step-by-step run/test commands, see
`run.md`.

---

## 1. What problem is this solving?

When malware infects a computer, it needs to "phone home" to its
attacker's server for instructions. It does this by sending a small
message every so often — every 60 seconds, say — like a heartbeat. This
is called **beaconing**.

The traffic is encrypted, so we can't read *what* is being sent. But we
can still watch *the pattern* of when it's sent: how regularly, how much
data goes each way, how the connection opens and closes. Malware
beaconing tends to be suspiciously regular and suspiciously
lopsided — a tiny "I'm still here" message out, almost nothing back —
compared to normal human browsing.

**This system looks at those patterns and learns to tell "malware
checking in" apart from "normal device checking in"** — without ever
decrypting anything.

## 2. The big picture, in one paragraph

Take a log of every network connection a device made. Group connections
between the same pair of devices into a "conversation." For each
conversation, measure things like *how evenly spaced are these
connections* and *how much data went each way*. Feed those measurements
into two different detectors: one that has been shown examples of both
malware and normal traffic and learned to tell them apart, and one that
has only ever seen normal traffic and flags anything that looks
unfamiliar. Combine their two opinions into one risk score. Turn that
score into a decision — let it through, raise an alert, or block it —
and explain *why* in a sentence a human can read.

## 3. Walking through it step by step

### Step 1 — Read the data in (`ingest.py`)

The input is a **Zeek `conn.log`** — a standard network-monitoring log
format where every row is one connection: who talked to whom, when, for
how long, how many bytes each way, and so on. This step:

- loads the file,
- checks that the columns we actually need are present (timestamps and
  the two IP addresses — some downloadable versions of the dataset strip
  these out, which would make the whole project impossible, so we check
  loudly and immediately rather than failing halfway through),
- cleans up Zeek's habit of writing `-` for "no value,"
- quietly adapts to a couple of harmless quirks some downloadable CSV
  exports have (an extra unlabelled index column left over from however
  they were saved; a single `label` column carrying the detailed
  category instead of Zeek's usual two separate columns) — these are
  cosmetic differences in how the same information was packaged, not
  differences in what the data means, so this step normalises them
  rather than making every later step aware of two possible formats,
- and labels each row: is this a *malware command-and-control*
  connection, a *normal* connection, or *something else entirely* (like a
  port scan) that we deliberately ignore because it isn't beaconing and
  including it would confuse the model about what it's supposed to be
  learning.

See `run.md` §4 for exactly which dataset this project is currently
configured against and what its real label distribution looks like.

### Step 2 — Group connections into conversations (`sessionize.py`)

A single connection tells you nothing about regularity — you need a
*sequence* of connections between the same two devices to see a pattern.
This step groups all connections from device A to device B (on the same
port) together, sorts them by time, and measures the gap between each one
and the next. Conversations with fewer than 4 connections are dropped —
you can't judge "is this regular?" from one or two data points.

### Step 3 — Turn conversations into measurements (`features/`)

For each conversation, compute a set of numbers that describe its shape.
These fall into a few families:

- **Timing** — how long is the average gap between connections, and how
  *consistent* is that gap? (A gap that's always almost exactly 60
  seconds is a red flag; a gap that varies wildly, like a person
  browsing, is normal.)
- **Volume** — how much data went out versus came back? A connection
  that sends a small "check-in" and gets almost nothing back is
  suspicious; a real web page load sends a little and downloads a lot.
- **Connection behaviour** — what protocol, what kind of connection state
  (did it complete normally, get rejected, reset?), does it use a
  well-known port or a random one?
- **Neighbourhood** — how many *different* devices talk to this same
  destination? A destination that only one infected device talks to
  looks different from popular services that everyone talks to.

Each of these families is computed by its own file, and every feature is
tagged with which family it belongs to — this matters later (see §7).

### Step 4 — Two independent opinions (`models/`)

This is the core idea of the project: **use two different kinds of
detector that make mistakes in different ways, so their combination is
stronger than either alone.**

- **The supervised branch** has been trained on labelled examples of
  both malware and normal conversations. It's good at recognising
  *patterns it's seen before*. Its weakness: it can only recognise malware
  families it was actually trained on.
- **The anomaly branch** has only ever seen *normal* traffic during
  training. It doesn't know what malware looks like — it just learns
  what "normal" looks like, and flags anything that deviates from that.
  Its strength is exactly the supervised branch's weakness: it can catch
  a brand-new malware family it has never seen, purely because that
  malware doesn't behave like normal traffic.

### Step 5 — Combine the two opinions (`models/fusion.py`)

The two branches each produce a score from 0 to 1. This step blends them
into one final **risk score**, using a weighted average (configurable —
you can decide how much to trust each branch).

### Step 6 — Turn a score into a decision (`decision.py`)

A risk score on its own isn't actionable. This step converts it into one
of three actions:

- **ALLOW** — low risk, let it through.
- **ALERT** — medium risk, flag it for a human to look at.
- **BLOCK** — high risk, treat it as confirmed malicious.

Where exactly those cutoffs sit is set from a practical constraint: "an
analyst can realistically review N alerts a day." Instead of picking an
arbitrary number like "0.8 is high risk," the system works backwards from
that daily budget to figure out what score corresponds to it. Every
decision is written to a log with both branch scores, so you can always
see *why* something was flagged.

### Step 7 — Explain the decision (`explain.py`)

Nobody trusts a black box that just says "BLOCK." This step uses a
technique called SHAP to figure out, for one specific conversation, which
three measurements contributed most to its score, and turns that into a
plain sentence like:

> `interval CV 0.03 (highly regular) · response bytes 0 · destination fan-in 1`

meaning: this was flagged because the timing was suspiciously regular,
almost nothing came back in response, and only this one device talks to
that destination.

### Step 8 — Test it under attack (`adversarial.py`)

A smart attacker knows regular timing gives them away, so they add random
delay ("jitter") to their beacon to look less predictable. This step
deliberately does that to known-malicious traffic in the test data, at
increasing levels (0% up to 50% random variation), and re-measures how
well the system still catches it. The result is a **degradation curve**
— it answers "how much randomness does an attacker need before we stop
catching them?", which is a far more honest measure of real-world
usefulness than performance on unmodified data.

### Step 9 — Check the results honestly (`evaluate.py`)

This is where the project is graded, and it deliberately grades itself
three different, increasingly harsh ways:

1. **Random split** — the easy version: shuffle everything and hold back
   15% for testing. This is what most projects report, and it tends to
   look great because the test data is very similar to the training data.
2. **Temporal split** — train on earlier captures, test on later ones.
   Closer to reality: you never get to train on the future.
3. **Held-out-family split** — train on some malware families, test on
   *completely different ones the model has never seen at all*. This is
   the hardest and most honest test: it answers "does this actually
   generalise, or did it just memorise?"

The gap between how well the model does on test #1 versus tests #2 and
#3 is itself an important, honest finding — a big gap means the model
was memorising specifics rather than learning genuine beaconing
behaviour.

It also measures how much each *family* of measurements (timing, volume,
neighbourhood) contributes on its own — so we can say precisely how well
the system would still work if encryption someday hid even more
information than it does today.

## 4. Why "one conversation = one row," not "one connection = one row"

This is the single most important design choice in the code, so it's
worth calling out directly: **the system makes one decision per
conversation (device pair), not per individual connection.** A single
connection can't be "regular" or "irregular" — only a sequence of them
can. So every measurement, every score, and every decision in the system
is about a whole conversation, built by looking at all the connections
within it. The one exception is the destination-popularity measurement
(fan-in), which naturally looks at the whole capture at once, since
"how many devices talk to this destination" isn't a property of one
conversation either.

## 5. Why some "malicious" traffic is thrown away, not used

The dataset contains other attacks too — port scans, denial-of-service
traffic, and so on. Those are real attacks, but they are **not
beaconing**, and they don't look anything like it (a port scan is
thousands of rapid connection attempts; a beacon is occasional, quiet
check-ins). If we told the model "these count as malicious too," it would
learn to detect generic bad traffic instead of the specific pattern this
project is about, and the whole thing would become a much less rigorous
"detect anything weird" system. So those rows are dropped entirely —
neither malicious nor benign for this model's purposes — and the
decision to do so is stated explicitly rather than buried.

## 6. Why the "learn categories" step happens separately from "compute features"

A few measurements aren't numbers, they're categories — e.g. "which
protocol was used" (TCP, UDP...). Before feeding those into a model they
need to become numeric columns (one column per category, filled with 0
or 1). Here's the subtlety: which categories exist has to be decided
using **only the training data** — if the model is allowed to see a
category that only shows up in the test data, that's a form of cheating
(it wouldn't happen in the real world, where you can't peek at the
future). So this conversion step happens *after* the training/test split
is made, using only what was seen during training — anything unfamiliar
in the test data just gets labelled "unknown" rather than breaking the
model or leaking information.

## 7. What this system deliberately does NOT do (yet)

To keep the project honest and achievable, several things are explicitly
out of scope for this phase, with a plan for later:

- **It doesn't watch live network traffic.** It reads log files that
  were already captured, not a live connection.
- **It doesn't look inside encrypted handshakes** (e.g., TLS
  fingerprinting). The dataset used here doesn't include that
  information, so the code defines where it *would* plug in later but
  doesn't pretend to use it now.
- **It doesn't actually block anything on a real network.** The
  BLOCK/ALLOW/ALERT decision is simulated and logged, not wired up to a
  real firewall.
- **It doesn't retrain itself automatically.** Every run starts fresh
  from the same settings file; there's no self-updating "live model."

These are documented as a deliberate, planned second phase — not
oversights.

## 8. File map (what lives where)

```
c2-beaconing-detection/
├── config/default.yaml   ← every setting/threshold lives here, not buried in code
├── beacon_detection/
│   ├── ingest.py          ← Step 1: read and validate the raw log
│   ├── sessionize.py      ← Step 2: group connections into conversations
│   ├── features/          ← Step 3: turn conversations into numbers
│   ├── models/
│   │   ├── supervised.py  ← Step 4a: the "seen it before" detector
│   │   ├── anomaly.py     ← Step 4b: the "this looks unfamiliar" detector
│   │   └── fusion.py      ← Step 5: combine the two
│   ├── decision.py        ← Step 6: turn a score into ALLOW/ALERT/BLOCK
│   ├── explain.py         ← Step 7: explain a decision in plain words
│   ├── adversarial.py     ← Step 8: test against an evasive attacker
│   └── evaluate.py        ← Step 9: grade the system, three honest ways
├── run_pipeline.py        ← runs all nine steps in order, one command
├── tests/                 ← checks the individual pieces work correctly
└── notebooks/             ← for exploring data and results by hand
```

See `run.md` for the actual commands to install, run, and test this.
