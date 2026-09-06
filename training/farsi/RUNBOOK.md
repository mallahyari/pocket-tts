# Farsi Pocket TTS — step-by-step runbook

**Start here.** [`README.md`](README.md) is the reference manual: it explains
*why* each piece is the way it is, organized by topic. This file is the
opposite — a linear checklist you follow top to bottom, with a **STOP** gate
between phases so that nothing expensive starts before something cheap has
proved it will work.

It is written for someone who has not run GPU jobs on GCP before. The single
most expensive mistake in this whole project is not a bad hyperparameter — it
is leaving an `a3-highgpu-8g` running overnight for no reason (**~$88/hour,
~$2,100/day**). Every phase below therefore ends with *what you are still
paying for*.

### The five phases

| phase | where | you pay | takes | what you get |
|---|---|---|---|---|
| **0** | your laptop | **$0** | 1-2 h | the pipeline installed and understood, on real Persian data |
| **1** | GCP console | **$0** | 20 min + 0-2 days waiting | budget alarms, GPU quota approved |
| **2** | 1 cheap GPU VM | **~$15** | 8-12 h (mostly unattended) | 600 h of downloaded, aligned Farsi data on a disk |
| **3** | 1 mid GPU VM | **~$25** | 12-24 h | a pilot model that proves the data is good |
| **4** | 8×H100 | **~$800-1,000** | ~24 h | the real model |

Phases 0-3 cost about **$40 in total** and are where every problem you are
actually going to have will show up. Do not skip them to get to phase 4 sooner.

---

# Phase 0 — On your laptop, $0

No cloud account needed. No GPU needed. Goal: prove the code runs and see what
Persian data actually looks like going through it.

### 0.1 Install

```bash
cd pocket-tts
uv sync
```

Takes a few minutes (it pulls torch). If you do not have `uv`:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 0.2 Run the tests

```bash
uv run pytest training/tests -q
```

**Expect:** `89 passed`. If this fails, stop and fix it before anything else —
nothing downstream can work.

### 0.3 See what Persian normalization does

```bash
uv run python -c "
from training.farsi.normalize_fa import normalize
for t in ['در سال ۱۴۰۲ حدود ۲۵٪ از مردم', 'می‌روم به خانه ي پدري كه در ايران است', '.این چیزی نیست']:
    print(t, ' -> ', normalize(t))
"
```

**Expect** digits spelled out as words, Arabic `ي`/`ك` folded to Persian
`ی`/`ک`, and the stray leading period gone. This is [§3 of the
README](README.md#3-persian-text-the-part-that-is-genuinely-different) in
action. If this output looks wrong *to you as a Persian speaker*, that is the
most valuable bug report in the project — fix it here, before it is baked into
600 hours of manifests.

### 0.4 Download a tiny slice of real data (~330 MB, no GPU)

```bash
uv run python -m training.farsi.prepare_data_fa \
    --hours 1 --sources commonvoice --cv-splits dev \
    --manifests-out /tmp/fa_pilot --audio-out /tmp/fa_audio \
    --skip-align --vocab-size 1000
```

**Expect**, after ~5-10 minutes:

```
[INFO] text/duration filtering: {'kept': 779, 'bad_duration': ..., ...}
[INFO] train.jsonl: 739 utterances (~0.9h)
[INFO] valid.jsonl: 40 utterances (~0.1h)
wrote /tmp/fa_pilot/tokenizer.model (vocab 1000)
```

Then **listen to a few clips** and read their transcripts:

```bash
head -3 /tmp/fa_pilot/train.jsonl | uv run python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line); print(d['duration'], d['transcript']); print('  ', d['path'])
"
```

Open one of those files in any audio player. Does the audio say what the
transcript says? That question is the entire quality bar of this project.

### 0.5 Check the corpus statistics tool

```bash
uv run python -m training.farsi.normalize_fa /tmp/fa_pilot/train.jsonl --stats-only
```

**Expect:** a rejection breakdown and a character histogram. Nothing should be
marked `<-- UNEXPECTED`. This is the command you will run again on the real
600-hour corpus in phase 2.

> ### ✅ STOP — gate 0
> Do not open a GCP console until: tests pass, normalization looks right to
> you, and you have listened to at least three clips and read their
> transcripts. **You are paying $0 and can stay here as long as you like.**

---

# Phase 1 — GCP account safety, $0

Twenty minutes of setup, then possibly a day or two of waiting for quota. Do
this *before* phase 2 so the waiting overlaps with nothing.

### 1.1 Pick a project and set it as default

```bash
gcloud auth login
gcloud projects list
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

### 1.2 Set a billing budget with alerts — do this first

Console → **Billing → Budgets & alerts → Create budget**. Set the amount to
what you are willing to lose (start at **$100** for phases 0-3, raise it before
phase 4), and add alert thresholds at 50%, 90% and 100% of it.

> ⚠️ **A GCP budget does not stop anything.** It only emails you. There is no
> "hard cap" switch. The thing that actually protects you is deleting VMs, and
> the daily check in §1.5.

### 1.3 Request GPU quota (this is the part with a waiting time)

First see what you already have. The obvious command prints unusable parallel
arrays without `--flatten`:

```bash
gcloud compute regions describe us-central1 --flatten="quotas[]" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -iE "h100|a100|l4|a2_cpus"
```

Then Console → **IAM & Admin → Quotas & System Limits**, filter
`Service: Compute Engine API`, and request what is missing. **The console shows
display names, not the API constants**, which is the main reason people cannot
find these rows:

| API name | what the console calls it | ask for | needed by |
|---|---|---|---|
| `GPUS_ALL_REGIONS` | GPUs (all regions) | 8 | everything (global cap; often already high) |
| `NVIDIA_L4_GPUS` | NVIDIA L4 GPUs | 1 | phase 2 |
| `NVIDIA_A100_80GB_GPUS` | NVIDIA A100 80GB GPUs | 1 | phase 3 |
| `A2_CPUS` | A2 CPUs | 12 per A100 | phase 3 — the hidden co-requisite |
| `PREEMPTIBLE_NVIDIA_H100_GPUS` | Preemptible NVIDIA H100 80GB GPUs | 8 | phase 4 |
| `NVIDIA_H100_GPUS` | NVIDIA H100 80GB GPUs | 8 | phase 4 only if you refuse Spot |

Search "preemptible" to find the Spot rows — GCP still labels Spot quota that
way. Tick the row's checkbox, then **EDIT QUOTAS**, enter the new limit and a
specific justification ("training a text-to-speech model; 8x H100 Spot for a
~16 hour run in us-central1"). Vague requests get denied more often.

Four things that surprise people here:

* **Quota is region-scoped.** An approval for `us-central1` does nothing for
  `us-east4`. Pick one region and stay in it.
* **You may be approved for Spot only.** `PREEMPTIBLE_NVIDIA_H100_GPUS: 8` with
  no on-demand H100 quota is a normal outcome, and is enough — phase 4 runs on
  Spot anyway.
* **GPU quota is not GPU capacity.** Approved quota still gets
  `ZONE_RESOURCE_POOL_EXHAUSTED` when the zone is full; see §4.3.
* **`gcloud compute regions describe` only lists legacy quota metrics.** Newer
  GPU types can be missing from it entirely while existing in the console, so
  check both before concluding you have no quota.

**If H100 is denied**, you are not blocked: phases 2 and 3 need one small GPU,
and phase 4 runs on `a3-highgpu-4g` or `-2g` with a different `batch_size`
(§4.3). If **EDIT QUOTAS is greyed out**, the project usually lacks billing
history — a few hours on the cheap phase-2 VM often unlocks it.

### 1.4 Get a HuggingFace token

```bash
uv run hf auth login
```

Needed on every VM. **Only the fine-tune path needs the gated
`kyutai/pocket-tts` weights** (accept the license on its model page first); the
from-scratch path uses public weights and this token is only for download
rate limits.

### 1.5 Learn the two commands that save you money

```bash
gcloud compute instances list     # anything RUNNING is costing you money right now
gcloud compute disks list         # disks cost money even when every VM is deleted
```

Run these **every single day** you are working on this, and again the day after
you think you have finished. Put a reminder in your calendar now.

> ### ✅ STOP — gate 1
> Budget created, quota approved (or a fallback machine chosen), and you have
> run `gcloud compute instances list` once. **Still $0.**

---

# Phase 2 — Data prep on a cheap VM, ~$15

**Why not do this on the H100 machine?** Because downloading ~60 GB and
decoding Mana-TTS is hours of work that uses almost no GPU. Doing it on the
8×H100 box would cost **$300-500 of idle H100 time**. On an L4 it costs about
$10. This single decision is the biggest saving in the runbook.

### 2.1 Create a data disk that outlives every VM

```bash
gcloud compute disks create fa-data --size=1000GB --type=pd-balanced --zone=us-central1-a
```

**This disk costs ~$100/month (~$3.30/day) whether or not a VM is attached.**
It is the one thing you will keep between phases. Delete it when the project is
done (§5.3). 1000 GB is sized for a 600-hour corpus with headroom; 500 GB is
enough if you stick to `manatts,filimo`.

### 2.2 Create the cheap VM

```bash
# Find a current Deep Learning VM image (families get re-cut regularly), then
# assign it -- pasting a <placeholder> makes the shell read "<" as a redirect
# and fail with "no such file or directory".
gcloud compute images list --project deeplearning-platform-release \
    --filter="family~'common-cu12'" --format="value(family)" | sort -u

IMAGE_FAMILY=common-cu129-ubuntu-2204-nvidia-580     # or whatever that printed
gcloud compute instances create fa-prep \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --image-family="$IMAGE_FAMILY" \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --boot-disk-size=100GB --boot-disk-type=pd-balanced \
    --disk=name=fa-data,device-name=fa-data,mode=rw,boot=no,auto-delete=no \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

`g2-standard-8` is 1×L4: **~$0.90/h on-demand, ~$0.30/h on Spot**. Spot is fine
here — `prepare_data_fa.py` is resumable, so a preemption costs you a restart,
not the work.

### 2.3 Set the VM up

```bash
gcloud compute ssh fa-prep --zone=us-central1-a
```

On the VM (the first `nvidia-smi` may take a minute while the driver installs):

```bash
nvidia-smi                                              # 1 GPU visible

sudo mkfs.ext4 -F /dev/disk/by-id/google-fa-data        # FIRST TIME ONLY -- erases the disk
sudo mkdir -p /mnt/data && sudo mount /dev/disk/by-id/google-fa-data /mnt/data
sudo chown "$USER" /mnt/data

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/kyutai-labs/pocket-tts && cd pocket-tts
uv sync
uv run hf auth login
```

> ⚠️ **`training/farsi/` is not in the upstream repo.** A plain clone of
> `kyutai-labs/pocket-tts` gives you the model code but none of this pipeline,
> and `pytest` will quietly report only upstream's tests. Either push this work
> to your own fork and clone that (`git clone -b farsi https://github.com/<you>/pocket-tts.git`),
> or copy it up from your laptop:
>
> ```bash
> # from the repo root ON YOUR LAPTOP
> gcloud compute scp --recurse --zone=us-central1-a training/farsi fa-prep:~/pocket-tts/training/
> gcloud compute scp --zone=us-central1-a training/tests/test_farsi.py fa-prep:~/pocket-tts/training/tests/
> ```

Then, on the VM:

```bash
uv add --dev pyarrow            # Mana-TTS ships as parquet
uv run pytest training/tests -q
```

**Expect the 27 `test_farsi.py` tests to be among those that pass** — that is
the real check, not the total (upstream's own count drifts between commits).
`uv run pytest training/tests/test_farsi.py -q` says it directly: `27 passed`.

> ⚠️ Run `mkfs` **only the first time**. Running it again on a later VM wipes
> the data you just paid to download.

Make the mount survive a reboot — a Spot VM can be stopped at any moment, and
an unmounted `/mnt/data` is silently just an empty directory on the boot disk,
so the next run re-downloads 60 GB onto a 100 GB disk and fills it:

```bash
echo '/dev/disk/by-id/google-fa-data /mnt/data ext4 discard,defaults,nofail 0 2' \
  | sudo tee -a /etc/fstab
```

`nofail` keeps the VM bootable if the disk is ever detached. Do this on every
VM that attaches `fa-data`, phases 2 through 4.

**If you are preempted mid-job:**

```bash
gcloud compute instances start fa-prep --zone=us-central1-a
gcloud compute ssh fa-prep --zone=us-central1-a
df -h /mnt/data          # mounted, because of the fstab line above
# rerun the same prepare_data_fa command -- it skips what it already has
```

### 2.4 Run the data preparation — in tmux

```bash
tmux new -s prep      # so an ssh drop does not kill an 8-hour job

uv run python -m training.farsi.prepare_data_fa \
    --hours 600 \
    --sources manatts,filimo,youtube \
    --manifests-out /mnt/data/farsi_600h \
    --audio-out /mnt/data/farsi_audio \
    --align-shards 1 \
    2>&1 | tee /mnt/data/prep.log
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t prep`. Use
`tee -a` instead of `tee` when resuming, to keep the earlier log.

**Expect ~5-8 hours**, in this order: Mana-TTS parquet download and decode
(~15 min; the `HIGH` match-quality filter keeps roughly 60 h of its 114 h),
Filimo and YouTube tar downloads (2-3 h), tokenizer training (minutes), then
forced alignment on the single L4 (3-5 h for 600 h of audio).

Watch it without reattaching:

```bash
grep "INFO prepare_data_fa" /mnt/data/prep.log | tail -20   # milestones only
cat /mnt/data/farsi_600h/done_*.txt | wc -l                 # archives completed
du -sh /mnt/data/farsi_audio/*                              # which source is growing
df -h /mnt/data                                             # expect ~120 GB at the end
```

Filter on the logger name, not on `INFO` — `httpx` logs one INFO line per
download redirect and there are thousands of them.

**It is resumable at archive granularity.** Each source appends its rows to
`raw_<source>.jsonl` and its finished archives to `done_<source>.txt` in the
manifests directory, so rerunning the identical command after a crash or a
preemption skips straight to the next unprocessed tar or parquet instead of
re-downloading tens of gigabytes.

### 2.5 Check the result before you pay for a GPU to train on it

```bash
uv run python -m training.farsi.normalize_fa /mnt/data/farsi_600h/train.jsonl --stats-only | head -40

uv run python -c "
import json
from pathlib import Path
root = Path('/mnt/data/farsi_600h')
n_in = sum(1 for _ in open(root / 'train.jsonl'))
rows = [json.loads(l) for l in open(root / 'train_aligned.jsonl')]
print(f'{len(rows)} aligned of {n_in} submitted  ->  {len(rows)/n_in:.2%} coverage')
print(f'{sum(r[\"duration\"] for r in rows)/3600:.0f} hours of training audio')
d = rows[0]
print(d['transcript'])
print(d['words'][:5])
"
```

Coverage is `train_aligned.jsonl` divided by `train.jsonl`, **not** the share of
rows carrying a `words` field: `align_data.py` leaves an unalignable utterance
out of the output entirely rather than writing it without timestamps, so every
row in the aligned file has words by construction and that ratio is always
100%. The number that means something is how many rows survived.

**Expect:** ~600 hours, **>95% with word alignments**, no `UNEXPECTED`
characters, and word timestamps that increase sensibly. A low alignment rate
means the aligner and your transcripts disagree — see [README
§10](README.md#10-troubleshooting) before spending another cent.

### 2.6 Delete the VM, keep the disk

```bash
exit                                                     # leave the ssh session
gcloud compute instances delete fa-prep --zone=us-central1-a
gcloud compute instances list                            # expect: no instances
gcloud compute disks list                                # expect: fa-data, still there
```

> ### ✅ STOP — gate 2
> You have 600 h of aligned Farsi data on `fa-data`, and no VM running.
> **Still paying: ~$3.30/day for the 1 TB disk. Nothing else.**

---

# Phase 3 — Pilot training, ~$25

Goal: 60k steps on one GPU, listen to the samples, and answer one question —
*is this data good enough to spend $800 on?* Not a good model. A verdict.

### 3.0 If your H100 quota is already approved, merge phase 3 into phase 4

The separate pilot VM exists to keep a bad-data verdict off an expensive
machine. But the verdict is a step count, not a wall-clock duration: on 8xH100
you reach 30k steps in **~75 minutes**. So start the *real* run from
[§4](#phase-4--the-real-run-800-1000), set `sample_freq: 5000` instead of
10000, and take the §3.4 listening gate one hour in:

* data is bad -> kill it, having spent ~$35 on Spot (~$110 on-demand)
* data is good -> it keeps running to 400k, with no second VM to build

That is cheaper in your time than a 14-hour A100 pilot and only slightly more
in money. Use the separate pilot below when H100 quota is still pending, or
when you want the verdict without holding an 8-GPU reservation.

### 3.1 Pick a pilot machine

| machine | GPU | Spot $/h | 60k steps | `batch_size` / `grad_accum_steps` |
|---|---|---|---|---|
| `a2-ultragpu-1g` | 1×A100 80 GB | ~$1.60 | ~14 h (~$22) | 64 / 1 |
| `g2-standard-8` | 1×L4 24 GB | ~$0.30 | ~48 h (~$14) | 16 / 4 |

Same disk, same commands, so create it exactly as in §2.2 with the new
`--machine-type` and name `fa-pilot`, then repeat §2.3 — **but skip the
`mkfs` line**, the disk already holds your data:

```bash
sudo mount /dev/disk/by-id/google-fa-data /mnt/data
```

### 3.2 Point the config at your data and shorten the run

```bash
cd pocket-tts
sed -i 's|data/farsi_600h|/mnt/data/farsi_600h|g' \
    training/farsi/configs/model_farsi.yaml training/farsi/configs/lsd_scratch_fa.yaml
sed -i 's|^max_steps: 400000|max_steps: 60000|' training/farsi/configs/lsd_scratch_fa.yaml
sed -i 's|^run_dir: runs/lsd_scratch_fa|run_dir: /mnt/data/runs/pilot_fa|' training/farsi/configs/lsd_scratch_fa.yaml
# on an A100 (one GPU, 64 x 1 = 64):
sed -i 's|^batch_size: 8|batch_size: 64|' training/farsi/configs/lsd_scratch_fa.yaml
# on an L4 instead (one GPU, 16 x 4 = 64) -- BOTH lines, the second one matters:
#   sed -i 's|^batch_size: 8|batch_size: 16|;s|^grad_accum_steps: 1|grad_accum_steps: 4|' \
#       training/farsi/configs/lsd_scratch_fa.yaml

grep -nE "jsonl|tokenizer_path|batch_size|grad_accum|max_steps|run_dir" \
    training/farsi/configs/lsd_scratch_fa.yaml training/farsi/configs/model_farsi.yaml
```

That last `grep` is the check: every path must point at `/mnt/data`, and
`batch_size × GPUs × grad_accum_steps` must equal **64**.

### 3.3 Train

```bash
tmux new -s pilot
uv run python training/train.py training/farsi/configs/lsd_scratch_fa.yaml
```

**Expect within the first minutes:** `flow_lm + objective: ~316M trainable
params`, then step logs.

**Watch `flow_diag`, not `flow_loss`.** With LSD's default `normalize: True`
the reported loss is uncertainty-weighted:

```python
flow_diag = flow_diag * diag_logvar.exp() / x_0.shape[-1] - diag_logvar
```

That `- diag_logvar` is unbounded below, so `flow_loss` goes **negative** once
the weighting network learns — normal, not divergence. (The upstream README's
"flow_loss ~0.35-0.4 at 2k steps" does not survive this; ignore it.)
`flow_diag` is the raw flow-matching MSE and is the interpretable one: on a
healthy 500 h Farsi run it fell from ~32 at step 10 to ~0.15 by step 4k and
kept dropping. `grad_norm` should stay order 1-3.

If `flow_diag` is flat or `grad_norm` is exploding, kill it — that is a data
problem, and more steps will not fix it.

### 3.4 The listening gate — the whole point of this phase

Samples are written every 10k steps to `/mnt/data/runs/pilot_fa/samples/`. From
your laptop:

```bash
gcloud compute scp --recurse --zone=us-central1-a \
    fa-pilot:/mnt/data/runs/pilot_fa/samples ./pilot_samples
```

Listen to `step00060000_*.wav`.

| what you hear | what it means | what to do |
|---|---|---|
| intelligible Persian words, flat/dull prosody | **exactly right for 60k steps** | proceed to phase 4 |
| Persian-sounding babble, no real words | undertrained, or transcripts do not match audio | check §2.5 alignment rate; try 20k more steps |
| not Persian phonotactics at all | audio and transcripts are misaligned | fix the data. Do not proceed. |
| words but heavy background music | too much film audio | re-run phase 2 with `--sources manatts,filimo` |
| silence, or generations that never stop | alignment / EOS problem | [README §10](README.md#10-troubleshooting) |

**If it is noise, check the corpus directly rather than the latents.** Play a
few source files against their transcripts — unambiguous, and it needs no
understanding of the codec:

```bash
uv run python -c "
import json
for i, line in enumerate(open('/mnt/data/farsi_600h/train_aligned.jsonl')):
    d = json.loads(line); print(d['path'], round(d['duration'],1), d['transcript'])
    if i == 4: break
"
```

Then scp two or three of those paths to your laptop and listen. If the audio
says what the transcript says, the corpus is fine.

Do **not** try to verify the corpus by decoding the cached Mimi latents back to
audio: encode-then-decode does not round-trip cleanly enough to judge by ear,
so it produces false alarms.

Also check that validation loss is falling — validation encodes audio on the
fly rather than from the latent cache, so it is an independent read on the
same data:

```bash
grep '"valid"' /mnt/data/runs/lsd_scratch_fa/progress.jsonl | tail -8
```

**Judge acoustic quality on EMA weights, never on `samples/`.** `write_samples`
synthesizes "from the live (raw) weights", which wobble step to step: a real run
sounded clean at 40k, echoey at 45-50k, and clean again from the EMA export at
the same step. Use `samples/` only for "is it speaking the language and
following the text". For quality, generate from the run's `model.safetensors`
(the EMA export, refreshed at every checkpoint):

```bash
sed -e 's|data/farsi_600h|/mnt/data/farsi_600h|g' \
    -e 's|^weights_path:.*|weights_path: /mnt/data/runs/<run>/model.safetensors|' \
    training/farsi/configs/model_farsi_24l.yaml > /tmp/model_fa_ema.yaml

uv run pocket-tts generate --config /tmp/model_fa_ema.yaml --voice <clean_prompt.wav> \
    --text "..." --temperature 0.3 --eos-threshold -2 --output-path /tmp/ema.wav
```

Use a clean voice prompt (Mana-TTS, not a film clip) so you are judging the
model rather than the conditioning.

Also run a real eval to establish your baseline numbers:

```bash
uv run python -m training.farsi.eval_fa /mnt/data/runs/pilot_fa \
    --manifest /mnt/data/farsi_600h/valid_aligned.jsonl \
    --use-ema --reference-floor --num-items 200
```

**Read `wer_floor` first** — that is your ASR's error rate on real recordings,
and it is the number your model is competing with, not zero. Write both numbers
down; phase 4 has to beat them.

### 3.5 Delete the VM

```bash
gcloud compute instances delete fa-pilot --zone=us-central1-a
gcloud compute instances list
```

> ### ✅ STOP — gate 3
> You have heard Persian speech come out of your own model and you have a
> baseline WER. **Total spent so far: roughly $40.** Only now is phase 4
> justified.

---

# Phase 4 — The real run, ~$800-1,000

### 4.1 Raise your budget alert first

Console → Billing → Budgets → raise to **$1,500**. Then re-read the one rule:
`a3-highgpu-8g` is **~$88/h on-demand and ~$30-40/h on Spot** — a forgotten
weekend costs more than the training run.

### 4.2 Create a GCS bucket for checkpoints

```bash
gcloud storage buckets create gs://YOUR-BUCKET-fa --location=us-central1

# The VM writes as the default compute service account, which in newer projects
# holds NO project roles. --scopes=cloud-platform grants the scope; the account
# still needs an IAM role, or every sync fails with storage.objects.create denied.
PROJECT_NUM=$(gcloud projects describe "$(gcloud config get-value project)" \
    --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding gs://YOUR-BUCKET-fa \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role=roles/storage.objectAdmin
```

This is what makes a Spot preemption survivable: the launcher mirrors the run
directory here, so a *replacement* VM resumes instead of restarting. Storage is
~$0.02/GB/month — negligible next to the GPUs.

The sync is a convenience, not a dependency. If the binding is wrong the
launcher prints one warning and keeps training; checkpoints still land in
`run_dir` on the persistent disk. Omit the bucket argument entirely to skip it.

### 4.3 Create the 8×H100 VM, on Spot

Expect `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` on the first few attempts.
Spot H100 capacity moves minute to minute — `us-central1-a` will tell you to
try `-c`, and `-c` will tell you to try `-a`. Do not sit there retyping the
command; loop over zones and machine sizes until one lands.

`fa-data` is **zonal**, so a second zone needs a copy of it:

```bash
gcloud compute snapshots create fa-data-snap \
    --source-disk=fa-data --source-disk-zone=us-central1-a
gcloud compute disks create fa-data-c \
    --source-snapshot=fa-data-snap --zone=us-central1-c --type=pd-balanced
```

Then let this run until it wins (delete the copy you did not use afterwards —
each 1 TB disk is ~$100/month):

```bash
cat > ~/try-create.sh <<'EOF'
#!/usr/bin/env bash
COMMON=(
  --image-family=common-cu129-ubuntu-2204-nvidia-580
  --image-project=deeplearning-platform-release
  --maintenance-policy=TERMINATE
  --metadata=install-nvidia-driver=True
  --boot-disk-size=200GB --boot-disk-type=pd-balanced
  --provisioning-model=SPOT --instance-termination-action=STOP
  --scopes=https://www.googleapis.com/auth/cloud-platform
)
while true; do
  for spec in "us-central1-a:fa-data" "us-central1-c:fa-data-c"; do
    zone="${spec%%:*}"; disk="${spec##*:}"
    for mt in a3-highgpu-8g a3-highgpu-4g a3-highgpu-2g; do
      printf '%s  trying %-16s in %s ... ' "$(date +%T)" "$mt" "$zone"
      if gcloud compute instances create fa-train \
           --zone="$zone" --machine-type="$mt" \
           --disk=name="$disk",device-name=fa-data,mode=rw,boot=no,auto-delete=no \
           "${COMMON[@]}" >/tmp/create.out 2>&1; then
        echo "GOT IT"; echo "=> machine=$mt zone=$zone disk=$disk"; exit 0
      fi
      echo "no capacity"
    done
  done
  echo "--- nothing available, sleeping 60s ---"; sleep 60
done
EOF
chmod +x ~/try-create.sh && ~/try-create.sh
```

Note `device-name=fa-data` stays constant even when the disk is named
`fa-data-c`: the device name is what creates `/dev/disk/by-id/google-fa-data`,
so every mount instruction below works unchanged in either zone.

**Whichever machine size wins decides one config value.** The effective batch
must be 64 in every case:

| GPUs | `batch_size` | `grad_accum_steps` | 400k steps |
|---|---|---|---|
| 8 | 8 | 1 | ~12-16 h |
| 4 | 16 | 1 | ~21 h |
| 2 | 32 | 1 | ~33 h |

`run_training.sh` reads the GPU count from `nvidia-smi` itself, so only
`batch_size` needs editing.

**The clock starts the moment it boots**, and every later `gcloud` command
needs the zone that won — `--zone=us-central1-c` if the loop landed there.
Have §4.4 pasted and ready.

### 4.4 Set up and launch, in one sitting

```bash
gcloud compute ssh fa-train --zone=us-central1-a
```

```bash
nvidia-smi                                        # expect 8 GPUs
sudo mount /dev/disk/by-id/google-fa-data /mnt/data     # NO mkfs
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone -b farsi https://github.com/<you>/pocket-tts.git pocket-tts && cd pocket-tts
# ^ your fork, NOT kyutai-labs -- upstream has no training/farsi/ (see §2.3)
uv sync && uv add --dev pyarrow && uv run hf auth login
uv run pytest training/tests/test_farsi.py -q            # 27 passed

# same edits as §3.2, but the real settings for 8 GPUs:
sed -i 's|data/farsi_600h|/mnt/data/farsi_600h|g' \
    training/farsi/configs/model_farsi.yaml training/farsi/configs/lsd_scratch_fa.yaml
sed -i 's|^run_dir: runs/lsd_scratch_fa|run_dir: /mnt/data/runs/lsd_scratch_fa|' \
    training/farsi/configs/lsd_scratch_fa.yaml
grep -nE "jsonl|tokenizer_path|batch_size|grad_accum|max_steps|run_dir" \
    training/farsi/configs/lsd_scratch_fa.yaml
# expect: batch_size 8, grad_accum_steps 1, max_steps 400000  (8 x 8 x 1 = 64)

tmux new -s train
nohup ./training/farsi/gcp/run_training.sh \
    training/farsi/configs/lsd_scratch_fa.yaml gs://YOUR-BUCKET-fa &
tail -f nohup.out
```

The launcher restarts `torchrun` after a preemption, and `train.py` resumes
from the newest checkpoint by itself, so at `ckpt_freq: 2500` a preemption
costs at most ~7 minutes of work.

### 4.4a PREFLIGHT — run this before every training run

Ten seconds, and it is not optional:

```bash
uv run python -m training.farsi.preflight training/farsi/configs/model_farsi.yaml \
    --manifest /mnt/data/farsi_600h/train_aligned.jsonl
```

Wants `PASS: audio encodes to non-degenerate latents`.

**Why this exists.** `kyutai/pocket-tts` is gated, and substituting the public
`kyutai/pocket-tts-without-voice-cloning` looks harmless — same architecture,
same tensor names, `load_state_dict(strict=True)` succeeds. But that checkpoint
has Mimi's `encoder` (22 tensors) and `encoder_transformer` (20 tensors)
**zeroed**; that is how voice cloning was removed from it. `encode_to_latent`
then returns all-zero latents, and training optimizes against silence.

Nothing warns you. It is worse than that — the failure looks like success:

| signal | what it showed |
|---|---|
| `flow_diag` | fell 32 -> 0.005, "healthy" (predicting zero is easy) |
| valid loss | fell monotonically for 30k steps |
| EOS behaviour | learned real length control from text |
| samples | a constant drone, identical for any text, voice or seed |

A real run was lost this way. The tell, if you ever see it again, is
`emb_std` collapsing to a single uniform value of **0.3677** — exactly
`0.999 ** stats_ema_steps`, the signature of averaging in a standard deviation
of zero for 1000 steps.

### 4.4b The one-time latent precompute

Before the first step, upstream's `train.py` runs Mimi over the whole training
manifest once and caches per-utterance latents, so the training loop never
re-encodes audio:

```
[INFO train] precomputing latents for train_aligned.jsonl (one-time)
encode train_aligned.jsonl:  0%|  | 5/1855 [00:31<1:27:30, 2.84s/it]
```

* **~7 minutes** on 8xH100 for a 500 h corpus, once per corpus rather than once
  per run. (The tqdm estimate in the first minute reads ~90 minutes; ignore it,
  it extrapolates from the slowest early chunks.)
* written to `<manifest dir>/latents/<mimi-hash>/`, ~3-4 GB for 500 h -- keep it
  on the persistent disk with the manifests, never on the boot disk
* resumable and atomic (existing chunks are skipped, files land via rename), so
  a preemption here costs only the chunk in flight
* `valid_jsonl` must stay pointed at the **original** manifest; `train.py`
  refuses a precomputed one so validation metrics stay exact

If your clone predates this feature, you will not see the stage at all and
training starts straight away.

### 4.5 While it runs (~16 hours)

Check in a few times rather than watching it:

```bash
tail -5 /mnt/data/runs/lsd_scratch_fa/progress.jsonl
gcloud compute instances list                     # is it even still alive after a preemption?
```

Milestones: `flow_diag` down two orders of magnitude by ~4k (see §3.3 —
`flow_loss` itself goes negative and is not a useful gauge); intelligible by
15-50k; **the acoustic-quality jump at 150-200k** — this is why
`max_steps` is 400000 and why shortening the schedule wastes the entire run;
expressivity still improving at 400k.

### 4.6 Distil the teacher into the 6-layer student

The 24-layer teacher exists to supervise this. The 6-layer student is the model
that runs on a CPU at ~6x real time, and in the repo's own numbers it matches or
beats its teacher (0.76% vs 0.82% WER) because it inherits the teacher's flow
head verbatim and has guidance baked in.

Stop the teacher run first, or systemd will relaunch it forever once it
completes:

```bash
sudo systemctl disable --now fa-train
ls -la /mnt/data/runs/lsd_scratch_fa2/checkpoint_00400000.pt   # must exist

cd ~/pocket-tts
sed -i \
  -e 's|data/farsi_600h|/mnt/data/farsi_600h|g' \
  -e 's|^distill_teacher_weights:.*|distill_teacher_weights: /mnt/data/runs/lsd_scratch_fa2/checkpoint_00400000.pt|' \
  -e 's|^run_dir: runs/lsd_distill_fa|run_dir: /mnt/data/runs/lsd_distill_fa|' \
  -e 's|^sample_freq: 10000|sample_freq: 5000|' \
  training/farsi/configs/lsd_depth_distill_fa.yaml

grep -nE "model_config|teacher|jsonl|run_dir|batch_size|max_steps" \
  training/farsi/configs/lsd_depth_distill_fa.yaml

# repoint systemd so Spot preemptions keep auto-recovering
sudo sed -i 's|lsd_scratch_fa\.yaml|lsd_depth_distill_fa.yaml|' /etc/systemd/system/fa-train.service
grep ExecStart /etc/systemd/system/fa-train.service     # must say lsd_depth_distill_fa.yaml
sudo systemctl daemon-reload && sudo systemctl enable --now fa-train
```

At startup expect `depth distillation: teacher=..., seeded 127 tensors, kept
teacher layers [0, 1, 2, 21, 22, 23]` — the "ends" seeding strategy, bottom
three and top three of the teacher's 24.

**Throughput:** ~17 it/s on 8xH100, so 200k steps is ~3.2 h. The first few
hundred steps read ~4 it/s; that is compile warmup, not the real rate.

#### Pick the best student checkpoint

Unlike the teacher run, this config sets `num_ckpt_keep: 999`, so **every**
checkpoint survives and you can choose rather than being stuck with the last
one. WER and speaker similarity typically reach teacher parity by ~40k; only
prosody settles after that, and on a small or noisy corpus later is not
automatically better.

Evaluate three and take the winner. **Use `--cfg 1.0`** — the student has
guidance baked in, and 2.0 double-applies it:

```bash
for step in 00040000 00100000 00200000; do
  CKPT=/mnt/data/runs/lsd_distill_fa/checkpoint_${step}.pt
  [ -f "$CKPT" ] || continue
  echo "=== $step ==="
  uv run python -m training.farsi.eval_fa /mnt/data/runs/lsd_distill_fa \
      --manifest /mnt/data/farsi_600h/manatts_eval.jsonl --checkpoint "$CKPT" \
      --use-ema --reference-floor --num-items 100 --batch-size 8 \
      --cfg 1.0 --eos-threshold -2 2>&1 | grep FINAL
done
```

Compare `wer` against `wer_floor` (the same ASR's error on the real
recordings), plus `sim`, against your teacher's numbers. Then check the winner
on held-out speakers, where the number that matters is the loop count:

```bash
BEST=/mnt/data/runs/lsd_distill_fa/checkpoint_<the winner>.pt
uv run python -m training.farsi.eval_fa /mnt/data/runs/lsd_distill_fa \
    --manifest /mnt/data/farsi_600h/valid_aligned.jsonl --checkpoint "$BEST" \
    --use-ema --num-items 50 --batch-size 8 --cfg 1.0 --eos-threshold -2 2>&1 | grep FINAL

uv run python -c "
import json, glob, os, jiwer, statistics as st
d = max(glob.glob('/mnt/data/runs/lsd_distill_fa/fa_eval_*'), key=os.path.getmtime)
recs = [r for r in json.load(open(d+'/records.json')) if r['ref']]
w = sorted(jiwer.wer(r['ref'], r['hyp']) for r in recs)
print(f'median {st.median(w):.2f}  mean {sum(w)/len(w):.2f}  loops {sum(x>1.0 for x in w)}/{len(w)}')
"
```

And listen. Your ear outranks these numbers when the references are
subtitle-derived — generate the same sentence from the winning checkpoint's EMA
export and compare with the teacher's.

```bash
# get the models off the machine BEFORE deleting it
gcloud storage rsync --recursive /mnt/data/runs/lsd_distill_fa gs://YOUR-BUCKET-fa/lsd_distill_fa
gcloud storage cp /mnt/data/farsi_600h/tokenizer.model gs://YOUR-BUCKET-fa/farsi_600h/tokenizer.model
```

### 4.7 Then, the most important command in this document

```bash
gcloud compute instances delete fa-train --zone=us-central1-a
gcloud compute instances list                     # expect: nothing
```

Only after the models are safely in GCS and on your laptop.

**Keep more checkpoints than you think you need.** The teacher config ships
`num_ckpt_keep: 3`. On a real run the validation curves diverged from ~150k
while training loss kept falling, and the 400k model measured *worse* on
intelligibility than 55k had — but the 150k checkpoint was long gone, so there
was nothing to roll back to. If your corpus is smaller or noisier than
HiFiTTS-2, raise `num_ckpt_keep` to 999 on the teacher too and pick by
evaluation, exactly as §4.6 does for the student.

---

### 4.8 Operational troubleshooting

Everything below was hit in a real run of this runbook, in this order.

| symptom | cause | fix |
|---|---|---|
| `pytest` reports ~64 tests and no Farsi ones | the VM cloned upstream, which has no `training/farsi/` | scp it up or clone your fork (§2.3); verify with `pytest training/tests/test_farsi.py -q` |
| `zsh: no such file or directory: the` | you pasted a `<placeholder>` and the shell read `<` as a redirect | fill in the value, or put it in a variable first |
| quota rows are impossible to find in the console | the console shows display names, not API constants | see the mapping table in §1.3 |
| `gcloud compute regions describe` prints giant parallel arrays | missing `--flatten="quotas[]"` | §1.3 |
| `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` | Spot GPU capacity, not quota | the retry loop in §4.3; accept a smaller A3 |
| `Connection closed by remote host` mid-session, then IAP tunnel errors | Spot preempted the VM; a stopped VM has **no external IP**, so gcloud silently falls back to IAP | `gcloud compute instances list` — if `TERMINATED`, `instances start` it (or delete and recreate on-demand) |
| `Required 'compute.instances.delete' permission` | you ran the delete **from inside the VM** | exit back to your laptop; the VM's service account cannot delete itself |
| `/mnt/data` empty after a restart | the mount is not in `/etc/fstab`; the job then re-downloads onto the boot disk and fills it | add the fstab line in §2.3 on every VM |
| `storage.objects.create` denied on every sync | the default compute service account has no IAM role on the bucket | grant `roles/storage.objectAdmin` (§4.2) |
| GCS bucket stays empty although training runs | under systemd, `gcloud` is not on the unit's `PATH` (Ubuntu images put it in `/snap/bin`) | `which gcloud` on the VM, then add that directory to `Environment=PATH=` in the unit; `run_training.sh` now prepends the usual locations |
| NCCL dies on all ranks: `DistBackendError ... invalid usage` | GCP's A3 images export `NCCL_NET=gIB` and an **A3 Ultra** tuner config; A3 High has no such fabric | `export NCCL_NET=Socket` — `run_training.sh` now sets this itself. Verify with the 8-rank `all_reduce` test below |
| launcher says `torchrun exited 0` yet keeps retrying | fixed: `$?` after a failed `if` is the *if statement's* status, not the command's | update `run_training.sh`; it now reports the real code and stops after 3 fast failures |
| `precomputing latents` before training starts | upstream's one-time Mimi cache | expected; see §4.4b |
| samples are a constant tone identical for every text/voice/seed, while losses look great | Mimi's encoder is zeroed — the `without-voice-cloning` checkpoint | §4.4a preflight; use the gated `kyutai/pocket-tts` weights, delete the latents cache, restart |
| `emb_std` is one uniform value of 0.3677 | `0.999 ** 1000`: the stats EMA averaged in a std of zero | same as above — your latents are zero |

The NCCL check, worth running once on any new multi-GPU VM before starting a
long job — it takes ten seconds and isolates the fabric from the training code:

```bash
cat > /tmp/nccl_test.py <<'EOF'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
t = torch.ones(1, device=f"cuda:{r}") * r
dist.all_reduce(t)
print(f"rank {r}: all_reduce -> {t.item()}")
dist.destroy_process_group()
EOF
NCCL_NET=Socket NCCL_DEBUG=WARN uv run torchrun --nproc-per-node 8 /tmp/nccl_test.py 2>&1 | tail -20
```

Every rank must print the sum `0+1+...+n-1` (28.0 on 8 GPUs). `NCCL WARN lib
wrapper not initialized` from `ibvwrap` alongside correct sums is harmless — it
is NCCL probing for InfiniBand, not finding it, and falling back.

**Read failures with `grep`, not `tail`.** Torchrun's per-rank epilogue is
hundreds of lines, so `tail -40` shows you the summary and hides the actual
error:

```bash
grep -nE "Traceback|RuntimeError|CUDA out of memory|Killed|AssertionError|Error:" \
    /mnt/data/train.log | tail -30
grep -i nccl /mnt/data/train.log | head -20
```

# 5. Cost hygiene

### 5.1 Every day you work on this

```bash
gcloud compute instances list    # RUNNING = burning money
gcloud compute disks list        # cost even with no VM
```

Console → Billing → Reports shows yesterday's actual spend. Check it the day
after your first VM — that is when a surprise is still small.

### 5.2 What costs money when you are not looking

| thing | while idle | fix |
|---|---|---|
| a **stopped** VM | $0 for GPU/CPU, but its **boot disk** still bills | delete the VM, not just stop it |
| the 1 TB `fa-data` disk | ~$3.30/day, forever | delete when the project ends |
| **local SSD** on A3 machines | data is **destroyed** when the VM stops | never keep anything you care about there |
| GCS bucket | ~$0.02/GB/month | keep it; it is the cheap part |
| a Spot VM that got preempted and auto-restarted | full price again while running | that is intended — but check it is making progress |

### 5.3 When the project is finished

```bash
gcloud compute instances list                            # must be empty
gcloud compute disks delete fa-data --zone=us-central1-a  # AFTER the model is in GCS
gcloud storage ls gs://YOUR-BUCKET-fa                    # confirm the model is really there
```

Keep the bucket (a few cents a month) until the model is uploaded to
HuggingFace — see [README §9](README.md#9-distillation-and-shipping).

---

# 6. If you only remember five things

1. **Phases 0-3 cost ~$40 and catch ~every problem.** Phase 4 is just compute.
2. **Prepare data on a cheap VM**, never on the 8×H100 box.
3. **Delete VMs, do not stop them.** `gcloud compute instances list`, daily —
   and run every `gcloud` command from your **laptop**, not from inside the VM,
   which cannot delete itself.
4. **A budget alert does not stop spending.** Only you do.
5. **Listen to the pilot samples before spending $800.** If they are not
   Persian, the data is wrong and no amount of H100 time fixes that.
