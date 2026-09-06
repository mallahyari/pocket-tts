#!/usr/bin/env bash
# Preemption-proof training launcher for a Spot GCP VM.
#
# train.py resumes from the newest checkpoint in run_dir on its own, so
# surviving a preemption is only a matter of (a) restarting the process and
# (b) making sure run_dir still exists. This script does both: it restarts
# torchrun until the run reaches max_steps, and mirrors run_dir to GCS after
# every attempt so the run survives losing the VM entirely.
#
#   ./training/farsi/gcp/run_training.sh training/farsi/configs/lsd_scratch_fa.yaml \
#       gs://my-bucket/pocket-tts-fa
#
# Run it under systemd (or tmux) so a Spot restart brings it back by itself.
set -uo pipefail

# GCP's A3 images export NCCL_NET=gIB plus an A3-Ultra tuner config, for the
# RDMA NICs that a3-highgpu (A3 High) does not have -- NCCL then dies at init
# with "invalid usage" on every rank. All 8 GPUs are in one box and talk over
# NVLink, so no network plugin is needed. Override with POCKET_TTS_NCCL_NET on
# a machine that really does have the fabric (A3 Ultra, A4, multi-node).
export NCCL_NET="${POCKET_TTS_NCCL_NET:-Socket}"

# Under systemd the PATH is whatever the unit sets, and GCP images put gcloud in
# different places (/snap/bin on Ubuntu images, /usr/bin elsewhere). Without
# this the GCS mirror fails silently for the whole run.
export PATH="/snap/bin:/usr/lib/google-cloud-sdk/bin:${PATH}"

CONFIG="${1:?usage: run_training.sh <config.yaml> [gs://bucket/prefix]}"
GCS_PREFIX="${2:-}"
NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
RUN_DIR="$(uv run python -c "
from training.args import load_args
print(load_args('${CONFIG}').run_dir)
")"

echo "config=${CONFIG} run_dir=${RUN_DIR} gpus=${NPROC} gcs=${GCS_PREFIX:-<none>}"

# Pull a previous run back down before the first attempt: on a fresh VM after a
# preemption this is what makes the restart a resume instead of a restart.
if [[ -n "${GCS_PREFIX}" && ! -d "${RUN_DIR}" ]]; then
    mkdir -p "${RUN_DIR}"
    gcloud storage rsync --recursive "${GCS_PREFIX}/$(basename "${RUN_DIR}")" "${RUN_DIR}" || true
fi

# One warning per failing sync, not one gcloud error per object: a bucket the
# VM's service account cannot write to would otherwise bury the training log.
sync_warned=0
sync_up() {
    [[ -n "${GCS_PREFIX}" ]] || return 0
    if ! gcloud storage rsync --recursive "${RUN_DIR}" \
            "${GCS_PREFIX}/$(basename "${RUN_DIR}")" >/tmp/gcs-sync.log 2>&1; then
        if [[ ${sync_warned} -eq 0 ]]; then
            sync_warned=1
            echo "WARNING: cannot sync ${RUN_DIR} to ${GCS_PREFIX} -- see /tmp/gcs-sync.log."
            echo "         Training continues; checkpoints are still written to ${RUN_DIR}."
            echo "         Usually the VM service account lacks storage.objectAdmin on the bucket:"
            echo "           gcloud storage buckets add-iam-policy-binding ${GCS_PREFIX%%/*}//${GCS_PREFIX#gs://} \\"
            echo "             --member=serviceAccount:\$(gcloud config get-value account) --role=roles/storage.objectAdmin"
        fi
        return 1
    fi
    return 0
}
trap 'sync_up || true' EXIT

# Syncing only when torchrun exits leaves GCS empty for the whole run, which is
# precisely when losing the disk would hurt most. Mirror on an interval instead.
# A checkpoint being written while rsync runs can land partially copied; the
# next pass overwrites it, and num_ckpt_keep older complete checkpoints remain.
periodic_sync() {
    [[ -n "${GCS_PREFIX}" ]] || return 0
    while true; do
        sleep "${SYNC_INTERVAL:-600}"
        sync_up || true
    done
}

attempt=0
fast_failures=0
while true; do
    attempt=$((attempt + 1))
    last_start=${SECONDS}
    echo "=== attempt ${attempt} $(date -Is) ==="
    # Run it plainly and read $? straight away: inside `if cmd; then ... fi`,
    # a failing cmd still leaves $? as the *if statement's* status, which is 0.
    periodic_sync &
    syncer=$!
    uv run torchrun --nproc-per-node "${NPROC}" training/train.py "${CONFIG}"
    status=$?
    kill "${syncer}" 2>/dev/null || true
    wait "${syncer}" 2>/dev/null || true
    sync_up || true
    if [[ ${status} -eq 0 ]]; then
        echo "training finished"
        exit 0
    fi
    echo "torchrun exited ${status}; resuming from the last checkpoint in 60s"
    # A run that dies instantly is a broken config, not a preemption -- looping
    # on it just burns GPU time, so give up after several rapid failures.
    if (( SECONDS - last_start < 120 )); then
        fast_failures=$(( fast_failures + 1 ))
    else
        fast_failures=0
    fi
    if (( fast_failures >= 3 )); then
        echo "three failures in under two minutes each -- stopping so you can read the error."
        exit "${status}"
    fi
    sleep 60
done
