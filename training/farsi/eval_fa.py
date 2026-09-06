"""Farsi TTS eval: WER, speaker similarity and UTMOS on a held-out manifest.

The LibriSpeech eval in training/eval/librispeech.py is English to the bone
(its .lst protocol, its ASR and its text normalizer), so a Farsi run needs its
own scorer. Everything that is language-neutral -- checkpoint loading, Mimi
decoding, the x-vector embedder -- is imported from it rather than copied.

The eval list is built from a manifest (typically valid_aligned.jsonl, or a
manifest of the Common Voice fa test split): utterances are paired inside each
speaker, so the voice prompt and the reference recording are the same voice
but never the same audio as the text being synthesized.

  - WER: whisper (or any HF ASR) transcript vs the reference text, both run
    through training.farsi.normalize_fa, punctuation and ZWNJ removed.
  - speaker similarity: cosine between generated audio and the real recording
    of the target utterance (WavLM x-vector).
  - UTMOS: audio quality, if the `utmos_pytorch` package is importable. It was
    trained on English MOS ratings, so read it as a relative signal between
    your own checkpoints, not as an absolute Persian MOS.

Usage:
    python -m training.farsi.eval_fa runs/lsd_scratch_fa \
        --manifest data/farsi_600h/valid_aligned.jsonl --use-ema
"""

import argparse
import json
import logging
import multiprocessing
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jiwer
import numpy as np
import sphn
import torch

from training.eval.librispeech import MIN_FRAMES, EvalResults, latents_to_wav, load_run
from training.farsi.normalize_fa import KEPT_PUNCT, ZWNJ, normalize

logger = logging.getLogger("eval_fa")


class FaEvalResults(EvalResults):
    """EvalResults plus the ASR's own error rate on the real recordings.

    Persian ASR is nowhere near as good as English ASR, so a Farsi TTS scored
    with whisper cannot reach the sub-1% WER the English model reports -- the
    measurement floor is tens of percent. `wer_floor` is that floor, measured
    on the same items by transcribing the reference recordings: the number to
    compare your model against is the floor, not zero.
    """

    wer_floor: float | None = None


DEFAULT_ASR = "openai/whisper-large-v3"
# Just under whisper's 30s window: at or past it the feature extractor emits
# 3001 mel frames instead of 3000 and batched collation dies.
WHISPER_MAX_SAMPLES = 16000 * 30 - 160
_PUNCT_RE = re.compile(f"[{re.escape(KEPT_PUNCT)}]")


def wer_text(text: str) -> str:
    """Scoring form: normalized Persian, no punctuation, ZWNJ as a space.

    "می‌رود" and "می رود" are the same utterance spoken aloud, so a ZWNJ
    disagreement between the ASR and the reference must not count as an error.
    """
    text = normalize(text).replace(ZWNJ, " ")
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text)).strip()


def build_items(manifest: Path, num_items: int | None, seed: int) -> list[dict]:
    """Pair utterances within each speaker: prompt = one, target = another."""
    by_speaker: defaultdict = defaultdict(list)
    with open(manifest) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_speaker[row.get("speaker", "unknown")].append(row)
    rng = random.Random(seed)
    items = []
    for rows in by_speaker.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (r["path"], r.get("start", 0.0)))
        rng.shuffle(rows)
        for i in range(0, len(rows) - 1, 2):
            prompt, target = rows[i], rows[i + 1]
            items.append({"prompt": prompt, "target": target, "text": target["transcript"]})
    rng.shuffle(items)
    if num_items:
        items = items[:num_items]
    for idx, item in enumerate(items):
        item["idx"] = idx
    return items


def read_row(row: dict, sample_rate: int, max_sec: float | None = None) -> np.ndarray:
    """The audio window a manifest row points at, mono at `sample_rate`."""
    start = float(row.get("start", 0.0))
    duration = float(row["duration"])
    if max_sec:
        duration = min(duration, max_sec)
    wav, sr = sphn.read(row["path"], start_sec=start, duration_sec=duration)
    wav = wav.mean(axis=0)
    if sr != sample_rate:
        wav = sphn.resample(wav, src_sample_rate=int(sr), dst_sample_rate=int(sample_rate))
    return np.ascontiguousarray(wav, dtype=np.float32)


def build_transcriber(asr_name: str, device):
    """A batched Persian transcriber. Whisper needs the language pinned: it
    otherwise detects Arabic on short or noisy Persian clips."""
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_name,
        device=device,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    kwargs = {}
    is_whisper = "whisper" in asr_name
    if is_whisper:
        kwargs["generate_kwargs"] = {"language": "fa", "task": "transcribe"}

    def prepare(wavs: list) -> list:
        # Whisper's feature extractor emits 3000 mel frames for audio up to its
        # 30s window and 3001 for anything past it, and the pipeline's batch
        # collator cannot reconcile the two ("expanded size ... 3000 ... 3001").
        # Mimi's decoder appends padding, so a --max-sec 30 generation lands
        # just over the boundary. Trim a hair under it.
        return [w[:WHISPER_MAX_SAMPLES] for w in wavs] if is_whisper else wavs

    def transcribe(wavs: list) -> list[str]:
        batch = prepare(wavs)
        try:
            outs = asr(
                [{"array": w, "sampling_rate": 16000} for w in batch],
                batch_size=len(batch),
                **kwargs,
            )
            return [o["text"] for o in outs]
        except RuntimeError as exc:  # noqa: BLE001 -- one bad row must not lose the eval
            logger.warning(f"batched ASR failed ({exc}); falling back to one at a time")
            out = []
            for w in batch:
                try:
                    out.append(asr({"array": w, "sampling_rate": 16000}, **kwargs)["text"])
                except RuntimeError:
                    out.append("")
            return out

    return transcribe


def score_items(items: list[dict], device, args) -> tuple[list[dict], int]:
    model, mimi, step = load_run(
        args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint
    )
    torch.manual_seed(args.seed)
    transcribe = build_transcriber(args.asr, device)

    spk = None
    if not args.skip_sim:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        spk_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        spk_model = (
            WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        )

        def embed(wavs: list) -> torch.Tensor:
            inputs = spk_fe(
                [w.cpu().numpy() if torch.is_tensor(w) else w for w in wavs],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
            return spk_model(**inputs).embeddings

        spk = embed

    utmos = None
    try:
        from utmos_pytorch import UTMOSScoreTorch

        utmos = UTMOSScoreTorch(device=str(device))
    except ImportError:
        pass

    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    max_frames = int(args.max_sec * mimi.frame_rate)
    records = []
    bs = max(1, args.batch_size)
    for start_i in range(0, len(items), bs):
        chunk = items[start_i : start_i + bs]
        tokens = [torch.tensor(tokenize(normalize(c["text"])), dtype=torch.long) for c in chunk]
        with torch.no_grad():
            voice_latents = []
            for c in chunk:
                wav = read_row(c["prompt"], mimi.sample_rate, args.voice_sec)
                voice_latents.append(
                    mimi.encode_to_latent(torch.from_numpy(wav)[None, None].to(device))[0]
                )
            outs = model.generate(
                tokens,
                voice_latents,
                max_frames=max_frames,
                temp=args.temp,
                n_steps=args.n_steps,
                cfg_coef=args.cfg,
                eos_threshold=args.eos_threshold,
            )
        good, gens = [], []
        for item, latents in zip(chunk, outs):
            capped = int(latents.shape[0] >= max_frames)
            ref_text = wer_text(item["text"])
            if latents.shape[0] < MIN_FRAMES:
                records.append({"ref": ref_text, "hyp": "", "silent": 1, "no_eos": capped})
                continue
            with torch.no_grad():
                audio = latents_to_wav(mimi, latents, device)
            if args.save_audio:
                out_dir = Path(args.save_audio)
                out_dir.mkdir(parents=True, exist_ok=True)
                sphn.write_wav(
                    str(out_dir / f"{item['idx']:04d}.wav"),
                    audio.cpu().numpy(),
                    int(mimi.sample_rate),
                )
            try:
                gen16k = sphn.resample(
                    audio.cpu().numpy(), src_sample_rate=mimi.sample_rate, dst_sample_rate=16000
                )
            except BaseException:  # noqa: BLE001 -- sphn panics (Rust) on degenerate audio
                records.append({"ref": ref_text, "hyp": "", "silent": 1, "no_eos": capped})
                continue
            gens.append(gen16k)
            good.append((item, capped))
        if not good:
            continue

        hyps = transcribe(gens)
        refs_audio = None
        if spk is not None or args.reference_floor:
            refs_audio = [read_row(item["target"], 16000) for item, _ in good]
        sims = None
        if spk is not None:
            with torch.no_grad():
                e_gen, e_ref = spk(gens), spk(refs_audio)
                sims = torch.nn.functional.cosine_similarity(e_gen, e_ref, dim=-1).tolist()
        ref_hyps = transcribe(refs_audio) if args.reference_floor else None
        for i, ((item, capped), hyp) in enumerate(zip(good, hyps)):
            rec = {
                "ref": wer_text(item["text"]),
                "hyp": wer_text(hyp),
                "silent": 0,
                "no_eos": capped,
            }
            if sims is not None:
                rec["sim"] = sims[i]
            if ref_hyps is not None:
                rec["hyp_ref"] = wer_text(ref_hyps[i])
            if utmos is not None:
                with torch.no_grad():
                    rec["utmos"] = float(
                        utmos.score(torch.from_numpy(gens[i])[None, None].to(device))
                    )
            records.append(rec)
    return records, step


def _shard_worker(payload):
    device_idx, items, args = payload
    torch.cuda.set_device(device_idx)
    return score_items(items, torch.device("cuda", device_idx), args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--manifest", required=True, help="held-out manifest (valid_aligned.jsonl)")
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=2.0, help="1.0 for a distilled student")
    parser.add_argument("--eos-threshold", type=float, default=-1.0)
    parser.add_argument("--max-sec", type=float, default=30.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--num-items", type=int, default=500)
    parser.add_argument("--asr", default=DEFAULT_ASR, help="any HF ASR model that speaks Persian")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument(
        "--reference-floor",
        action="store_true",
        help="also transcribe the real recordings and report their WER: the floor your "
        "generated WER is competing with, given how good Persian ASR is",
    )
    parser.add_argument("--checkpoint", default=None, help="pin a checkpoint instead of the latest")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voice-sec", type=float, default=5.0)
    parser.add_argument("--save-audio", default=None, help="directory for the generated wavs")
    parser.add_argument("--shards", type=int, default=0, help="GPUs to split over (0 = all)")
    args = parser.parse_args()

    items = build_items(Path(args.manifest), args.num_items, args.seed)
    if not items:
        raise SystemExit(
            f"{args.manifest} yielded no pairs: it needs at least two utterances "
            "sharing a `speaker` field"
        )
    n_gpu = torch.cuda.device_count()
    shards = max(1, min(args.shards or n_gpu or 1, n_gpu or 1, len(items)))
    logger.info(f"{len(items)} items over {shards} shard(s), asr={args.asr}")

    if shards == 1:
        records, step = score_items(items, torch.device("cuda" if n_gpu else "cpu"), args)
    else:
        payloads = [(i, items[i::shards], args) for i in range(shards)]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(shards, mp_context=ctx) as pool:
            results = list(pool.map(_shard_worker, payloads))
        records = [r for recs, _ in results for r in recs]
        step = results[0][1]

    refs = [r["ref"] for r in records if r["ref"]]
    hyps = [r["hyp"] for r in records if r["ref"]]
    sims = [r["sim"] for r in records if "sim" in r]
    moses = [r["utmos"] for r in records if "utmos" in r]
    floor_hyps = [r["hyp_ref"] for r in records if "hyp_ref" in r and r["ref"]]
    floor_refs = [r["ref"] for r in records if "hyp_ref" in r and r["ref"]]
    results = FaEvalResults(
        wer_floor=jiwer.wer(floor_refs, floor_hyps) if floor_hyps else None,
        asr=args.asr,
        step=step,
        num_items=len(records),
        wer=jiwer.wer(refs, hyps),
        sim=sum(sims) / len(sims) if sims else None,
        utmos=sum(moses) / len(moses) if moses else None,
        silent=sum(r["silent"] for r in records),
        no_eos=sum(r["no_eos"] for r in records),
        temp=args.temp,
        cfg=args.cfg,
        n_steps=args.n_steps,
    )
    name = f"fa_eval_step{step}_t{args.temp}_cfg{args.cfg}" + ("" if args.use_ema else "_raw")
    out_dir = Path(args.run_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(results.model_dump_json(indent=2))
    (out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    logger.info(f"FINAL {results.model_dump_json()}")


if __name__ == "__main__":
    main()
