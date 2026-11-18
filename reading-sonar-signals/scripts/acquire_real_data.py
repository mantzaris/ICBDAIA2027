#!/usr/bin/env python3
"""Acquire bounded provider-hosted real audio, retaining exact PCM and provenance.

No acoustic transform, normalization, resampling, or model label is produced here.
MBARI daily WAVs are read with HTTP byte ranges; raw PCM is copied losslessly into
30-second WAV containers. DeepShip is the provider's public subset only.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import random
import struct
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROV = ROOT / "provenance/data"
MANIFEST = ROOT / "data/manifests"
REVISION = "77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6"
SEED = 20261117


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(url, **kwargs):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=90, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def download_deepship(entry):
    rel = entry["path"]
    path = RAW / "deepship" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/irfankamboh/DeepShip/{REVISION}/{rel}"
    if not path.exists():
        body = get(url).content
        path.write_bytes(body)
    else:
        body = path.read_bytes()
    if len(body) != entry["size"]:
        raise ValueError(f"Provider byte length mismatch for {rel}")
    with sf.SoundFile(io.BytesIO(body)) as wav:
        rate, channels, frames = wav.samplerate, wav.channels, wav.frames
        subtype = wav.subtype
        decoded = wav.read(dtype="float32", always_2d=True)
        if decoded.shape != (frames, channels):
            raise ValueError(f"Truncated PCM in {rel}")
    result = dict(dataset="DeepShip public GitHub subset", domain="deepship",
                  source_id=rel, group_id="deepship:" + rel, source_url=url,
                  provider_url="https://github.com/irfankamboh/DeepShip",
                  version=REVISION, path=str(path.relative_to(ROOT)),
                  sha256=sha(body), bytes=len(body), sample_rate=rate,
                  channels=channels, subtype=subtype, frames=frames,
                  duration_seconds=frames/rate, source_class=rel.split("/")[0],
                  calibration="not supplied with this public subset",
                  license="No explicit audio license found in public author repository; redistribution not authorized by this audit",
                  grouping_level="original recording file only; vessel/session/day identities unavailable",
                  prior_sibling_project_file_reuse=False,
                  underlying_overlap_with_sibling_onc_data="not ruled out; no files copied from sibling project",
                  acquired_at_utc=datetime.now(timezone.utc).isoformat())
    print("downloaded", rel, round(frames/rate, 2), flush=True)
    return result


def mbari_header(url):
    response = get(url, headers={"Range": "bytes=0-4095"})
    if response.status_code != 206 or len(response.content) != 4096:
        raise ValueError("Server did not honor bounded header range")
    data = response.content
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected standard RIFF WAV")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        tag, size = data[pos:pos+4], struct.unpack_from("<I", data, pos+4)[0]
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", data, pos+8)
        if tag == b"data":
            if fmt is None:
                raise ValueError("WAV fmt missing")
            encoding, channels, rate, byte_rate, block_align, bits = fmt
            if encoding != 1 or block_align != channels * bits // 8:
                raise ValueError("Only integer PCM accepted")
            return dict(data_offset=pos+8, data_bytes=size, channels=channels,
                        sample_rate=rate, bits=bits, block_align=block_align,
                        etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"),
                        header_sha256=sha(data), header_bytes=data)
        pos += 8 + size + size % 2
    raise ValueError("WAV data chunk not found in bounded header")


def download_mbari(item):
    key = item["key"]
    url = "https://pacific-sound-16khz.s3.us-west-2.amazonaws.com/" + key
    header = mbari_header(url)
    # Sampling rule fixed before looking at any waveform: keyed pseudorandom
    # minute between 01:00 and 22:59 UTC, same 30-second duration for each day.
    start = 3600 + (int(sha(key.encode())[:8], 16) % (22 * 60)) * 60
    frames = 30 * header["sample_rate"]
    count = frames * header["block_align"]
    rejected_zero_intervals = []
    for retry in range(24):
        candidate_start = (start + retry * 3600) % 86400
        begin = header["data_offset"] + candidate_start * header["sample_rate"] * header["block_align"]
        response = get(url, headers={"Range": f"bytes={begin}-{begin+count-1}",
                                     "If-Match": header["etag"]})
        if response.status_code != 206 or len(response.content) != count:
            raise ValueError(f"Server did not return exact PCM range for {key}")
        pcm = response.content
        if pcm.strip(b"\x00"):
            start = candidate_start
            break
        rejected_zero_intervals.append(candidate_start)
    else:
        raise ValueError("all_zero_at_24_hourly_offsets:" + key)
    scene_id = "mb_" + sha(key.encode())[:16]
    path = RAW / "scenes" / (scene_id + ".wav")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(header["channels"])
        wav.setsampwidth(header["bits"]//8)
        wav.setframerate(header["sample_rate"])
        wav.writeframes(pcm)
    with wave.open(str(path)) as wav:
        if wav.readframes(frames) != pcm:
            raise ValueError("PCM copy did not roundtrip exactly")
    (PROV / (scene_id + "_source_header.bin")).write_bytes(header.pop("header_bytes"))
    source = dict(dataset="MBARI Pacific Sound 16 kHz daily release", domain="mbari",
                  source_id=key, group_id="mbari:day:" + key.split("MARS-")[1][:8],
                  source_url=url, provider_url="https://registry.opendata.aws/pacific-sound/",
                  version="provider object ETag recorded", sha256=None,
                  sha256_scope="Full daily source not downloaded; exact excerpt PCM and header hashed",
                  bytes=item["size"], **header,
                  source_start_seconds=start, excerpt_duration_seconds=30,
                  pcm_sha256=sha(pcm), http_byte_range=[begin, begin+count-1],
                  rejected_all_zero_intervals_seconds=rejected_zero_intervals,
                  calibration="Provider metadata available; this experiment uses unchanged digital PCM, no pressure conversion",
                  license="CC BY 4.0", redistribution="Permitted with attribution under CC BY 4.0",
                  grouping_level="recording day; entire MBARI domain reserved as external test",
                  prior_sibling_project_file_reuse=False,
                  acquired_at_utc=datetime.now(timezone.utc).isoformat())
    scene = dict(scene_id=scene_id, stratum="real", domain="mbari", split="test",
                 test_domain="external", group_id=source["group_id"], source_id=key,
                 path=str(path.relative_to(ROOT)), source_start_seconds=start,
                 duration_seconds=30, sample_rate=header["sample_rate"], channels=header["channels"],
                 sha256=sha(path.read_bytes()), pcm_sha256=sha(pcm),
                 label_status="unannotated", source_class=None, source_url=url,
                 normalization="none", annotation_required=True)
    print("downloaded", key, "offset", start, flush=True)
    return source, scene


def assign_deepship(sources):
    # Parent-level split is selected using duration/capacity only, without
    # examining audio. The limited public subset cannot supply vessel IDs.
    usable = [x for x in sources if x["duration_seconds"] >= 30]
    capacities = {x["source_id"]: min(10, int(x["duration_seconds"] // 30)) for x in usable}
    total = sum(capacities.values())
    targets = {"train":90, "validation":30, "test":30}
    if total < sum(targets.values()):
        raise RuntimeError(f"Only {total} eligible 30 s scenes at max 10 per source; document reduction rather than silently change")
    rng = random.Random(SEED)
    selected = None
    for attempt in range(20000):
        ordered = usable.copy()
        rng.shuffle(ordered)
        n = len(ordered)
        groups = dict(train=ordered[:round(n*.60)],
                      validation=ordered[round(n*.60):round(n*.80)],
                      test=ordered[round(n*.80):])
        if all(sum(capacities[x["source_id"]] for x in groups[k]) >= target for k,target in targets.items()):
            selected = groups
            break
    if selected is None:
        raise RuntimeError("Cannot meet scene counts with disjoint file groups")
    scenes = []
    for split, group in selected.items():
        # Round-robin across parent files, limiting domination by long files.
        candidates = []
        for offset_index in range(10):
            for source in sorted(group, key=lambda x:x["source_id"]):
                capacity = capacities[source["source_id"]]
                if offset_index < capacity:
                    total_blocks = int(source["duration_seconds"]//30)
                    block = offset_index * total_blocks // capacity
                    candidates.append((source, block * 30))
        for source, start in candidates[:targets[split]]:
            scene_id = "ds_" + sha(f"{source['source_id']}:{start}".encode())[:16]
            target = RAW / "scenes" / (scene_id + ".wav")
            with sf.SoundFile(str(ROOT / source["path"])) as wav:
                wav.seek(start * wav.samplerate)
                samples = wav.read(30 * wav.samplerate, dtype="float32", always_2d=True)
            pcm = samples.tobytes()
            sf.write(str(target), samples, source["sample_rate"], subtype=source["subtype"])
            restored, _ = sf.read(str(target), dtype="float32", always_2d=True)
            if restored.tobytes() != pcm:
                raise ValueError("DeepShip floating-point samples failed bit-exact copy")
            scenes.append(dict(scene_id=scene_id, stratum="real", domain="deepship", split=split,
                               test_domain="internal" if split=="test" else None,
                               group_id=source["group_id"], source_id=source["source_id"],
                               path=str(target.relative_to(ROOT)), source_start_seconds=start,
                               duration_seconds=30, sample_rate=source["sample_rate"], channels=source["channels"],
                               sha256=sha(target.read_bytes()), pcm_sha256=sha(pcm),
                               label_status="unannotated", source_class=source["source_class"],
                               source_url=source["source_url"], normalization="none", annotation_required=True))
    return scenes, {split:[x["source_id"] for x in group] for split,group in selected.items()}


def main():
    for path in [RAW / "scenes", PROV, MANIFEST]:
        path.mkdir(parents=True, exist_ok=True)
    tree = get(f"https://api.github.com/repos/irfankamboh/DeepShip/git/trees/{REVISION}?recursive=1").json()
    (PROV / "deepship_tree.json").write_text(json.dumps(tree, indent=2))
    entries = [x for x in tree["tree"] if x["path"].lower().endswith(".wav")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        deep = list(pool.map(download_deepship, entries))
    jsonl(MANIFEST / "deepship_sources.jsonl", deep)
    scenes, groups = assign_deepship(deep)
    jsonl(MANIFEST / "real_scenes.jsonl", scenes)
    (MANIFEST / "deepship_group_split.json").write_text(json.dumps(groups, indent=2))
    listing_url = "https://pacific-sound-16khz.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=1000&prefix=2016/"
    listing = get(listing_url)
    (PROV / "mbari_2016_listing.xml").write_bytes(listing.content)
    tree_xml = ET.fromstring(listing.content)
    ns = {"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
    available = [dict(key=x.findtext("s:Key",namespaces=ns), size=int(x.findtext("s:Size",namespaces=ns)))
                 for x in tree_xml.findall("s:Contents",ns)
                 if x.findtext("s:Key",namespaces=ns).endswith(".wav")]
    # Spread 60 held-out days over the year, ordered by provider key. Selection
    # depends on date/availability only, not the acoustic content.
    indices = [i*(len(available)-1)//59 for i in range(60)]
    chosen = [available[i] for i in indices]
    (PROV / "mbari_selected_objects.json").write_text(json.dumps(chosen,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        mbari = list(pool.map(download_mbari, chosen))
    sources = deep + [x[0] for x in mbari]
    scenes += [x[1] for x in mbari]
    jsonl(MANIFEST / "sources.jsonl", sources)
    jsonl(MANIFEST / "real_scenes.jsonl", scenes)
    scene_hashes = [x["pcm_sha256"] for x in scenes]
    if len(scene_hashes) != len(set(scene_hashes)):
        raise RuntimeError("Exact duplicate scene PCM detected; inspect before accepting benchmark")
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                   scenes=len(scenes), source_records=len(sources),
                   by_split={k:sum(x["split"]==k for x in scenes) for k in ["train","validation","test"]},
                   by_domain={k:sum(x["domain"]==k for x in scenes) for k in ["deepship","mbari"]},
                   groups_by_split={k:len({x["group_id"] for x in scenes if x["split"]==k}) for k in ["train","validation","test"]},
                   exact_duplicate_pcm=0, near_duplicate_gpu_audit="pending",
                   independent_human_annotation_status="not performed",
                   annotation_method="AI references per subsequent user direction; not independently verified ground truth",
                   split_manifest_sha256=sha((MANIFEST / "real_scenes.jsonl").read_bytes()),
                   limitations=["Public DeepShip subset only, no verified vessel IDs or per-file session/day identifiers",
                                "DeepShip audio redistribution terms unresolved",
                                "Naturally overlapping scenes not yet independently identified",
                                "MBARI whole-domain external test; dataset/class confounding must not be interpreted as vessel classification",
                                "MBARI 16 kHz supports 0.5–2 and 2–5 kHz carriers, but not a full 5–8 kHz carrier with transition margins"])
    (MANIFEST / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
