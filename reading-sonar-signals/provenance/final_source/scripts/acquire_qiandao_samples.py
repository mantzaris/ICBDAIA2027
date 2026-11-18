#!/usr/bin/env python3
"""Download a bounded short-window audit from the author's public anonymous share.

Signed anonymous download credentials remain in process memory and are never saved.
These three-second clips are not admitted to the continuous-scene benchmark.
"""
import concurrent.futures
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SHARE = "https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/duxiaoyang22_mails_ucas_ac_cn/EomiGNu7mO5FmUke62y6Q7IBIP64kpJrJMJOZp_c-qkFAA?e=x8gxuL"
FOLDERS = {"KaiYuan": "01VJOHBVGWMUS6VG3URBB32YHGNJT5N6Z7",
           "SpeedBoat": "01VJOHBVDOPUVHRSOTBNB3Z34N3C75W3NL",
           "UUV": "01VJOHBVFRC64C4ORERFAYMTACCGXTJ7RR"}


def main():
    page = requests.get(SHARE, timeout=60)
    page.raise_for_status()
    context = json.loads(re.search(r"var _spPageContextInfo=(.*?);",page.text).group(1))
    drive = context["driveInfo"]
    selected = []
    for source_class, folder in FOLDERS.items():
        url = drive[".driveUrl"] + "/items/" + folder + "/children?$top=1000&" + drive[".driveAccessToken"]
        parents = set()
        for _ in range(8):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            result = response.json()
            for item in result.get("value", []):
                if not item["name"].lower().endswith(".wav"):
                    continue
                parent = re.split(r"_\d+_label_", item["name"])[0]
                if parent in parents:
                    continue
                parents.add(parent)
                selected.append((source_class, parent, item))
                if len(parents) >= 4:
                    break
            if len(parents)>=4 or not result.get("@odata.nextLink"):
                break
            url = result["@odata.nextLink"]
    rows=[]
    destination=ROOT / "data/raw/qiandao_short"
    destination.mkdir(parents=True,exist_ok=True)
    for source_class,parent,item in selected:
        url=item.get("@content.downloadUrl",item.get("@microsoft.graph.downloadUrl"))
        if not url:
            raise RuntimeError("No anonymous download route returned")
        response=requests.get(url,timeout=60)
        response.raise_for_status()
        body=response.content
        if len(body)!=item["size"]:
            raise ValueError("Qiandao file length mismatch")
        digest=hashlib.sha256(body).hexdigest()
        opaque="qd_"+digest[:16]
        path=destination/(opaque+".wav")
        path.write_bytes(body)
        info=sf.info(io.BytesIO(body))
        samples,rate=sf.read(io.BytesIO(body),dtype="int32",always_2d=True)
        if len(samples)!=info.frames:
            raise ValueError("Qiandao incomplete decode")
        rows.append(dict(scene_id=opaque,stratum="short_window_access_audit",domain="qiandaoear22",
                         split="audit_only_not_benchmark",group_id="qiandao:day:"+parent[:8],
                         source_id=item["id"],original_filename=item["name"],parent_name=parent,
                         source_class=source_class,path=str(path.relative_to(ROOT)),
                         source_url=SHARE,provider_url="https://github.com/xiaoyangdu22/QiandaoEar22",
                         bytes=len(body),sha256=digest,pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                         sample_rate=rate,channels=info.channels,frames=info.frames,
                         subtype=info.subtype,duration_seconds=info.duration,
                         label_status="provider clip-level target metadata only; independent morphology unavailable",
                         source_start_seconds=None,source_start_status="Segment index not verified as exact original sample offset",
                         continuity_reconstruction="not performed; unrelated or unverifiably contiguous clips are never concatenated",
                         license="No explicit audio redistribution license found in author repository/share",
                         acquired_at_utc=datetime.now(timezone.utc).isoformat()))
        print(opaque,info.duration,"seconds",rate,"Hz",flush=True)
    out=ROOT/"data/manifests/qiandao_short_windows.jsonl"
    out.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
    print("Saved",len(rows),"short-window audit files",flush=True)


if __name__=="__main__":
    main()
