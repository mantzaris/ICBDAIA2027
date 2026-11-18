"""Development-only batched serialization/memory fixture, never acoustic evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
import time
import traceback

from PIL import Image, ImageDraw
from sonar.vlm import LocalVLM
from sonar.batching import BatchedLocalVLM
from sonar.constrained_decoding import action_json_schema, report_json_schema, grammar_provenance
from sonar.controller import report_schema_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-vl-8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--padding-repetitions", type=int, default=800)
    options = parser.parse_args()
    if options.batch_size not in [4,8]: raise ValueError("Fixture batch must be four or eight")
    output = Path(options.output); output.mkdir(parents=True,exist_ok=True)
    if (output/"result.json").exists() or (output/"failure.json").exists():
        raise ValueError("Preserve previous smoke attempts: choose a fresh output directory")
    image = Image.new("RGB",(1024,768),"white")
    ImageDraw.Draw(image).text((20,20),"SERIALIZATION AND MEMORY TEST ONLY. NO ACOUSTIC EVIDENCE.",fill="black")
    image_path = output/"fixture.png"; image.save(image_path)
    started = time.time()
    def report(count):
        return dict(components=[dict(id=f"C{i+1}",type="tone",frequency_domain="acoustic",
            time_s=[10,20],frequency_hz=[100+i,100+i],frequency_track=[[10,100+i],[20,100+i]],
            carrier_band_hz=None,source_identity="unknown",evidence_ids=[]) for i in range(count)],
            relations=[],claims=[],unresolved_questions=[],abstained=count==0)
    expected = [dict(action_index=0,reason="inspect"),report(0),dict(action_index=7,reason="inspect"),
                report(3),report(6),dict(action_index=2,reason="inspect"),report(6),report(0)]
    kinds = ["action","report","action","report","report","action","report","report"]
    caps = [160,2048,160,2048,8,160,2048,2048]
    fixtures = []
    for index in range(options.batch_size):
        padding = ("Memory fixture context with no measurements or factual acoustic content. " *
                   max(0,options.padding_repetitions-index*7))
        prompt = padding + "\nCopy precisely this JSON object and no other content: " + json.dumps(expected[index],separators=(",",":"))
        fixtures.append(dict(prompt=prompt,images=[image_path]*(1+index%3),max_new_tokens=caps[index],
            json_schema=action_json_schema([0,2,7]) if kinds[index]=="action" else report_json_schema()))
    configuration = dict(batch_size=options.batch_size,padding_repetitions=options.padding_repetitions,
                         max_new_tokens=caps[:options.batch_size],grammar=grammar_provenance(),
                         interpretation="Development serialization and memory fixture; not acoustic accuracy or a bound on all future memory use")
    (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    try:
        backend = LocalVLM(options.model)
        barrier = Barrier(options.batch_size)
        with BatchedLocalVLM(backend,max_batch_size=options.batch_size,max_wait_ms=2000) as model:
            def request(fixture):
                barrier.wait()
                return model.generate(**fixture)
            with ThreadPoolExecutor(max_workers=options.batch_size) as pool:
                results = list(pool.map(request,fixtures))
            ledger = list(model.batch_ledger)
        checks=[]
        for index,result in enumerate(results):
            value=result["parsed"]
            if caps[index]==8:
                valid=result["output_tokens"]==8 and value is None
            elif kinds[index]=="action":
                valid=isinstance(value,dict) and type(value.get("action_index")) is int and value["action_index"] in [0,2,7]
            else:
                valid=not report_schema_errors(value)
            checks.append(dict(row=index,valid_structure_or_expected_truncation=valid,
                exact_fixture_agreement=value==expected[index],input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],caps_respected=result["output_tokens"]<=caps[index]))
        passed=(len(ledger)==1 and ledger[0]["batch_size"]==options.batch_size and
                all(c["valid_structure_or_expected_truncation"] and c["caps_respected"] for c in checks))
        record=dict(passed=passed,checks=checks,generations=results,batch_ledger=ledger,
                    elapsed_wall_seconds=time.time()-started,**configuration)
        (output/"result.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(dict(passed=passed,checks=checks,batch_ledger=ledger)),flush=True)
        if not passed: raise RuntimeError("Constrained-decoding development fixture failed")
    except Exception as error:
        (output/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),elapsed_wall_seconds=time.time()-started,**configuration),indent=2)+"\n")
        raise


if __name__ == "__main__": main()
