"""CUDA multimodal compatibility check; no scientific performance claim."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from sonar.vlm import LocalVLM

output = Path('reports/model_smoke')
output.mkdir(parents=True, exist_ok=True)
canvas = Image.new('RGB', (640, 320), 'white')
ImageDraw.Draw(canvas).text((30, 140), 'MODEL COMPATIBILITY CHECK: this is not acoustic evidence.', fill='black')
canvas.save(output / 'input.png')
model = LocalVLM('models/qwen3-vl-8b')
result = model.generate('Return exactly a JSON object with keys image_text and acoustic_evidence_present. Describe only the supplied image.', [output / 'input.png'], 128)
result['device_audit'] = model.device_audit
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
