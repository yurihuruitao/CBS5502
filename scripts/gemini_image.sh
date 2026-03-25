#!/usr/bin/env bash
# Gemini Image Generation Skill
# Usage: ./scripts/gemini_image.sh "<prompt>" <output_path>
#
# Model: gemini-3-pro-image-preview
# API Key: set via GEMINI_API_KEY env or hardcoded below

PROMPT="$1"
OUTPUT="${2:-output.png}"
API_KEY="${GEMINI_API_KEY:-AIzaSyBISNGaR7UybHBSbLKqqOL7KreoLfSllOU}"
MODEL="gemini-3-pro-image-preview"

python3 -c "
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import sys

client = genai.Client(api_key='${API_KEY}')
response = client.models.generate_content(
    model='${MODEL}',
    contents='''${PROMPT}''',
    config=types.GenerateContentConfig(
        response_modalities=['IMAGE', 'TEXT'],
    ),
)
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        img = Image.open(BytesIO(part.inline_data.data))
        if img.width < 3840:
            r = 3840 / img.width
            img = img.resize((3840, int(img.height * r)), Image.LANCZOS)
        img.save('${OUTPUT}', 'PNG')
        print(f'Saved: ${OUTPUT} ({img.width}x{img.height})')
        sys.exit(0)
print('ERROR: No image in response')
for part in response.candidates[0].content.parts:
    if part.text:
        print(part.text[:300])
sys.exit(1)
"
