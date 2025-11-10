'''
Copyright 2025 EditMGT Team. All rights reserved.
'''
import os
import sys
sys.path.append("./")
from PIL import Image
from src.editmgt import init_edit_mgt
from src.v2_model import negative_prompt

if __name__ == "__main__":
    pipe = init_edit_mgt(device='cuda:0')
    # Forcing the use of bf16 can improve speed, but it will incur a performance penalty. We noticed that GEditBench dropped by about 0.8.
    # pipe = init_edit_mgt(device, enable_bf16=False)

    # pipe.local_guidance=0.01 # After starting, it will use the local GS auxiliary mode.
    # pipe.local_query_text = 'owl' # Use specific words as attention queries
    # pipe.attention_enable_blocks = [i for i in range(28, 37)]  # attention layer used
    input_image = Image.open('assets/case_5.jspg')
    result = pipe(
        prompt=['Make it into Ghibli style'],
        height=1024,
        width=1024,
        num_inference_steps=36, # For some simple tasks, 16 steps are enough!
        guidance_scale=6,
        reference_strength=1.1,
        reference_image=[input_image.resize((1024, 1024))],
        negative_prompt=negative_prompt or None,
    )
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"edited_case_5.png")
    w, h = input_image.size
    result.images[0].resize((w, h)).save(file_path)
