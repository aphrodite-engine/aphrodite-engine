# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys


def test_diffusion_prompt_preserves_structured_input():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from aphrodite.omni.outputs import OmniRequestOutput

prompt = {"prompt": "a red fox", "negative_prompt": "blur"}
output = OmniRequestOutput.from_diffusion("image-1", [], prompt=prompt)
assert output.prompt == "a red fox"
assert output.diffusion_prompt is prompt
assert output.to_dict()["prompt"] is prompt

token_prompt = {"prompt_token_ids": [1, 2, 3]}
output = OmniRequestOutput.from_diffusion("image-2", [], prompt=token_prompt)
assert output.prompt is None
assert output.diffusion_prompt is token_prompt
assert output.to_dict()["prompt"] is token_prompt
""",
        ],
        check=True,
    )
