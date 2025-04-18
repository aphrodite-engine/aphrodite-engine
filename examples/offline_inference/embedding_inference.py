import os
import time

from aphrodite import LLM

# Sample prompts.
prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox jumps over the lazy dog.",
    "The meaning of life is",
] * 10000

# needed for BERT models
os.environ["APHRODITE_ATTENTION_BACKEND"] = "XFORMERS"

# Create an LLM.
model = LLM(model="BAAI/bge-base-en-v1.5", enforce_eager=True)
start_time = time.time()
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = model.encode(prompts)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Print the outputs.
# for output in outputs:
#     print(output.outputs.embedding)  # list of 4096 floats
