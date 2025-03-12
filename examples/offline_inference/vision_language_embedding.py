import os

from PIL import Image

from aphrodite import LLM

# Input image and question
image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "../vision/burg.jpg")
image = Image.open(image_path).convert("RGB")
prompt = "<|image_1|> Represent the given image with the following question: What is in the image" # noqa: E501


# Create an LLM.
llm = LLM(
    model="TIGER-Lab/VLM2Vec-Full",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    mm_processor_kwargs={"num_crops": 16},
)

# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = llm.encode({"prompt": prompt, "multi_modal_data": {"image": image}})

# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 3072 floats
