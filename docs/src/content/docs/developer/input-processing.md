---
title: Input Processing
---

Each model can override parts of Aphrodite's input processing pipeline via `INPUT_REGISTRY` and `MULTIMODAL_REGISTRY`.

Currently, this mechanism is only utilized in multi-modal models for preprocessing multi-modal input data in addition to input prompt, but it can be extended to text-only language models when needed.

## Input Processing Pipeline
1. Input data is passed to [`AphroditeEngine`](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/engine/aphrodite_engine.py) (or [`AsyncAphrodite`](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/engine/async_aphrodite.py) for asynchronous inference).
2. Tokenize the data if necessary.
3. Process the inputs using [`INPUT_REGISTRY.process_input`](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/inputs/registry.py).
4. Send the processed inputs to `ExecutorBase`.
5. Distribute the inputs via `WorkerBase` to `ModelRunnerBase`.
6. If the data contains multi-modal data, convert it into keyword arguments using [`MULTIMODAL_REGISTRY.map_input`](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/multimodal/registry.py).
- For example, convert a `PIL.Image.Image` input to its pixel values for a vision model.