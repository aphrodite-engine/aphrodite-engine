---
title: Speculative Decoding
---

Speculative decoding is a technique to speed up inference of large language models by using a smaller model. The idea is to use a smaller model to generate a set of candidate tokens, and then use a larger model to score these candidates. Please see this [great X post by Andrej Karpathy](https://x.com/karpathy/status/1697318534555336961) on the subject.

In short, Speculative decoding is an optimization technique for inference that makes educated guesses about future tokens while generating the current token, all within a single forward pass. It incorporates a verification mechanism to ensure the correctness of these speculated tokens, thereby guaranteeing that the overall output of speculative decoding is identical to that of vanilla decoding. Optimizing the cost of inference of large language models (LLMs) is arguably one of the most critical factors in reducing the cost of generative AI and increasing its adoption. Towards this goal, various inference optimization techniques are available, including custom kernels, dynamic batching of input requests, and quantization of large models. Aphrodite implements many of these techniques, which you will find in the next pages.



Reference:

- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

The next sections will explain each method and how to use them with Aphrodite.

