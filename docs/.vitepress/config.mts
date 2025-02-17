import { defineConfig } from 'vitepress';

export default defineConfig({
  title: "Aphrodite Engine",
  head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
  description: "User and Developer Documentation",
  themeConfig: {
		// nav: [{ text: "Home", link: "/" }],

		sidebar: [
			{
				text: "Installation",
				link: "/pages/installation",
				items: [
					{
						text: "NVIDIA GPU",
						link: "/pages/installation/installation",
					},
					{
						text: "Microsoft Windows",
						link: "/pages/installation/installation-windows",
					},
					{
						text: "AMD GPU",
						link: "/pages/installation/installation-rocm",
					},
					{
						text: "CPU",
						link: "/pages/installation/installation-cpu",
					},
					{
						text: "AWS Trainium1 & Inferentia2",
						link: "/pages/installation/installation-neuron",
					},
					{
						text: "Google TPU",
						link: "/pages/installation/installation-tpu",
					},
					{
						text: "Intel XPU",
						link: "/pages/installation/installation-xpu",
					},
				],
			},
			{
				text: "Usage",
				link: "/pages/usage",
				items: [
					{
						text: "Quick Start",
						link: "/pages/usage/getting-started",
					},
					{
						text: "Debugging Instructions",
						link: "/pages/usage/debugging",
					},
					{
						text: "OpenAI API",
						link: "/pages/usage/openai",
					},
					{
						text: "Vision Language Models",
						link: "/pages/usage/vlm",
					},
					{
						text: "Encoder-Decoder Models",
						link: "/pages/usage/encoder-decoder",
					},
					{
						text: "Distributed Inference",
						link: "/pages/usage/distributed",
					},
					{
						text: "Production Metrics",
						link: "/pages/usage/metrics",
					},
					{
						text: "Supported Models",
						link: "/pages/usage/models",
					},
				]
			},
			{
				text: "Quantization",
				link: "/pages/quantization",
				items: [
					{
						text: "Support Overview",
						link: "/pages/quantization/support-matrix",
					},
					{
						text: "Quantization Methods",
						link: "/pages/quantization/quantization-methods",
					},
					{
						text: "KV Cache Quantization",
						link: "/pages/quantization/kv-cache",
					},
				],
			},
			{
				text: "Prompt Caching",
				link: "/pages/prompt-caching",
				items: [
					{
						text: "Overview",
						link: "/pages/prompt-caching/introduction",
					},
					{
						text: "Implementation",
						link: "/pages/prompt-caching/implementation",
					},

				],
			},
			{
				text: "Speculative Decoding",
				link: "/pages/spec-decoding",
				items: [
					{
						text: "Overview",
						link: "/pages/spec-decoding/overview",
					},
					{
						text: "Draft Model Decoding",
						link: "/pages/spec-decoding/draft-model",
					},
					{
						text: "Ngram Prompt Lookup",
						link: "/pages/spec-decoding/ngram",
					},
					{
						text: "MLPSpeculator",
						link: "/pages/spec-decoding/mlpspeculator",
					},
				],
			},
			{
				text: "Model Adapters",
				link: "/pages/adapters",
				items: [
					{
						text: "LoRA",
						link: "/pages/adapters/lora",
					},
					{
						text: "Soft Prompts",
						link: "/pages/adapters/soft-prompts",
					},
				]
			},
			{
				text: "Developer Documentation",
				link: "/pages/developer",
				items: [
					{
						text: "Adding a New Model",
						link: "/pages/developer/adding-model",
					},
					{
						text: "Adding Multimodal Capabilities",
						link: "/pages/developer/multimodal",
					},
					{
						text: "Input Processing",
						link: "/pages/developer/input-processing",
					},
					{
						text: "Tool Parser Plugin",
						link: "/pages/developer/tool-plugin",
					},
					{
						text: "Paged Attention",
						link: "/pages/developer/paged-attention",
					},
					{
						text: "NVIDIA CUTLASS Epilogues",
						link: "/pages/developer/cutlass-epilogue",
					},
					{
						text: "Adding a Custom Class in PyTorch",
						link: "/pages/developer/torch-custom-class"
					},
				],
			},
		],

		socialLinks: [
			{ icon: "github", link: "https://github.com/PygmalionAI/aphrodite-engine" },
		],

		search: {
			provider: "local",
			options: {
				detailedView: true,
			},
		},
  },
  markdown: {
	math: true
  }
})
