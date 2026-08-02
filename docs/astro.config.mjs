import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import starlightLinksValidator from 'starlight-links-validator';

// https://astro.build/config
export default defineConfig({
	site: "https://sonar.dphn.ai",
	integrations: [
		starlight({
      		plugins: [
				starlightLinksValidator(),
			],
  			favicon: '/favicon.ico',
			customCss: [
				"./src/styles/custom.css"
			],
			head: [
				{
					tag: 'script',
					attrs: {
						src: '/reo.js',
					},
				},
			],
			title: 'Sonar',
			social: {
				github: 'https://github.com/dphnAI/sonar',
			},
			sidebar: [
				{
					label: 'Start',
					items: [
						{ label: 'Choose a path', slug: 'getting-started/choose-a-path' },
						{ label: 'Install Sonar', slug: 'getting-started/installation' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
						{ label: 'Upgrade Sonar', slug: 'getting-started/upgrading' },
					],
				},
				{
					label: 'Use Sonar',
					items: [
						{ label: 'Python LLM API', slug: 'guides/offline-inference' },
						{ label: 'Sampling and structured output', slug: 'guides/sampling-and-structured-output' },
						{ label: 'OpenAI-compatible API', slug: 'serving/openai' },
						{ label: 'Other APIs', slug: 'serving/other-apis' },
					],
				},
				{
					label: 'Deploy',
					items: [
						{ label: 'Optimize a deployment', slug: 'deployment/optimization' },
						{ label: 'Benchmarking', slug: 'deployment/benchmarking' },
						{ label: 'Choose parallelism', slug: 'deployment/parallelism' },
						{ label: 'Distributed deployment', slug: 'deployment/distributed' },
						{ label: 'Production deployment', slug: 'deployment/production' },
						{ label: 'Rust frontend', slug: 'deployment/rust-frontend' },
						{ label: 'Deployment recipes', slug: 'deployment/recipes' },
						{ label: 'Model loading and storage', slug: 'deployment/model-storage' },
						{ label: 'Security', slug: 'deployment/security' },
						{ label: 'Observability', slug: 'features/observability' },
					],
				},
				{
					label: 'Features',
					items: [
						{ label: 'Feature guide', slug: 'features' },
						{ label: 'LoRA adapters', slug: 'features/lora' },
						{ label: 'Prefix caching', slug: 'features/prefix-caching' },
						{ label: 'Speculative decoding', slug: 'features/speculative-decoding' },
						{ label: 'Tool calling', slug: 'features/tool-calling' },
						{ label: 'Reasoning and tool parsers', slug: 'features/reasoning-and-tools' },
						{ label: 'Multimodal inputs', slug: 'features/multimodal' },
						{ label: 'FP8 vision attention', slug: 'features/fp8-vit-attention' },
						{ label: 'Online MXFP6', slug: 'features/mxfp6' },
						{ label: 'Observability', slug: 'features/observability' },
					],
				},
				{
					label: 'Help',
					items: [
						{ label: 'Troubleshooting', slug: 'troubleshooting' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Supported models', slug: 'reference/models' },
						{ label: 'Quantization support', slug: 'reference/quantization' },
						{ label: 'Server arguments', slug: 'reference/server-arguments' },
					],
				},
			],
		}),
	],
});
