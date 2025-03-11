import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import starlightLinksValidator from 'starlight-links-validator';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
      		plugins: [
				starlightLinksValidator(),
			],
  			favicon: '/favicon.ico',
			customCss: [
				"./src/styles/custom.css"
			],
			title: 'Aphrodite Engine',
			social: {
				github: 'https://github.com/aphrodite-engine/aphrodite-engine',
			},
			sidebar: [
				{
					label: 'Installation',
					autogenerate: { directory: 'installation' },

				},
				{
					label: 'Usage',
					autogenerate: { directory: 'usage' },
				},
				{

					label: 'Adapters',
					autogenerate: { directory: 'adapters' },
				},
				{
					label: 'Developer',
					autogenerate: { directory: 'developer' },
				},
				{
					label: 'Prompt Caching',
					autogenerate: { directory: 'prompt-caching' },
				},
				{
					label: 'Quantization',
					autogenerate: { directory: 'quantization' },
				},
				{
					label: 'Spec Decoding',
					autogenerate: { directory: 'spec-decoding' },
				},		
			],
		}),
	],
});
