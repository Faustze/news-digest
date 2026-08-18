export default defineNuxtConfig({
  ssr: false,
  devtools: { enabled: false },

  app: {
    // Set by the deploy workflow (NUXT_APP_BASE_URL=/<repo>/); locally the
    // site is served from the root so '/' is the right fallback.
    baseURL: process.env.NUXT_APP_BASE_URL || '/',
    head: {
      title: 'News Digest — Настройка',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Настрой персональный новостной дайджест' },
      ],
    },
  },

  compatibilityDate: '2025-01-01',
})
