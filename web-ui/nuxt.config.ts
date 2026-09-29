export default defineNuxtConfig({
  ssr: false,
  devtools: { enabled: false },

  app: {
    // Served from the site root; Nuxt still honours NUXT_APP_BASE_URL if set.
    baseURL: '/',
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
