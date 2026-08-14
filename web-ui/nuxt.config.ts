export default defineNuxtConfig({
  ssr: false,
  devtools: { enabled: false },

  app: {
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
