<template>
  <div id="app">
    <NuxtPage />
  </div>
</template>

<script setup lang="ts">
const { init, isOnboarded } = useProfile()

onMounted(() => {
  init()
})
</script>

<style>
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

:root {
  --color-bg: #0f0f0f;
  --color-surface: #1a1a2e;
  --color-surface-hover: #222240;
  --color-primary: #6c63ff;
  --color-primary-hover: #5a52d5;
  --color-text: #e0e0e0;
  --color-text-muted: #888;
  --color-border: #2a2a4a;
  --color-success: #4caf50;
  --color-danger: #ef5350;
  --radius: 12px;
  --radius-sm: 8px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body {
  font-family: var(--font);
  background: var(--color-bg);
  color: var(--color-text);
  line-height: 1.6;
  min-height: 100dvh;
}

#app {
  max-width: 600px;
  margin: 0 auto;
  padding: 16px;
  padding-bottom: 100px;
}

button {
  font-family: var(--font);
  cursor: pointer;
  border: none;
  border-radius: var(--radius-sm);
  padding: 12px 24px;
  font-size: 16px;
  font-weight: 600;
  transition: background 0.2s, transform 0.1s;
}

button:active {
  transform: scale(0.98);
}

.btn-primary {
  background: var(--color-primary);
  color: white;
  width: 100%;
}

.btn-primary:hover {
  background: var(--color-primary-hover);
}

.btn-primary:disabled {
  background: var(--color-border);
  color: var(--color-text-muted);
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
}

.btn-secondary:hover {
  background: var(--color-surface-hover);
}

.btn-danger {
  background: var(--color-danger);
  color: white;
}

h1 {
  font-size: 28px;
  margin-bottom: 8px;
}

h2 {
  font-size: 22px;
  margin-bottom: 16px;
}

.subtitle {
  color: var(--color-text-muted);
  margin-bottom: 24px;
}
</style>
