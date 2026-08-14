<template>
  <div class="profile-page">
    <NuxtLink to="/" class="back-link">← На главную</NuxtLink>
    <h1>⚙️ Мой профиль</h1>

    <div v-if="!profile" class="empty-state">
      <p>Профиль не найден</p>
      <NuxtLink to="/" class="btn-primary" style="display: block; text-align: center; text-decoration: none;">
        Пройти onboarding
      </NuxtLink>
    </div>

    <template v-else>
      <!-- Categories -->
      <section class="section">
        <h2>Мои интересы</h2>
        <div class="cat-list">
          <div v-for="cat in allCategories" :key="cat.id" class="cat-card" @click="toggleCatExpand(cat.id)">
            <div class="cat-header">
              <span class="cat-emoji">{{ cat.emoji }}</span>
              <span class="cat-name">{{ cat.label }}</span>
              <label class="toggle" @click.stop>
                <input type="checkbox" :checked="profile.categories[cat.id]?.enabled" @change="toggleCat(cat.id)" />
                <span class="toggle-slider"></span>
              </label>
            </div>
            <div v-if="expandedCat === cat.id && profile.categories[cat.id]?.enabled" class="cat-details">
              <div v-for="st in cat.subtopics" :key="st.id" class="subtopic-row">
                <span class="st-label">{{ st.label }}</span>
                <div class="interest-control">
                  <button
                    v-for="val in [0, 1, 2, 3, 4, 5]"
                    :key="val"
                    :class="['interest-btn', { active: getInterest(cat.id, st.id) === val }]"
                    @click="setInt(cat.id, st.id, val)"
                  >
                    {{ val }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- General settings -->
      <section class="section">
        <h2>Общие настройки</h2>

        <div class="setting-row">
          <label>Подробность</label>
          <select :value="profile.general.detail_level" @change="setGen('detail_level', ($event.target as HTMLSelectElement).value)">
            <option value="short">Коротко</option>
            <option value="normal">Нормально</option>
            <option value="detailed">Подробно</option>
          </select>
        </div>

        <div class="setting-row">
          <label>Язык сложности</label>
          <select :value="profile.general.language_level" @change="setGen('language_level', ($event.target as HTMLSelectElement).value)">
            <option value="simple">Простыми словами</option>
            <option value="standard">Обычный уровень</option>
            <option value="advanced">Продвинутый</option>
          </select>
        </div>

        <div class="setting-row">
          <label>Время на новости</label>
          <select :value="profile.general.reading_time" @change="setGen('reading_time', Number(($event.target as HTMLSelectElement).value))">
            <option :value="5">5 минут</option>
            <option :value="10">10 минут</option>
            <option :value="20">20 минут</option>
            <option :value="30">30+ минут</option>
          </select>
        </div>

        <div class="setting-row">
          <label>Язык дайджеста</label>
          <select :value="profile.general.language" @change="setGen('language', ($event.target as HTMLSelectElement).value)">
            <option value="ru">Русский</option>
            <option value="en">English</option>
          </select>
        </div>

        <div class="setting-row">
          <label>Приоритет</label>
          <select :value="profile.general.priority" @change="setGen('priority', ($event.target as HTMLSelectElement).value)">
            <option value="important_only">Только самое важное</option>
            <option value="balanced">Баланс</option>
            <option value="everything">Быть в курсе всего</option>
          </select>
        </div>

        <div class="setting-row">
          <label>Надёжность источников</label>
          <select :value="profile.general.source_reliability" @change="setGen('source_reliability', ($event.target as HTMLSelectElement).value)">
            <option value="verified">Только проверенные</option>
            <option value="balanced">Баланс</option>
            <option value="broad">Как можно шире</option>
          </select>
        </div>
      </section>

      <!-- Regions -->
      <section class="section">
        <h2>Регионы</h2>
        <div class="chips">
          <button
            v-for="r in regionOptions"
            :key="r.value"
            :class="['chip', { active: profile.general.regions.includes(r.value) }]"
            @click="toggleReg(r.value)"
          >
            {{ r.label }}
          </button>
        </div>
      </section>

      <!-- Exclusions -->
      <section class="section">
        <h2>Исключения</h2>
        <div class="chips">
          <button
            v-for="e in exclusionOptions"
            :key="e.value"
            :class="['chip', { active: profile.general.exclusions.includes(e.value) }]"
            @click="toggleExcl(e.value)"
          >
            {{ e.label }}
          </button>
        </div>
      </section>

      <!-- Personal context -->
      <section class="section">
        <h2>Личный контекст</h2>
        <textarea
          class="context-input"
          placeholder="Например: «Готовлюсь к полумарафону...»"
          :value="profile.general.personal_context"
          @input="setGen('personal_context', ($event.target as HTMLTextAreaElement).value)"
        ></textarea>
      </section>

      <!-- Actions -->
      <section class="section actions">
        <button class="btn-primary" @click="doSave">Сохранить</button>
        <div class="action-row">
          <button class="btn-secondary" @click="doExport">Экспорт JSON</button>
          <button class="btn-secondary" @click="triggerImport">Импорт JSON</button>
        </div>
        <input ref="fileInput" type="file" accept=".json" style="display: none" @change="doImport" />
        <button class="btn-danger" @click="doReset">Сбросить профиль</button>
      </section>
    </template>
  </div>
</template>

<script setup lang="ts">
import { CATEGORIES } from '~/lib/categories'

const { profile, save, reset, exportJson, importJson, setGeneral, setInterest, toggleRegion, toggleExclusion } = useProfile()
const router = useRouter()
const fileInput = ref<HTMLInputElement>()

const expandedCat = ref<string | null>(null)
const allCategories = CATEGORIES

const regionOptions = [
  { value: 'russia', label: 'Россия' },
  { value: 'europe', label: 'Европа' },
  { value: 'usa', label: 'США' },
  { value: 'asia', label: 'Азия' },
  { value: 'other', label: 'Другие страны' },
  { value: 'global', label: 'Весь мир' },
]

const exclusionOptions = [
  { value: 'politics', label: 'Политика' },
  { value: 'clickbait', label: 'Кликбейт' },
  { value: 'advertising', label: 'Реклама' },
  { value: 'repeats', label: 'Повторы' },
  { value: 'old_news', label: 'Старые новости' },
  { value: 'unverified', label: 'Непроверенная инфа' },
  { value: 'negative', label: 'Негатив' },
]

function toggleCatExpand(catId: string) {
  expandedCat.value = expandedCat.value === catId ? null : catId
}

function toggleCat(catId: string) {
  if (!profile.value) return
  const cat = profile.value.categories[catId]
  if (cat) {
    cat.enabled = !cat.enabled
  } else {
    const catDef = CATEGORIES.find(c => c.id === catId)
    profile.value.categories[catId] = {
      enabled: true,
      interests: catDef ? Object.fromEntries(catDef.subtopics.map(st => [st.id, 3])) : {},
    }
  }
}

function getInterest(catId: string, stId: string): number {
  return profile.value?.categories[catId]?.interests[stId] ?? 3
}

function setInt(catId: string, stId: string, val: number) {
  setInterest(catId, stId, val)
}

function setGen(key: string, value: any) {
  setGeneral(key as any, value)
}

function toggleReg(region: string) {
  toggleRegion(region)
}

function toggleExcl(exclusion: string) {
  toggleExclusion(exclusion)
}

function doSave() {
  save()
  alert('Профиль сохранён!')
}

function doExport() {
  const json = exportJson()
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'user-profile.json'
  a.click()
  URL.revokeObjectURL(url)
}

function triggerImport() {
  fileInput.value?.click()
}

function doImport(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = () => {
    const json = reader.result as string
    if (importJson(json)) {
      alert('Профиль импортирован!')
    } else {
      alert('Ошибка: неверный формат файла')
    }
  }
  reader.readAsText(file)
  input.value = ''
}

function doReset() {
  if (confirm('Ты точно хочешь сбросить профиль?')) {
    reset()
    router.push('/')
  }
}
</script>

<style scoped>
.profile-page {
  padding-top: 20px;
}

.back-link {
  color: var(--color-primary);
  text-decoration: none;
  font-size: 14px;
  display: inline-block;
  margin-bottom: 16px;
}

.section {
  margin-bottom: 32px;
}

.section h2 {
  font-size: 18px;
  margin-bottom: 12px;
  color: var(--color-text-muted);
}

.cat-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cat-card {
  background: var(--color-surface);
  border-radius: var(--radius);
  overflow: hidden;
  cursor: pointer;
}

.cat-card:hover {
  background: var(--color-surface-hover);
}

.cat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
}

.cat-emoji {
  font-size: 22px;
}

.cat-name {
  flex: 1;
  font-weight: 500;
}

.toggle {
  position: relative;
  width: 44px;
  height: 24px;
}

.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  inset: 0;
  background: var(--color-border);
  border-radius: 12px;
  transition: background 0.2s;
  cursor: pointer;
}

.toggle-slider::before {
  content: '';
  position: absolute;
  width: 18px;
  height: 18px;
  left: 3px;
  top: 3px;
  background: white;
  border-radius: 50%;
  transition: transform 0.2s;
}

.toggle input:checked + .toggle-slider {
  background: var(--color-primary);
}

.toggle input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

.cat-details {
  padding: 0 16px 16px;
}

.subtopic-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-top: 1px solid var(--color-border);
}

.st-label {
  font-size: 14px;
}

.interest-control {
  display: flex;
  gap: 3px;
}

.interest-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--color-bg);
  border: 2px solid var(--color-border);
  color: var(--color-text);
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.interest-btn.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px solid var(--color-border);
}

.setting-row label {
  font-size: 15px;
}

.setting-row select {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 8px 12px;
  font-size: 14px;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.chip {
  padding: 8px 16px;
  border-radius: 20px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-text);
  font-size: 14px;
}

.chip.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.context-input {
  width: 100%;
  min-height: 80px;
  padding: 12px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
  font-family: var(--font);
  font-size: 14px;
  resize: vertical;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-row {
  display: flex;
  gap: 12px;
}

.action-row button {
  flex: 1;
}
</style>
