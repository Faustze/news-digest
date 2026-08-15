<template>
  <div v-if="isOnboarded" class="home">
    <h1>📰 News Digest</h1>
    <p class="subtitle">Профиль настроен</p>
    <NuxtLink to="/profile" class="btn-primary" style="display: block; text-align: center; text-decoration: none;">
      Открыть профиль
    </NuxtLink>
  </div>
  <div v-else-if="step === 'welcome'" class="welcome">
    <h1>👋 Добро пожаловать</h1>
    <p class="subtitle">Настрой свой персональный новостной дайджест</p>
    <p>Ответь на несколько вопросов, и я буду присылать только то, что тебе действительно интересно.</p>
    <br />
    <button class="btn-primary" @click="startCategories">Начать</button>
  </div>
  <div v-else-if="step === 'categories'" class="categories-step">
    <h1>Что тебе интересно?</h1>
    <p class="subtitle">Выбери категории (можно несколько)</p>

    <div class="category-grid">
      <button
        v-for="cat in allCategories"
        :key="cat.id"
        :class="['category-card', { selected: selectedCategories.has(cat.id) }]"
        @click="toggleCategory(cat.id)"
      >
        <span class="cat-emoji">{{ cat.emoji }}</span>
        <span class="cat-label">{{ cat.label }}</span>
      </button>
    </div>

    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'welcome'">Назад</button>
      <button class="btn-primary" :disabled="selectedCategories.size === 0" @click="startSubtopics">
        Продолжить ({{ selectedCategories.size }})
      </button>
    </div>
  </div>
  <div v-else-if="step === 'subtopics'" class="subtopics-step">
    <h1>{{ currentCategoryLabel }}</h1>
    <p class="subtitle">Насколько тебе это интересно?</p>
    <p class="progress">Категория {{ subtopicCategoryIndex + 1 }} из {{ selectedCategoriesList.length }}</p>

    <div class="subtopic-list">
      <div v-for="st in currentSubtopics" :key="st.id" class="subtopic-row">
        <span class="st-label">{{ st.label }}</span>
        <div class="interest-control">
          <button
            v-for="val in [0, 1, 2, 3, 4, 5]"
            :key="val"
            :class="['interest-btn', { active: getInterest(currentCategoryId, st.id) === val }]"
            :title="interestLabels[val]"
            @click="setInterestVal(currentCategoryId, st.id, val)"
          >
            {{ val }}
          </button>
        </div>
      </div>
    </div>

    <div class="step-nav">
      <button class="btn-secondary" @click="prevCategory">Назад</button>
      <button class="btn-primary" @click="nextCategory">
        {{ subtopicCategoryIndex < selectedCategoriesList.length - 1 ? 'Следующая категория' : 'Далее' }}
      </button>
    </div>
  </div>
  <div v-else-if="step === 'detail'" class="detail-step">
    <h1>Насколько подробно рассказывать?</h1>
    <div class="option-list">
      <button
        v-for="opt in detailOptions"
        :key="opt.value"
        :class="['option-card', { active: profile?.general.detail_level === opt.value }]"
        @click="setGeneralVal('detail_level', opt.value)"
      >
        {{ opt.label }}
      </button>
    </div>
    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'subtopics'">Назад</button>
      <button class="btn-primary" @click="step = 'lang_level'">Далее</button>
    </div>
  </div>
  <div v-else-if="step === 'lang_level'" class="detail-step">
    <h1>Сложность языка</h1>
    <div class="option-list">
      <button
        v-for="opt in langLevelOptions"
        :key="opt.value"
        :class="['option-card', { active: profile?.general.language_level === opt.value }]"
        @click="setGeneralVal('language_level', opt.value)"
      >
        {{ opt.label }}
      </button>
    </div>
    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'detail'">Назад</button>
      <button class="btn-primary" @click="step = 'time'">Далее</button>
    </div>
  </div>
  <div v-else-if="step === 'time'" class="detail-step">
    <h1>Сколько времени ты хочешь тратить на новости?</h1>
    <div class="option-list">
      <button
        v-for="opt in timeOptions"
        :key="opt.value"
        :class="['option-card', { active: profile?.general.reading_time === opt.value }]"
        @click="setGeneralVal('reading_time', opt.value)"
      >
        {{ opt.label }}
      </button>
    </div>
    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'lang_level'">Назад</button>
      <button class="btn-primary" @click="step = 'exclusions'">Далее</button>
    </div>
  </div>
  <div v-else-if="step === 'exclusions'" class="detail-step">
    <h1>Что тебе точно не хочется видеть?</h1>
    <p class="subtitle">Можно пропустить</p>
    <div class="option-list multi">
      <button
        v-for="opt in exclusionOptions"
        :key="opt.value"
        :class="['option-card', { active: profile?.general.exclusions.includes(opt.value) }]"
        @click="toggleExcl(opt.value)"
      >
        {{ opt.label }}
      </button>
    </div>
    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'time'">Назад</button>
      <button class="btn-primary" @click="step = 'context'">Далее</button>
    </div>
  </div>
  <div v-else-if="step === 'context'" class="detail-step">
    <h1>Есть что-то ещё?</h1>
    <p class="subtitle">Необязательное поле</p>
    <textarea
      class="context-input"
      placeholder="Например: «Готовлюсь к полумарафону, интересуюсь AI и люблю научную фантастику»"
      :value="profile?.general.personal_context"
      @input="setGeneralVal('personal_context', ($event.target as HTMLTextAreaElement).value)"
    ></textarea>
    <div class="step-nav">
      <button class="btn-secondary" @click="step = 'exclusions'">Назад</button>
      <button class="btn-primary" @click="finish">Готово!</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { CATEGORIES } from '~/lib/categories'

const {
  profile,
  isOnboarded,
  startOnboarding,
  enableCategory,
  disableCategory,
  setInterest,
  setGeneral,
  toggleExclusion,
  finishOnboarding,
} = useProfile()

type Step = 'welcome' | 'categories' | 'subtopics' | 'detail' | 'lang_level' | 'time' | 'exclusions' | 'context'

const step = ref<Step>('welcome')
const selectedCategories = ref<Set<string>>(new Set())
const subtopicCategoryIndex = ref(0)

const allCategories = CATEGORIES
const selectedCategoriesList = computed(() =>
  Array.from(selectedCategories.value).filter(id => CATEGORIES.some(c => c.id === id))
)

const currentCategoryId = computed(() => selectedCategoriesList.value[subtopicCategoryIndex.value] || '')
const currentCategoryLabel = computed(() => {
  const cat = CATEGORIES.find(c => c.id === currentCategoryId.value)
  return cat ? `${cat.emoji} Что тебе интересно в «${cat.label}»?` : ''
})
const currentSubtopics = computed(() => {
  const cat = CATEGORIES.find(c => c.id === currentCategoryId.value)
  return cat?.subtopics || []
})

const interestLabels: Record<number, string> = {
  0: 'Не показывать',
  1: 'Почти неинтересно',
  2: 'Иногда интересно',
  3: 'Интересно',
  4: 'Очень интересно',
  5: 'Высокий приоритет',
}

const detailOptions = [
  { value: 'short' as const, label: 'Коротко — только самое важное' },
  { value: 'normal' as const, label: 'Нормально — с контекстом' },
  { value: 'detailed' as const, label: 'Подробно — со всеми деталями' },
]

const langLevelOptions = [
  { value: 'simple' as const, label: 'Простыми словами' },
  { value: 'standard' as const, label: 'Обычный уровень' },
  { value: 'advanced' as const, label: 'Продвинутый — с терминами' },
]

const timeOptions = [
  { value: 5, label: '5 минут' },
  { value: 10, label: '10 минут' },
  { value: 20, label: '20 минут' },
  { value: 30, label: '30+ минут' },
]

const exclusionOptions = [
  { value: 'politics', label: 'Политика' },
  { value: 'clickbait', label: 'Кликбейт и сенсации' },
  { value: 'advertising', label: 'Реклама и промо' },
  { value: 'repeats', label: 'Повторы одной новости' },
  { value: 'old_news', label: 'Старые новости' },
  { value: 'unverified', label: 'Непроверенная информация' },
  { value: 'negative', label: 'Слишком негативные новости' },
]

function startCategories() {
  startOnboarding()
  step.value = 'categories'
}

function toggleCategory(catId: string) {
  if (selectedCategories.value.has(catId)) {
    selectedCategories.value.delete(catId)
    disableCategory(catId)
  } else {
    selectedCategories.value.add(catId)
    enableCategory(catId)
  }
}

function startSubtopics() {
  subtopicCategoryIndex.value = 0
  step.value = 'subtopics'
}

function prevCategory() {
  if (subtopicCategoryIndex.value > 0) {
    subtopicCategoryIndex.value--
  } else {
    step.value = 'categories'
  }
}

function nextCategory() {
  if (subtopicCategoryIndex.value < selectedCategoriesList.value.length - 1) {
    subtopicCategoryIndex.value++
  } else {
    step.value = 'detail'
  }
}

function getInterest(catId: string, stId: string): number {
  return profile.value?.categories[catId]?.interests[stId] ?? 3
}

function setInterestVal(catId: string, stId: string, val: number) {
  setInterest(catId, stId, val)
}

function setGeneralVal(key: string, value: any) {
  setGeneral(key as any, value)
}

function toggleExcl(val: string) {
  toggleExclusion(val)
}

function finish() {
  finishOnboarding()
}
</script>

<style scoped>
.home, .welcome, .categories-step, .subtopics-step, .detail-step {
  text-align: center;
  padding-top: 40px;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 24px;
}

.category-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px 8px;
  background: var(--color-surface);
  border: 2px solid var(--color-border);
  border-radius: var(--radius);
  color: var(--color-text);
  font-size: 13px;
}

.category-card.selected {
  border-color: var(--color-primary);
  background: var(--color-surface-hover);
}

.cat-emoji {
  font-size: 28px;
}

.cat-label {
  font-weight: 500;
}

.progress {
  color: var(--color-text-muted);
  font-size: 14px;
  margin-bottom: 20px;
}

.subtopic-list {
  text-align: left;
  margin-bottom: 24px;
}

.subtopic-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px solid var(--color-border);
}

.st-label {
  font-size: 15px;
}

.interest-control {
  display: flex;
  gap: 4px;
}

.interest-btn {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--color-surface);
  border: 2px solid var(--color-border);
  color: var(--color-text);
  font-size: 14px;
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

.interest-btn:hover:not(.active) {
  border-color: var(--color-primary);
}

.option-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
  text-align: left;
}

.option-card {
  padding: 16px;
  background: var(--color-surface);
  border: 2px solid var(--color-border);
  border-radius: var(--radius);
  color: var(--color-text);
  font-size: 16px;
  text-align: left;
  width: 100%;
}

.option-card.active {
  border-color: var(--color-primary);
  background: var(--color-surface-hover);
}

.option-card:hover:not(.active) {
  border-color: var(--color-primary);
}

.context-input {
  width: 100%;
  min-height: 100px;
  padding: 12px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
  font-family: var(--font);
  font-size: 15px;
  resize: vertical;
  margin-bottom: 24px;
}

.context-input:focus {
  outline: none;
  border-color: var(--color-primary);
}

.step-nav {
  display: flex;
  gap: 12px;
  justify-content: space-between;
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 16px;
  background: var(--color-bg);
  border-top: 1px solid var(--color-border);
  max-width: 600px;
  margin: 0 auto;
}

.step-nav button {
  flex: 1;
}

@media (max-width: 480px) {
  .category-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
