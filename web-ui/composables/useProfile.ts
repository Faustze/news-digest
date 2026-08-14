import { ref, computed } from 'vue'
import {
  type UserProfile,
  createEmptyProfile,
  loadProfile as loadFromStorage,
  saveProfile as saveToStorage,
  exportProfile as exportToJson,
  importProfile as importFromJson,
  resetProfile as resetStorage,
} from '~/lib/profile'
import { CATEGORIES } from '~/lib/categories'

const profile = ref<UserProfile | null>(null)
const isOnboarded = ref(false)

export function useProfile() {
  function load() {
    const loaded = loadFromStorage()
    if (loaded) {
      profile.value = loaded
      isOnboarded.value = true
    }
  }

  function init() {
    load()
  }

  function startOnboarding() {
    const p = createEmptyProfile()
    // Pre-enable all categories with default interests
    for (const cat of CATEGORIES) {
      p.categories[cat.id] = {
        enabled: false,
        interests: Object.fromEntries(cat.subtopics.map(st => [st.id, 3])),
      }
    }
    profile.value = p
    isOnboarded.value = false
  }

  function enableCategory(catId: string) {
    if (!profile.value) return
    if (!profile.value.categories[catId]) {
      const cat = CATEGORIES.find(c => c.id === catId)
      profile.value.categories[catId] = {
        enabled: true,
        interests: cat ? Object.fromEntries(cat.subtopics.map(st => [st.id, 3])) : {},
      }
    } else {
      profile.value.categories[catId].enabled = true
    }
  }

  function disableCategory(catId: string) {
    if (!profile.value?.categories[catId]) return
    profile.value.categories[catId].enabled = false
  }

  function setInterest(catId: string, subtopicId: string, value: number) {
    if (!profile.value?.categories[catId]) return
    profile.value.categories[catId].interests[subtopicId] = value
  }

  function setGeneral<K extends keyof UserProfile['general']>(key: K, value: UserProfile['general'][K]) {
    if (!profile.value) return
    ;(profile.value.general as any)[key] = value
  }

  function toggleRegion(region: string) {
    if (!profile.value) return
    const idx = profile.value.general.regions.indexOf(region)
    if (idx >= 0) {
      profile.value.general.regions.splice(idx, 1)
    } else {
      profile.value.general.regions.push(region)
    }
  }

  function toggleExclusion(exclusion: string) {
    if (!profile.value) return
    const idx = profile.value.general.exclusions.indexOf(exclusion)
    if (idx >= 0) {
      profile.value.general.exclusions.splice(idx, 1)
    } else {
      profile.value.general.exclusions.push(exclusion)
    }
  }

  function finishOnboarding() {
    if (!profile.value) return
    saveToStorage(profile.value)
    isOnboarded.value = true
  }

  function save() {
    if (!profile.value) return
    saveToStorage(profile.value)
  }

  function reset() {
    resetStorage()
    profile.value = null
    isOnboarded.value = false
  }

  function exportJson(): string {
    if (!profile.value) return ''
    return exportToJson(profile.value)
  }

  function importJson(json: string): boolean {
    const imported = importFromJson(json)
    if (imported) {
      profile.value = imported
      saveToStorage(imported)
      isOnboarded.value = true
      return true
    }
    return false
  }

  const enabledCategories = computed(() => {
    if (!profile.value) return []
    return CATEGORIES.filter(c => profile.value!.categories[c.id]?.enabled)
  })

  return {
    profile,
    isOnboarded,
    init,
    load,
    startOnboarding,
    enableCategory,
    disableCategory,
    setInterest,
    setGeneral,
    toggleRegion,
    toggleExclusion,
    finishOnboarding,
    save,
    reset,
    exportJson,
    importJson,
    enabledCategories,
  }
}
