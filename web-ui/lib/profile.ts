export interface UserProfile {
  version: number
  categories: Record<string, { enabled: boolean; interests: Record<string, number> }>
  general: {
    detail_level: 'short' | 'normal' | 'detailed'
    language_level: 'simple' | 'standard' | 'advanced'
    reading_time: number
    frequency: 'morning' | 'evening' | 'daily' | 'weekly' | 'important_only'
    priority: 'important_only' | 'balanced' | 'everything'
    language: 'ru' | 'en'
    source_reliability: 'verified' | 'balanced' | 'broad'
    regions: string[]
    exclusions: string[]
    personal_context: string
  }
}

export function createEmptyProfile(): UserProfile {
  const categories: UserProfile['categories'] = {}
  return {
    version: 1,
    categories,
    general: {
      detail_level: 'normal',
      language_level: 'standard',
      reading_time: 10,
      frequency: 'daily',
      priority: 'balanced',
      language: 'ru',
      source_reliability: 'balanced',
      regions: [],
      exclusions: [],
      personal_context: '',
    },
  }
}

const STORAGE_KEY = 'news-digest-profile'

const DETAIL_LEVELS = new Set(['short', 'normal', 'detailed'])
const LANGUAGE_LEVELS = new Set(['simple', 'standard', 'advanced'])
const FREQUENCIES = new Set(['morning', 'evening', 'daily', 'weekly', 'important_only'])
const PRIORITIES = new Set(['important_only', 'balanced', 'everything'])
const LANGUAGES = new Set(['ru', 'en'])
const SOURCE_RELIABILITIES = new Set(['verified', 'balanced', 'broad'])

function isValidProfile(data: unknown): data is UserProfile {
  if (typeof data !== 'object' || data === null) return false
  const p = data as Partial<UserProfile> & Record<string, unknown>
  if (p.version !== 1) return false
  if (typeof p.categories !== 'object' || p.categories === null) return false

  for (const cat of Object.values(p.categories as Record<string, unknown>)) {
    if (typeof cat !== 'object' || cat === null) return false
    const c = cat as Record<string, unknown>
    if (typeof c.enabled !== 'boolean') return false
    if (typeof c.interests !== 'object' || c.interests === null) return false
    for (const val of Object.values(c.interests as Record<string, unknown>)) {
      if (typeof val !== 'number' || !Number.isInteger(val) || val < 0 || val > 5) {
        return false
      }
    }
  }

  const g = p.general as Record<string, unknown> | undefined
  if (typeof g !== 'object' || g === null) return false
  if (!DETAIL_LEVELS.has(g.detail_level as string)) return false
  if (!LANGUAGE_LEVELS.has(g.language_level as string)) return false
  if (typeof g.reading_time !== 'number' || !Number.isInteger(g.reading_time)) return false
  if (!FREQUENCIES.has(g.frequency as string)) return false
  if (!PRIORITIES.has(g.priority as string)) return false
  if (!LANGUAGES.has(g.language as string)) return false
  if (!SOURCE_RELIABILITIES.has(g.source_reliability as string)) return false
  if (!Array.isArray(g.regions)) return false
  if (!Array.isArray(g.exclusions)) return false
  if (typeof g.personal_context !== 'string') return false

  return true
}

export function loadProfile(): UserProfile | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const data: unknown = JSON.parse(raw)
    return isValidProfile(data) ? data : null
  } catch {
    return null
  }
}

export function saveProfile(profile: UserProfile): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profile))
}

export function exportProfile(profile: UserProfile): string {
  return JSON.stringify(profile, null, 2)
}

export function importProfile(json: string): UserProfile | null {
  try {
    const data: unknown = JSON.parse(json)
    return isValidProfile(data) ? data : null
  } catch {
    return null
  }
}

export function resetProfile(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(STORAGE_KEY)
}
