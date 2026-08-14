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

export function loadProfile(): UserProfile | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as UserProfile
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
    const data = JSON.parse(json)
    if (data.version !== 1 || !data.categories || !data.general) return null
    return data as UserProfile
  } catch {
    return null
  }
}

export function resetProfile(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(STORAGE_KEY)
}
