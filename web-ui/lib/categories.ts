export interface CategoryDef {
  id: string
  label: string
  emoji: string
  subtopics: { id: string; label: string }[]
}

export const CATEGORIES: CategoryDef[] = [
  {
    id: 'ai',
    label: 'AI',
    emoji: '🤖',
    subtopics: [
      { id: 'new_models', label: 'Новые модели' },
      { id: 'ai_tools', label: 'AI-инструменты и сервисы' },
      { id: 'research', label: 'Исследования' },
      { id: 'generative_ai', label: 'Генеративный AI' },
      { id: 'ai_business', label: 'AI и бизнес' },
      { id: 'robotics', label: 'Робототехника' },
    ],
  },
  {
    id: 'technology',
    label: 'Технологии',
    emoji: '💻',
    subtopics: [
      { id: 'internet_web', label: 'Интернет и веб' },
      { id: 'mobile', label: 'Мобильные технологии' },
      { id: 'computers_hardware', label: 'Компьютеры и железо' },
      { id: 'cybersecurity', label: 'Кибербезопасность' },
      { id: 'cloud_infrastructure', label: 'Облака и инфраструктура' },
      { id: 'programming', label: 'Программирование' },
    ],
  },
  {
    id: 'science',
    label: 'Наука',
    emoji: '🔬',
    subtopics: [
      { id: 'medicine_biology', label: 'Медицина и биология' },
      { id: 'brain_psychology', label: 'Мозг и психология' },
      { id: 'earth_nature', label: 'Земля и природа' },
      { id: 'physics', label: 'Физика и фундаментальная наука' },
      { id: 'chemistry_materials', label: 'Химия и новые материалы' },
      { id: 'scientific_discoveries', label: 'Научные открытия и исследования' },
    ],
  },
  {
    id: 'space',
    label: 'Космос',
    emoji: '🚀',
    subtopics: [
      { id: 'space_missions', label: 'Космические миссии' },
      { id: 'rockets_launches', label: 'Ракеты и запуски' },
      { id: 'astronomy_observation', label: 'Астрономия и наблюдения' },
      { id: 'planets_objects', label: 'Планеты и космические объекты' },
      { id: 'new_discoveries', label: 'Новые открытия' },
      { id: 'human_spaceflight', label: 'Пилотируемая космонавтика' },
    ],
  },
  {
    id: 'gadgets',
    label: 'Гаджеты',
    emoji: '📱',
    subtopics: [
      { id: 'smartphones', label: 'Смартфоны' },
      { id: 'laptops_tablets', label: 'Ноутбуки и планшеты' },
      { id: 'headphones_audio', label: 'Наушники и аудио' },
      { id: 'smartwatches_wearables', label: 'Умные часы и носимые устройства' },
      { id: 'smart_home', label: 'Умный дом' },
      { id: 'new_devices', label: 'Новые устройства и технологии' },
    ],
  },
  {
    id: 'games',
    label: 'Игры',
    emoji: '🎮',
    subtopics: [
      { id: 'new_games_releases', label: 'Новые игры и релизы' },
      { id: 'gaming_technology_hardware', label: 'Игровые технологии и железо' },
      { id: 'esports', label: 'Киберспорт' },
      { id: 'game_companies', label: 'Новости игровых компаний' },
      { id: 'indie_games', label: 'Инди-игры' },
      { id: 'gaming_trends_industry', label: 'Игровые тренды и индустрия' },
    ],
  },
  {
    id: 'business',
    label: 'Бизнес',
    emoji: '💼',
    subtopics: [
      { id: 'precious_metals', label: 'Драгоценные металлы' },
      { id: 'savings_deposits', label: 'Вклады и сбережения' },
      { id: 'large_companies', label: 'Крупные компании' },
      { id: 'markets_economy', label: 'Рынки и экономика' },
      { id: 'safe_investing', label: 'Безопасное инвестирование' },
      { id: 'entrepreneurs_leaders', label: 'Предприниматели и руководители' },
    ],
  },
  {
    id: 'finance',
    label: 'Финансы',
    emoji: '💰',
    subtopics: [
      { id: 'currencies', label: 'Валюты' },
      { id: 'cryptocurrencies', label: 'Криптовалюты' },
      { id: 'stock_market', label: 'Фондовый рынок' },
      { id: 'banks_rates', label: 'Банки и ставки' },
      { id: 'taxes_financial_rules', label: 'Налоги и финансовые правила' },
      { id: 'personal_finance', label: 'Личные финансы' },
    ],
  },
  {
    id: 'running',
    label: 'Бег и тренировки',
    emoji: '🏃',
    subtopics: [
      { id: 'recovery', label: 'Восстановление после бега' },
      { id: 'motivation_habits', label: 'Мотивация и привычка бегать' },
      { id: 'beginners_running_technique', label: 'Бег для новичков и техника' },
      { id: 'marathon_half_marathon', label: 'Марафон и полумарафон' },
      { id: 'running_gear', label: 'Экипировка бегуна' },
      { id: 'running_nutrition', label: 'Питание и бег' },
    ],
  },
  {
    id: 'movies',
    label: 'Кино и сериалы',
    emoji: '🎬',
    subtopics: [
      { id: 'new_movies', label: 'Новые фильмы' },
      { id: 'new_series', label: 'Новые сериалы' },
      { id: 'trailers_announcements', label: 'Трейлеры и анонсы' },
      { id: 'actors_directors', label: 'Актёры и режиссёры' },
      { id: 'what_to_watch', label: 'Что посмотреть' },
      { id: 'streaming', label: 'Стриминги' },
    ],
  },
  {
    id: 'music',
    label: 'Музыка',
    emoji: '🎵',
    subtopics: [
      { id: 'new_releases', label: 'Новые релизы' },
      { id: 'artists_bands', label: 'Исполнители и группы' },
      { id: 'concerts_festivals', label: 'Концерты и фестивали' },
      { id: 'music_recommendations', label: 'Музыкальные рекомендации' },
      { id: 'awards_charts', label: 'Музыкальные премии и чарты' },
      { id: 'music_technology', label: 'Музыкальные технологии' },
    ],
  },
  {
    id: 'world',
    label: 'Мир',
    emoji: '🌍',
    subtopics: [
      { id: 'world_events', label: 'Главные события в мире' },
      { id: 'usa', label: 'США' },
      { id: 'europe', label: 'Европа' },
      { id: 'asia_other_regions', label: 'Азия и другие регионы' },
      { id: 'global_economy', label: 'Мировая экономика' },
      { id: 'international_relations', label: 'Международные отношения' },
    ],
  },
]
