/**
 * Application Constants
 * Centralized configuration for the Gesture Platform UI
 */

// Alphabet and common symbols
export const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
export const NUMBERS = '0123456789'.split('')
export const ALL_SYMBOLS = [...ALPHABET, ...NUMBERS]

// UI Navigation Modes
export const MODES = {
  MENU: 'menu',
  PRACTICE: 'practice',
  LIVE_CAPTION: 'live-caption',
  SETTINGS: 'settings',
  CALIBRATION: 'calibration',
  DASHBOARD: 'dashboard'
}

// Feature cards for home screen
export const FEATURE_CARDS = [
  {
    key: 'practice',
    eyebrow: 'Training',
    title: 'Practice letters with calmer feedback loops',
    body: 'A steadier prompt flow, progress tracking, and session timing make repetition feel more intentional.',
    accent: 'from-emerald-500/35 via-emerald-300/10 to-transparent'
  },
  {
    key: 'live-caption',
    eyebrow: 'Transcription',
    title: 'Build phrases from stable predictions',
    body: 'Captioning now favors repeated detections and gives you simple editing controls for the sentence builder.',
    accent: 'from-sky-500/35 via-sky-300/10 to-transparent'
  },
  {
    key: 'calibration',
    eyebrow: 'Setup',
    title: 'Prepare your camera view before a session',
    body: 'The desktop app now gives a clearer setup flow while the Python recognizer remains available from the scripts.',
    accent: 'from-amber-400/35 via-amber-300/10 to-transparent'
  },
  {
    key: 'settings',
    eyebrow: 'Control',
    title: 'Tune confidence, theme, and preview overlays',
    body: 'Settings are grouped more clearly so you can adjust the desktop experience without hunting through the UI.',
    accent: 'from-slate-300/30 via-slate-100/10 to-transparent'
  }
]

// Common words for live caption suggestions
export const COMMON_WORDS = [
  'hello', 'help', 'home', 'have', 'here',
  'thank', 'thanks', 'the', 'this', 'that', 'they',
  'yes', 'you', 'your', 'no', 'not', 'name',
  'please', 'play', 'good', 'go', 'great',
  'bye', 'be', 'but', 'i', 'is', 'it', 'in',
  'want', 'what', 'where', 'when', 'who', 'why', 'with', 'will', 'was',
  'can', 'come', 'call', 'do', 'done', 'dont',
  'are', 'am', 'and', 'all', 'a', 'more', 'me', 'my', 'meet',
  'see', 'sorry', 'some', 'so', 'she', 'ready', 'right',
  'fine', 'for', 'from', 'friend', 'like', 'love', 'learn', 'later',
  'ok', 'of', 'need', 'understand', 'us', 'very', 'know', 'just'
]

// Camera options
export const CAMERA_CHOICES = [0, 1, 2, 3]

// Settings defaults
export const DEFAULT_SETTINGS = {
  theme: 'dark',
  cameraIndex: 0,
  confidenceThreshold: 0.7,
  smoothingEnabled: true,
  showLandmarks: true,
  languageModel: 'ASL'
}

// Practice mode settings
export const PRACTICE_CONFIG = {
  MAX_ATTEMPTS: 3,
  DEBOUNCE_MS: 900,
  MIN_CONSENSUS: 2,
  BUFFER_WINDOW: 5
}

// Live caption settings
export const LIVE_CAPTION_CONFIG = {
  MAX_CAPTIONS: 40,
  BUFFER_WINDOW: 5,
  MIN_CONSENSUS: 3,
  DEBOUNCE_MS: 500
}

// Calibration settings
export const CALIBRATION_CONFIG = {
  DURATION_MS: 3500,
  UPDATE_INTERVAL_MS: 50,
  DEFAULT_HAND_SIZE: 0.165
}

// Keyboard shortcuts
export const KEYBOARD_SHORTCUTS = {
  Escape: MODES.MENU,
  p: MODES.PRACTICE,
  P: MODES.PRACTICE,
  l: MODES.LIVE_CAPTION,
  L: MODES.LIVE_CAPTION,
  s: MODES.SETTINGS,
  S: MODES.SETTINGS,
  c: MODES.CALIBRATION,
  C: MODES.CALIBRATION,
  d: MODES.DASHBOARD,
  D: MODES.DASHBOARD
}

// Color schemes
export const COLORS = {
  primary: 'blue',
  success: 'emerald',
  warning: 'amber',
  error: 'red',
  info: 'sky',
  neutral: 'slate'
}

// Accent gradients for cards
export const ACCENT_GRADIENTS = {
  emerald: 'from-emerald-500/35 via-emerald-300/10 to-transparent',
  sky: 'from-sky-500/35 via-sky-300/10 to-transparent',
  amber: 'from-amber-400/35 via-amber-300/10 to-transparent',
  slate: 'from-slate-300/30 via-slate-100/10 to-transparent'
}

// Local storage keys
export const STORAGE_KEYS = {
  APP_STATE: 'gesture-platform-storage',
  USER_PROGRESS: 'gesture-platform-progress',
  CALIBRATION: 'gesture-platform-calibration',
  SETTINGS: 'gesture-platform-settings'
}
