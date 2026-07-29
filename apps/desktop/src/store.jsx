import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const emptyProgress = () => ({
  letters: [],
  words: [],
  totalPracticeTime: 0,
  streak: 0,
  lastPracticeDate: null
})

const getTodayKey = () => new Date().toISOString().split('T')[0]

const calculateStreak = (lastPracticeDate, currentStreak) => {
  if (!lastPracticeDate) {
    return 1
  }

  const previous = new Date(lastPracticeDate)
  const current = new Date(getTodayKey())
  const diffDays = Math.floor(
    (current.getTime() - previous.getTime()) / (1000 * 60 * 60 * 24)
  )

  if (diffDays <= 0) {
    return Math.max(currentStreak, 1)
  }

  if (diffDays === 1) {
    return currentStreak + 1
  }

  return 1
}

export const useStore = create()(
  persist(
    (set) => ({
      // ──────── SETTINGS (User Preferences) ────────
      settings: {
        theme: 'dark',
        cameraIndex: 0,
        confidenceThreshold: 0.7,
        smoothingEnabled: true,
        showLandmarks: true,
        languageModel: 'ASL'
      },

      // ──────── CALIBRATION (Device Profile) ────────
      calibration: {
        isCalibrated: false,
        // Median wrist-to-middle-fingertip distance measured by the backend.
        // Persisted and replayed on reconnect so Normalizer keeps its
        // calibration across restarts.
        handSize: null
      },

      // ──────── PROGRESS (Learner Stats) ────────
      progress: emptyProgress(),

      // ──────── SESSION STATE (Non-persisted) ────────
      session: {
        focusLetter: null
      },

      // ──────── REALTIME (Current Session) ────────
      realtime: {
        prediction: null,
        confidence: 0,
        predictionKind: null, // 'static' | 'dynamic' | null
        fps: 0,
        bridgeStatus: 'disconnected',
        languages: [], // [{ code, name, country, static_ready, dynamic_ready, supports_dynamic }]
        activeLanguage: null, // backend-confirmed active language code
        lastError: null,
        // Live calibration run driven by the backend:
        // 'idle' | 'started' | 'progress' | 'complete' | 'cancelled'
        calibrationState: 'idle',
        calibrationProgress: 0
      },

      // WS bridge command channel, wired up by useBridge() on mount so any
      // component (not just the one that called the hook) can send
      // commands like switching the active sign language.
      bridgeApi: {
        sendMessage: null
      },

      // ──────── ACTIONS ────────
      updateSettings: (updates) => set((state) => ({
        settings: { ...state.settings, ...updates }
      })),

      updateCalibration: (updates) => set((state) => ({
        calibration: { ...state.calibration, ...updates }
      })),

      updateProgress: (letter) =>
        set((state) => ({
          progress: {
            ...state.progress,
            letters: state.progress.letters.some(l => l.letter === letter)
              ? state.progress.letters
              : [...state.progress.letters, { letter, masteredAt: new Date().toISOString() }],
            streak: calculateStreak(
              state.progress.lastPracticeDate,
              state.progress.streak
            ),
            lastPracticeDate: getTodayKey()
          }
        })),

      addPracticeTime: (seconds) =>
        set((state) => ({
          progress: {
            ...state.progress,
            totalPracticeTime:
              state.progress.totalPracticeTime + Math.max(0, Math.round(seconds / 60))
          }
        })),

      setPrediction: (pred, conf, kind = null) => set((state) => ({
        realtime: { ...state.realtime, prediction: pred, confidence: conf, predictionKind: kind }
      })),

      setFps: (fps) => set((state) => ({
        realtime: { ...state.realtime, fps }
      })),

      setBridgeStatus: (status) => set((state) => ({
        realtime: { ...state.realtime, bridgeStatus: status }
      })),

      setLanguages: (languages, active) => set((state) => ({
        realtime: { ...state.realtime, languages, activeLanguage: active }
      })),

      setActiveLanguage: (code) => set((state) => ({
        realtime: { ...state.realtime, activeLanguage: code }
      })),

      setBridgeError: (message) => set((state) => ({
        realtime: { ...state.realtime, lastError: message }
      })),

      setCalibrationState: (calibrationState, calibrationProgress = 0) => set((state) => ({
        realtime: { ...state.realtime, calibrationState, calibrationProgress }
      })),

      setBridgeApi: (api) => set(() => ({ bridgeApi: api })),

      setFocusLetter: (letter) => set((state) => ({
        session: { ...state.session, focusLetter: letter }
      })),

      reset: () => set((state) => ({
        progress: emptyProgress(),
        realtime: {
          prediction: null,
          confidence: 0,
          predictionKind: null,
          fps: 0,
          bridgeStatus: 'disconnected',
          languages: state.realtime.languages,
          activeLanguage: state.realtime.activeLanguage,
          lastError: null,
          calibrationState: 'idle',
          calibrationProgress: 0
        }
      })),

      resetProgress: () => set({
        progress: emptyProgress()
      })
    }),
    {
      name: 'gesture-platform',
      version: 3,
      // Optimize persistence: only persist settings and progress, not realtime
      partialize: (state) => ({
        settings: state.settings,
        calibration: state.calibration,
        progress: state.progress
        // session and realtime are NOT persisted
      })
    }
  )
)
