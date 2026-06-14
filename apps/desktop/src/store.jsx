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
        handSize: null
      },

      // ──────── PROGRESS (Learner Stats) ────────
      progress: emptyProgress(),

      // ──────── REALTIME (Current Session) ────────
      realtime: {
        prediction: null,
        confidence: 0
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
            letters: state.progress.letters.includes(letter)
              ? state.progress.letters
              : [...state.progress.letters, letter],
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

      setPrediction: (pred, conf) => set({
        realtime: { prediction: pred, confidence: conf }
      }),

      reset: () => set({
        progress: emptyProgress(),
        realtime: { prediction: null, confidence: 0 }
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
      })
    }
  )
)
