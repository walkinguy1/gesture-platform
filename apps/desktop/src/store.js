import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppState {
  // Prediction state
  prediction: string | null
  confidence: number

  // Calibration state
  isCalibrated: boolean
  handSize: number | null

  // Progress tracking
  progress: {
    letters: string[]
    words: string[]
    totalPracticeTime: number
    streak: number
    lastPracticeDate: string | null
  }

  // Settings
  settings: {
    confidenceThreshold: number
    smoothingEnabled: boolean
    showLandmarks: boolean
    languageModel: 'ASL' | 'BSL'
    theme: 'dark' | 'light'
    cameraIndex: number
  }

  // Actions
  setPrediction: (pred: string | null) => void
  setConfidence: (conf: number) => void
  setCalibrated: (calibrated: boolean) => void
  setHandSize: (size: number) => void
  updateProgress: (letter: string) => void
  updateSettings: (settings: Partial<AppState['settings']>) => void
  resetProgress: () => void
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      prediction: null,
      confidence: 0,
      isCalibrated: false,
      handSize: null,

      progress: {
        letters: [],
        words: [],
        totalPracticeTime: 0,
        streak: 0,
        lastPracticeDate: null
      },

      settings: {
        confidenceThreshold: 0.7,
        smoothingEnabled: true,
        showLandmarks: true,
        languageModel: 'ASL',
        theme: 'dark',
        cameraIndex: 0
      },

      // Actions
      setPrediction: (pred) => set({ prediction: pred }),

      setConfidence: (conf) => set({ confidence: conf }),

      setCalibrated: (calibrated) => set({ isCalibrated: calibrated }),

      setHandSize: (size) => set({ handSize: size }),

      updateProgress: (letter) => set((state) => {
        const today = new Date().toISOString().split('T')[0]
        const lastDate = state.progress.lastPracticeDate

        // Calculate streak
        let streak = state.progress.streak
        if (lastDate) {
          const lastPracticeDate = new Date(lastDate)
          const todayDate = new Date(today)
          const diffDays = Math.floor(
            (todayDate.getTime() - lastPracticeDate.getTime()) / (1000 * 60 * 60 * 24)
          )

          if (diffDays === 1) {
            streak += 1
          } else if (diffDays > 1) {
            streak = 1
          }
        } else {
          streak = 1
        }

        // Add letter if not already mastered
        const letters = state.progress.letters.includes(letter)
          ? state.progress.letters
          : [...state.progress.letters, letter]

        return {
          progress: {
            ...state.progress,
            letters,
            streak,
            lastPracticeDate: today,
            totalPracticeTime: state.progress.totalPracticeTime + 1
          }
        }
      }),

      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),

      resetProgress: () => set({
        progress: {
          letters: [],
          words: [],
          totalPracticeTime: 0,
          streak: 0,
          lastPracticeDate: null
        }
      })
    }),
    {
      name: 'gesture-platform-storage'
    }
  )
)
