import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const API_URL = 'ws://127.0.0.1:8765/ws/predict'

export const useStore = create(
  persist(
    (set, get) => ({
      // Prediction state
      prediction: null,
      confidence: 0,

      // Calibration state
      isCalibrated: false,
      handSize: null,

      // WebSocket connection
      ws: null,
      isConnected: false,

      // Progress tracking
      progress: {
        letters: [],
        words: [],
        totalPracticeTime: 0,
        streak: 0,
        lastPracticeDate: null
      },

      // Settings
      settings: {
        confidenceThreshold: 0.7,
        smoothingWindow: 5,
        smoothingEnabled: true,
        showLandmarks: true,
        showFPS: true,
        languageModel: 'ASL'
      },

      // Actions
      setPrediction: (pred, conf) => set({ prediction: pred, confidence: conf }),

      setCalibrated: (calibrated) => set({ isCalibrated: calibrated }),

      setHandSize: (size) => set({ handSize: size }),

      // WebSocket connection management
      connect: () => {
        const { ws } = get()
        if (ws && ws.readyState === WebSocket.OPEN) return

        const socket = new WebSocket(API_URL)

        socket.onopen = () => {
          set({ ws: socket, isConnected: true })
        }

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)

            if (data.type === 'prediction') {
              set({ prediction: data.prediction, confidence: data.confidence || 0 })
            } else if (data.type === 'calibration') {
              if (data.status === 'complete') {
                set({
                  isCalibrated: true,
                  handSize: data.hand_size,
                })
              }
            }
          } catch { /* ignore parse errors */ }
        }

        socket.onclose = () => {
          set({ ws: null, isConnected: false })
        }

        socket.onerror = () => {
          set({ ws: null, isConnected: false })
        }

        set({ ws: socket })
      },

      disconnect: () => {
        const { ws } = get()
        if (ws) {
          ws.close()
          set({ ws: null, isConnected: false })
        }
      },

      sendFrame: (frameBase64) => {
        const { ws } = get()
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'predict', frame: frameBase64 }))
        }
      },

      sendCalibrationStart: () => {
        const { ws } = get()
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'calibrate_start' }))
        }
      },

      sendCalibrationStop: () => {
        const { ws } = get()
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'calibrate_stop' }))
        }
      },

      updateProgress: (letter) => {
        const { progress } = get()
        if (!progress.letters.includes(letter)) {
          set({
            progress: {
              ...progress,
              letters: [...progress.letters, letter]
            }
          })
        }
      },

      updateSettings: (newSettings) => {
        const { settings } = get()
        set({
          settings: {
            ...settings,
            ...newSettings
          }
        })
      },

      resetProgress: () => set({
        progress: {
          letters: [],
          words: [],
          totalPracticeTime: 0,
          streak: 0,
          lastPracticeDate: null
        },
        isCalibrated: false,
        handSize: null
      })
    }),
    {
      name: 'gesture-platform-storage',
      partialize: (state) => ({
        isCalibrated: state.isCalibrated,
        handSize: state.handSize,
        progress: state.progress,
        settings: state.settings,
      }),
    }
  )
)
