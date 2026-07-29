import { useEffect, useState, useRef } from 'react'
import { useStore } from '../store'
import { publishFrame, clearFrames } from '../frameStream'

const BRIDGE_URL = 'ws://127.0.0.1:8765'
const MAX_BACKOFF_MS = 16000
const INITIAL_BACKOFF_MS = 1000

/** Map the UI's camelCase settings onto the backend's snake_case command shape. */
export function toBackendSettings(settings) {
  return {
    confidence_threshold: settings.confidenceThreshold,
    smoothing_enabled: settings.smoothingEnabled,
    show_landmarks: settings.showLandmarks,
    camera_index: settings.cameraIndex,
  }
}

export function useBridge() {
  const [connected, setConnected] = useState(false)
  const [bridgeStatus, setBridgeStatus] = useState('disconnected')
  const [fps, setFps] = useState(0)

  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const backoffMsRef = useRef(INITIAL_BACKOFF_MS)

  const setPrediction = useStore((state) => state.setPrediction)
  const setFpsStore = useStore((state) => state.setFps)
  const setBridgeStatusStore = useStore((state) => state.setBridgeStatus)
  const setLanguages = useStore((state) => state.setLanguages)
  const setActiveLanguage = useStore((state) => state.setActiveLanguage)
  const setBridgeError = useStore((state) => state.setBridgeError)
  const setBridgeApi = useStore((state) => state.setBridgeApi)
  const setCalibrationState = useStore((state) => state.setCalibrationState)
  const updateCalibration = useStore((state) => state.updateCalibration)

  const connect = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setBridgeStatus('connecting')
    setBridgeStatusStore('connecting')

    try {
      const ws = new WebSocket(BRIDGE_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setBridgeStatus('connected')
        setBridgeStatusStore('connected')
        setBridgeError(null)
        backoffMsRef.current = INITIAL_BACKOFF_MS
        console.log('WebSocket bridge connected')

        // The backend announces its language list at startup, before any
        // client exists, so that broadcast is lost. Ask for it on connect,
        // then push our persisted settings and calibration so a restarted
        // backend adopts them instead of running on its CLI defaults.
        const { settings, calibration } = useStore.getState()
        ws.send(JSON.stringify({ type: 'list_languages' }))
        ws.send(JSON.stringify({ type: 'set_settings', settings: toBackendSettings(settings) }))
        ws.send(JSON.stringify({ type: 'set_language', code: settings.languageModel }))
        if (calibration.isCalibrated && calibration.handSize) {
          ws.send(JSON.stringify({ type: 'set_calibration', hand_size: calibration.handSize }))
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          switch (data.type) {
            case 'frame':
              publishFrame({
                url: `data:image/jpeg;base64,${data.data}`,
                width: data.width,
                height: data.height,
                receivedAt: Date.now(),
              })
              break
            case 'languages':
              setLanguages(data.languages || [], data.active ?? null)
              break
            case 'language_changed':
              setActiveLanguage(data.code ?? null)
              break
            case 'calibration':
              setCalibrationState(data.state, data.progress ?? 0)
              if (data.state === 'complete' && data.hand_size) {
                // Persist the real measured hand size: the backend feeds it
                // straight into Normalizer, and re-sending it on reconnect
                // saves the user from recalibrating after every restart.
                updateCalibration({ isCalibrated: true, handSize: data.hand_size })
              }
              break
            case 'settings':
              // Backend echo confirming what it actually applied.
              console.log('Backend applied settings:', data.settings)
              break
            case 'error':
              console.error('Bridge error:', data.message)
              setBridgeError(data.message ?? 'Unknown bridge error')
              break
            case 'prediction':
            default:
              if (data.prediction !== undefined) {
                setPrediction(data.prediction, data.confidence || 0, data.prediction_kind ?? null)
              }
              if (data.fps !== undefined) {
                setFps(data.fps)
                setFpsStore(data.fps)
              }
              break
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        setBridgeStatus('disconnected')
        setBridgeStatusStore('disconnected')
        clearFrames()
        console.log('WebSocket bridge disconnected')

        // Exponential backoff reconnect
        const backoff = backoffMsRef.current
        reconnectTimeoutRef.current = setTimeout(() => {
          backoffMsRef.current = Math.min(backoff * 2, MAX_BACKOFF_MS)
          connect()
        }, backoff)
      }

      ws.onerror = (error) => {
        console.error('WebSocket bridge error:', error)
      }
    } catch (err) {
      console.error('Failed to create WebSocket:', err)
      setBridgeStatus('disconnected')
      setBridgeStatusStore('disconnected')
    }
  }

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      // Drop the reconnect handler first: an intentional close should not
      // schedule a retry that outlives the unmounted hook.
      wsRef.current.onclose = null
      wsRef.current.close()
      wsRef.current = null
    }

    clearFrames()
    setConnected(false)
    setBridgeStatus('disconnected')
    setBridgeStatusStore('disconnected')
  }

  const sendMessage = (payload) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.warn('Cannot send bridge message, socket not open:', payload)
      return false
    }
    wsRef.current.send(JSON.stringify(payload))
    return true
  }

  useEffect(() => {
    connect()
    setBridgeApi({ sendMessage })

    return () => {
      setBridgeApi({ sendMessage: null })
      disconnect()
    }
  }, [])

  return { connected, bridgeStatus, fps, sendMessage }
}
