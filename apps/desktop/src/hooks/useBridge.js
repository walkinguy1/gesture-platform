import { useEffect, useState, useRef } from 'react'
import { useStore } from '../store'

const BRIDGE_URL = 'ws://127.0.0.1:8765'
const MAX_BACKOFF_MS = 16000
const INITIAL_BACKOFF_MS = 1000

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
        backoffMsRef.current = INITIAL_BACKOFF_MS
        console.log('WebSocket bridge connected')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          switch (data.type) {
            case 'languages':
              setLanguages(data.languages || [], data.active ?? null)
              break
            case 'language_changed':
              setActiveLanguage(data.code ?? null)
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
      wsRef.current.close()
      wsRef.current = null
    }

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
