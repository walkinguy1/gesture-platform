/**
 * Custom Hooks for Gesture Platform
 * Extracting reusable logic from components
 */

import { useEffect, useRef, useCallback, useState } from 'react'

/**
 * usePredictionBuffer
 * Shared logic for buffering and voting on predictions
 * Used by both PracticeMode and LiveCaptionMode
 */
export function usePredictionBuffer(windowSize = 5) {
  const buffer = useRef([])
  const lastPrediction = useRef(null)

  const add = useCallback((prediction) => {
    if (!prediction) return
    buffer.current = [...buffer.current.slice(-(windowSize - 1)), prediction]
  }, [windowSize])

  const getMajority = useCallback(() => {
    if (buffer.current.length === 0) return null

    const counts = {}
    buffer.current.forEach((p) => {
      counts[p] = (counts[p] || 0) + 1
    })

    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1])
    const [majority, count] = sorted[0] || [null, 0]

    return { prediction: majority, count, buffer: buffer.current }
  }, [])

  const clear = useCallback(() => {
    buffer.current = []
    lastPrediction.current = null
  }, [])

  const hasConsensus = useCallback((minCount = 3) => {
    const result = getMajority()
    return result && result.count >= minCount
  }, [getMajority])

  return { add, getMajority, clear, hasConsensus, buffer: buffer.current }
}

/**
 * useDebounce
 * Debounce predictions to avoid rapid-fire updates
 */
export function useDebounce(prediction, debounceMs = 900) {
  const lastAcceptedAt = useRef(0)

  return useCallback(() => {
    const now = Date.now()
    if (now - lastAcceptedAt.current >= debounceMs) {
      lastAcceptedAt.current = now
      return true
    }
    return false
  }, [debounceMs])
}

/**
 * usePredictionHandler
 * Combined logic for handling predictions with threshold, debouncing, and buffering
 */
export function usePredictionHandler({
  prediction,
  confidence,
  threshold = 0.7,
  debounceMs = 900,
  bufferSize = 5,
  onValid = null
}) {
  const isValid = prediction && confidence >= threshold
  const debounce = useDebounce(prediction, debounceMs)
  const buffer = usePredictionBuffer(bufferSize)

  const checkConsensus = useCallback(
    (minCount = 3) => {
      if (!isValid || !debounce()) return null

      buffer.add(prediction)

      if (buffer.hasConsensus(minCount)) {
        const result = buffer.getMajority()
        if (onValid) onValid(result)
        return result
      }

      return null
    },
    [isValid, prediction, buffer, debounce, onValid]
  )

  return { isValid, checkConsensus, buffer }
}

/**
 * useLocalStorage
 * Persist state to localStorage with type safety
 */
export function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.error(error)
      return initialValue
    }
  })

  const setValue = useCallback(
    (value) => {
      try {
        const valueToStore = value instanceof Function ? value(storedValue) : value
        setStoredValue(valueToStore)
        window.localStorage.setItem(key, JSON.stringify(valueToStore))
      } catch (error) {
        console.error(error)
      }
    },
    [key, storedValue]
  )

  return [storedValue, setValue]
}

/**
 * useAsync
 * Handle async operations with loading/error states
 */
export function useAsync(asyncFunction, immediate = true) {
  const [status, setStatus] = useState('idle')
  const [value, setValue] = useState(null)
  const [error, setError] = useState(null)

  const execute = useCallback(async () => {
    setStatus('pending')
    setValue(null)
    setError(null)
    try {
      const response = await asyncFunction()
      setValue(response)
      setStatus('success')
      return response
    } catch (error) {
      setError(error)
      setStatus('error')
      throw error
    }
  }, [asyncFunction])

  useEffect(() => {
    if (immediate) {
      execute()
    }
  }, [execute, immediate])

  return { execute, status, value, error }
}

/**
 * useTimeout
 * Cleanup-safe timeout hook
 */
export function useTimeout(callback, delay) {
  const savedCallback = useRef()

  useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  useEffect(() => {
    function tick() {
      savedCallback.current()
    }

    if (delay !== null) {
      let id = setTimeout(tick, delay)
      return () => clearTimeout(id)
    }
  }, [delay])
}

/**
 * useInterval
 * Cleanup-safe interval hook
 */
export function useInterval(callback, delay) {
  const savedCallback = useRef()

  useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  useEffect(() => {
    function tick() {
      savedCallback.current()
    }

    if (delay !== null) {
      let id = setInterval(tick, delay)
      return () => clearInterval(id)
    }
  }, [delay])
}
