import { useState, useEffect, useRef } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'

// Tauri v2 plugin imports – gracefully degrade in browser dev mode
let tauriSave, tauriWriteTextFile
try {
  const dialog = await import('@tauri-apps/dialog')
  const fs = await import('@tauri-apps/fs')
  tauriSave = dialog.save
  tauriWriteTextFile = fs.writeTextFile
} catch {
  // Running outside Tauri (e.g. vite dev server)
  tauriSave = null
  tauriWriteTextFile = null
}

export default function LiveCaptionMode({ onBack }) {
  const [isRecording, setIsRecording] = useState(false)
  const [captions, setCaptions] = useState([])
  const [showCaptions, setShowCaptions] = useState(true)

  const { prediction, confidence } = useStore()

  const predictionBuffer = useRef([])
  const lastStablePrediction = useRef('')

  useEffect(() => {
    if (!isRecording || !prediction) return

    predictionBuffer.current.push(prediction)

    if (predictionBuffer.current.length > 10) {
      predictionBuffer.current.shift()
    }

    const counts = {}
    predictionBuffer.current.forEach(p => {
      counts[p] = (counts[p] || 0) + 1
    })

    const mostCommon = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])[0]

    if (mostCommon && mostCommon[1] >= 6) {
      if (mostCommon[0] !== lastStablePrediction.current) {
        lastStablePrediction.current = mostCommon[0]

        const timestamp = new Date().toLocaleTimeString()
        const captionText = `[${timestamp}] ${mostCommon[0]}`

        setCaptions(prev => [...prev, captionText])
      }
    }
  }, [prediction, isRecording])

  const toggleRecording = () => {
    if (!isRecording) {
      setCaptions([])
      predictionBuffer.current = []
      lastStablePrediction.current = ''
    }
    setIsRecording(!isRecording)
  }

  const clearCaptions = () => {
    setCaptions([])
    predictionBuffer.current = []
    lastStablePrediction.current = ''
  }

  const saveCaptions = async () => {
    const text = captions.join('\n')

    if (tauriSave && tauriWriteTextFile) {
      // Tauri native file dialog
      const path = await tauriSave({
        defaultPath: 'captions.txt',
        filters: [{ name: 'Text', extensions: ['txt'] }]
      })
      if (path) {
        await tauriWriteTextFile(path, text)
      }
    } else {
      // Browser fallback – download as file
      const blob = new Blob([text], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'captions.txt'
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Live Caption Mode</h2>
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          Back
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <CameraView />

          <div className="flex gap-4">
            <button
              onClick={toggleRecording}
              className={`flex-1 py-3 rounded-lg font-semibold ${isRecording
                  ? 'bg-red-600 hover:bg-red-500'
                  : 'bg-green-600 hover:bg-green-500'
                }`}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>

            <button
              onClick={clearCaptions}
              disabled={captions.length === 0}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg"
            >
              Clear
            </button>

            <button
              onClick={saveCaptions}
              disabled={captions.length === 0}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg"
            >
              Save
            </button>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Captions</h3>
            <button
              onClick={() => setShowCaptions(!showCaptions)}
              className="text-sm text-gray-400 hover:text-white"
            >
              {showCaptions ? 'Hide' : 'Show'}
            </button>
          </div>

          {showCaptions && (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {captions.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  {isRecording
                    ? 'Start signing to see captions...'
                    : 'Press Start Recording to begin'}
                </div>
              ) : (
                captions.map((caption, i) => (
                  <div key={i} className="bg-gray-700 rounded p-2 text-sm">
                    {caption}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
