import { useState, useEffect, useRef } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'
import { save } from '@tauri-apps/api/dialog'
import { writeTextFile } from '@tauri-apps/api/fs'

interface LiveCaptionModeProps {
  onBack: () => void
}

export default function LiveCaptionMode({ onBack }: LiveCaptionModeProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [captions, setCaptions] = useState<string[]>([])
  const [showCaptions, setShowCaptions] = useState(true)

  const { prediction, confidence } = useStore()

  // Buffer predictions to avoid flickering
  const predictionBuffer = useRef<string[]>([])
  const lastStablePrediction = useRef<string>('')

  useEffect(() => {
    if (!isRecording || !prediction) return

    // Add to buffer
    predictionBuffer.current.push(prediction)

    // Keep only last 10
    if (predictionBuffer.current.length > 10) {
      predictionBuffer.current.shift()
    }

    // Get most common
    const counts: Record<string, number> = {}
    predictionBuffer.current.forEach(p => {
      counts[p] = (counts[p] || 0) + 1
    })

    const mostCommon = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])[0]

    // Only update if it's consistent (at least 60%)
    if (mostCommon && mostCommon[1] >= 6) {
      if (mostCommon[0] !== lastStablePrediction.current) {
        lastStablePrediction.current = mostCommon[0]

        // Add timestamp
        const timestamp = new Date().toLocaleTimeString()
        const captionText = `[${timestamp}] ${mostCommon[0]}`

        setCaptions(prev => [...prev.slice(-50), captionText])
      }
    }
  }, [prediction, isRecording])

  const handleToggleRecording = () => {
    setIsRecording(!isRecording)
    if (!isRecording) {
      // Start fresh
      predictionBuffer.current = []
      lastStablePrediction.current = ''
      setCaptions([])
    }
  }

  const handleSaveCaptions = async () => {
    try {
      const filePath = await save({
        defaultPath: `captions_${Date.now()}.txt`,
        filters: [{ name: 'Text Files', extensions: ['txt'] }]
      })

      if (filePath) {
        const content = captions.join('\n')
        await writeTextFile(filePath, content)
        alert('Captions saved successfully!')
      }
    } catch (error) {
      console.error('Error saving captions:', error)
      alert('Error saving captions')
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          ← Back
        </button>

        <h2 className="text-2xl font-bold">
          Live Captions
        </h2>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowCaptions(!showCaptions)}
            className={`px-4 py-2 rounded-lg ${
              showCaptions ? 'bg-blue-600' : 'bg-gray-700'
            }`}
          >
            {showCaptions ? 'Hide Captions' : 'Show Captions'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera View */}
        <div className="bg-gray-800 rounded-xl overflow-hidden">
          <CameraView />

          {/* Recording indicator */}
          <div className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-500'
              }`} />
              <span className="font-semibold">
                {isRecording ? 'Recording' : 'Paused'}
              </span>
            </div>

            <button
              onClick={handleToggleRecording}
              className={`px-6 py-2 rounded-lg font-semibold ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-500'
                  : 'bg-green-600 hover:bg-green-500'
              }`}
            >
              {isRecording ? '⏸ Pause' : '▶ Start'}
            </button>
          </div>
        </div>

        {/* Captions Display */}
        <div className="bg-gray-800 rounded-xl p-6 flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Captions</h3>
            <button
              onClick={handleSaveCaptions}
              disabled={captions.length === 0}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              💾 Save Captions
            </button>
          </div>

          {/* Caption box */}
          <div className={`flex-1 bg-gray-900 rounded-xl p-4 overflow-y-auto ${
            !showCaptions && 'hidden'
          }`}>
            {captions.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-500">
                {isRecording
                  ? 'Waiting for signs...'
                  : 'Press Start to begin recording'}
              </div>
            ) : (
              <div className="space-y-2">
                {captions.map((caption, i) => (
                  <div
                    key={i}
                    className="text-lg animate-fade-in"
                  >
                    {caption}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Current prediction */}
          {prediction && isRecording && (
            <div className="mt-4 p-4 bg-gray-700 rounded-xl">
              <div className="text-sm text-gray-400 mb-1">Current:</div>
              <div className="flex items-center gap-4">
                <span className="text-3xl font-bold text-green-400">
                  {prediction}
                </span>
                <span className={`px-2 py-1 rounded ${
                  confidence > 0.9
                    ? 'bg-green-600'
                    : confidence > 0.7
                      ? 'bg-yellow-600'
                      : 'bg-red-600'
                }`}>
                  {Math.round(confidence * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tips */}
      <div className="mt-6 bg-gray-800 rounded-xl p-4">
        <h4 className="font-semibold mb-2">Tips for best results:</h4>
        <ul className="list-disc list-inside text-gray-400 space-y-1">
          <li>Ensure good lighting on your hands</li>
          <li>Keep your hand centered in the camera frame</li>
          <li>Hold signs steady for better recognition</li>
          <li>Make sure only one hand is visible for alphabet letters</li>
        </ul>
      </div>
    </div>
  )
}
