import { useState, useEffect, useRef } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'
import { save } from '@tauri-apps/dialog'
import { writeTextFile } from '@tauri-apps/fs'

interface LiveCaptionModeProps {
  onBack: () => void
}

// Simple prefix-based word suggestion dictionary (most common English words)
const COMMON_WORDS = [
  'hello', 'help', 'home', 'have', 'here',
  'thank', 'thanks', 'the', 'this', 'that', 'they',
  'yes', 'you', 'your',
  'no', 'not', 'name',
  'please', 'play',
  'good', 'go', 'great',
  'bye', 'be', 'but',
  'i', 'is', 'it', 'in',
  'want', 'what', 'where', 'when', 'who', 'why', 'with', 'will', 'was',
  'can', 'come', 'call',
  'do', 'done', 'dont',
  'are', 'am', 'and', 'all', 'a',
  'more', 'me', 'my', 'meet',
  'see', 'sorry', 'some', 'so', 'she',
  'ready', 'right',
  'fine', 'for', 'from', 'friend',
  'like', 'love', 'learn', 'later',
  'ok', 'of',
  'need',
  'understand', 'us',
  'very',
  'know',
  'just',
]

function getSuggestions(prefix: string, max: number = 4): string[] {
  if (!prefix) return []
  const lower = prefix.toLowerCase()
  return COMMON_WORDS.filter((w) => w.startsWith(lower) && w !== lower).slice(0, max)
}

export default function LiveCaptionMode({ onBack }: LiveCaptionModeProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [captions, setCaptions] = useState<string[]>([])
  const [showCaptions, setShowCaptions] = useState(true)
  const [sentence, setSentence] = useState('')
  const [copySuccess, setCopySuccess] = useState(false)

  const { prediction, confidence, settings } = useStore()
  const isDark = settings.theme !== 'light'

  // Buffer predictions to avoid flickering
  const predictionBuffer = useRef<string[]>([])
  const lastStablePrediction = useRef<string>('')
  const captionsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll captions to bottom
  useEffect(() => {
    captionsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [captions])

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

        // Append letter to current sentence
        setSentence((prev: string) => prev + mostCommon[0])

        // Add timestamp caption
        const timestamp = new Date().toLocaleTimeString()
        const captionText = `[${timestamp}] ${mostCommon[0]}`
        setCaptions((prev: string[]) => [...prev.slice(-50), captionText])
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
      setSentence('')
    }
  }

  const handleAddSpace = () => {
    setSentence((prev: string) => prev + ' ')
    lastStablePrediction.current = ''
    predictionBuffer.current = []
  }

  const handleBackspace = () => {
    setSentence((prev: string) => prev.slice(0, -1))
  }

  const handleClearSentence = () => {
    setSentence('')
    lastStablePrediction.current = ''
    predictionBuffer.current = []
  }

  const handleSuggestionClick = (word: string) => {
    const parts = sentence.trimEnd().split(' ')
    parts[parts.length - 1] = word
    setSentence(parts.join(' ') + ' ')
  }

  const handleCopyToClipboard = async () => {
    const text = sentence || captions.join('\n')
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopySuccess(true)
      setTimeout(() => setCopySuccess(false), 2000)
    } catch {
      // Clipboard API not available (non-browser context)
      console.error('Clipboard write failed')
    }
  }

  const handleSaveCaptions = async () => {
    try {
      const filePath = await save({
        defaultPath: `captions_${Date.now()}.txt`,
        filters: [{ name: 'Text Files', extensions: ['txt'] }]
      })

      if (filePath) {
        const content = sentence
          ? `Sentence: ${sentence}\n\nRaw captions:\n${captions.join('\n')}`
          : captions.join('\n')
        await writeTextFile(filePath, content)
        alert('Captions saved successfully!')
      }
    } catch (error) {
      console.error('Error saving captions:', error)
      alert('Error saving captions')
    }
  }

  // Current partial word for suggestions
  const currentWord = sentence.split(' ').filter(Boolean).slice(-1)[0] || ''
  const suggestions = getSuggestions(currentWord)

  const card = isDark ? 'bg-gray-800' : 'bg-white shadow'
  const btn = isDark ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-900'

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBack}
          className={`px-4 py-2 ${btn} rounded-lg`}
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
              showCaptions ? 'bg-blue-600 text-white' : btn
            }`}
          >
            {showCaptions ? 'Hide Captions' : 'Show Captions'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera View */}
        <div className={`${card} rounded-xl overflow-hidden`}>
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
              className={`px-6 py-2 rounded-lg font-semibold text-white ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-500'
                  : 'bg-green-600 hover:bg-green-500'
              }`}
            >
              {isRecording ? '⏸ Pause' : '▶ Start'}
            </button>
          </div>
        </div>

        {/* Captions & Sentence Builder */}
        <div className={`${card} rounded-xl p-6 flex flex-col gap-4`}>
          {/* Sentence builder */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">Sentence</h3>
              <div className="flex gap-2">
                <button
                  onClick={handleCopyToClipboard}
                  disabled={!sentence}
                  title="Copy sentence to clipboard"
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                    copySuccess ? 'bg-green-600 text-white' : 'bg-blue-600 hover:bg-blue-500 text-white'
                  }`}
                >
                  {copySuccess ? '✓ Copied!' : '📋 Copy'}
                </button>
                <button
                  onClick={handleSaveCaptions}
                  disabled={captions.length === 0 && !sentence}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  💾 Save
                </button>
              </div>
            </div>

            {/* Sentence display */}
            <div className={`min-h-16 p-3 rounded-xl text-xl font-mono tracking-wider break-all ${
              isDark ? 'bg-gray-900' : 'bg-gray-100'
            }`}>
              {sentence || <span className="opacity-40">Spell letters to build a sentence…</span>}
              {isRecording && <span className="animate-pulse text-green-400">|</span>}
            </div>

            {/* Sentence controls */}
            <div className="flex gap-2 mt-2 flex-wrap">
              <button
                onClick={handleAddSpace}
                className={`px-3 py-1.5 ${btn} rounded-lg text-sm`}
                title="Add space"
              >
                ⎵ Space
              </button>
              <button
                onClick={handleBackspace}
                disabled={!sentence}
                className={`px-3 py-1.5 ${btn} rounded-lg text-sm disabled:opacity-50`}
                title="Backspace"
              >
                ⌫ Delete
              </button>
              <button
                onClick={handleClearSentence}
                disabled={!sentence}
                className="px-3 py-1.5 bg-red-700 hover:bg-red-600 text-white rounded-lg text-sm disabled:opacity-50"
                title="Clear sentence"
              >
                ✕ Clear
              </button>
            </div>

            {/* Word suggestions */}
            {suggestions.length > 0 && (
              <div className="mt-2">
                <p className={`text-xs mb-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                  Word suggestions:
                </p>
                <div className="flex gap-2 flex-wrap">
                  {suggestions.map((word) => (
                    <button
                      key={word}
                      onClick={() => handleSuggestionClick(word)}
                      className="px-3 py-1 bg-blue-700 hover:bg-blue-600 text-white rounded-full text-sm"
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Captions Display */}
          <div className={`flex flex-col flex-1 ${showCaptions ? '' : 'hidden'}`}>
            <h3 className="text-base font-semibold mb-2">Raw Captions</h3>
            <div className={`flex-1 min-h-32 max-h-56 overflow-y-auto rounded-xl p-4 ${
              isDark ? 'bg-gray-900' : 'bg-gray-100'
            }`}>
              {captions.length === 0 ? (
                <div className="h-full flex items-center justify-center opacity-40 text-sm">
                  {isRecording
                    ? 'Waiting for signs…'
                    : 'Press Start to begin recording'}
                </div>
              ) : (
                <div className="space-y-1">
                  {captions.map((caption, i) => (
                    <div key={i} className="text-sm animate-fade-in">
                      {caption}
                    </div>
                  ))}
                  <div ref={captionsEndRef} />
                </div>
              )}
            </div>
          </div>

          {/* Current prediction */}
          {prediction && isRecording && (
            <div className={`p-4 ${isDark ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl`}>
              <div className={`text-sm mb-1 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>Current:</div>
              <div className="flex items-center gap-4">
                <span className="text-3xl font-bold text-green-500">
                  {prediction}
                </span>
                <span className={`px-2 py-1 rounded text-white ${
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
      <div className={`mt-6 ${card} rounded-xl p-4`}>
        <h4 className="font-semibold mb-2">Tips for best results:</h4>
        <ul className={`list-disc list-inside space-y-1 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
          <li>Ensure good lighting on your hands</li>
          <li>Keep your hand centered in the camera frame</li>
          <li>Hold signs steady for better recognition</li>
          <li>Make sure only one hand is visible for alphabet letters</li>
          <li>Use ⎵ Space to separate letters into words</li>
          <li>Tap a word suggestion to auto-complete the current word</li>
        </ul>
      </div>
    </div>
  )
}
