import { useState, useEffect, useRef, useCallback } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

interface PracticeModeProps {
  onBack: () => void
}

export default function PracticeMode({ onBack }: PracticeModeProps) {
  const [currentLetter, setCurrentLetter] = useState('A')
  const [attempts, setAttempts] = useState(0)
  const [showSuccess, setShowSuccess] = useState(false)
  const [feedback, setFeedback] = useState('')

  const {
    prediction,
    confidence,
    progress,
    updateProgress,
    settings
  } = useStore()

  const maxAttempts = 3

  // Check prediction against current letter
  useEffect(() => {
    if (prediction === currentLetter && confidence > settings.confidenceThreshold) {
      const newAttempts = attempts + 1
      setAttempts(newAttempts)

      if (newAttempts >= maxAttempts) {
        // Letter mastered!
        setShowSuccess(true)
        setFeedback('Perfect! ✓')
        updateProgress(currentLetter)

        // Move to next letter after delay
        setTimeout(() => {
          setShowSuccess(false)
          setAttempts(0)
          moveToNextLetter()
        }, 2000)
      } else {
        setFeedback(`Good! ${maxAttempts - newAttempts} more to go`)
      }
    } else if (prediction && prediction !== currentLetter) {
      setAttempts(0)
      setFeedback(`That's ${prediction}, try ${currentLetter}`)
    }
  }, [prediction, confidence])

  const moveToNextLetter = () => {
    const currentIndex = ALPHABET.indexOf(currentLetter)
    const nextIndex = (currentIndex + 1) % ALPHABET.length
    setCurrentLetter(ALPHABET[nextIndex])
    setFeedback('')
  }

  const handleSkip = () => {
    moveToNextLetter()
    setAttempts(0)
    setFeedback('')
  }

  const handlePrevious = () => {
    const currentIndex = ALPHABET.indexOf(currentLetter)
    const prevIndex = (currentIndex - 1 + ALPHABET.length) % ALPHABET.length
    setCurrentLetter(ALPHABET[prevIndex])
    setAttempts(0)
    setFeedback('')
  }

  // Get letters not yet mastered
  const unmasteredLetters = ALPHABET.filter(l => !progress.letters.includes(l))
  const nextLetter = unmasteredLetters[0] || currentLetter

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
          Practice: Letter '{currentLetter}'
        </h2>

        <div className="text-gray-400">
          Progress: {progress.letters.length}/26
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera View */}
        <div className="bg-gray-800 rounded-xl overflow-hidden">
          <CameraView />

          {/* Feedback overlay */}
          {feedback && (
            <div className={`mt-4 p-4 rounded-lg text-center ${
              showSuccess
                ? 'bg-green-600'
                : confidence > 0.7
                  ? 'bg-yellow-600'
                  : 'bg-gray-700'
            }`}>
              <p className="text-xl font-bold">{feedback}</p>
              {showSuccess && (
                <p className="text-green-200">Moving to next letter...</p>
              )}
            </div>
          )}

          {/* Attempts indicator */}
          <div className="mt-4 px-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-400">Attempts:</span>
              <span className="font-bold">{attempts}/{maxAttempts}</span>
            </div>
            <div className="flex gap-2">
              {[...Array(maxAttempts)].map((_, i) => (
                <div
                  key={i}
                  className={`flex-1 h-3 rounded-full ${
                    i < attempts ? 'bg-green-500' : 'bg-gray-600'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Reference and Controls */}
        <div className="space-y-6">
          {/* Reference Image */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Reference Sign</h3>
            <div className="flex justify-center">
              <div className="w-48 h-48 bg-gray-700 rounded-xl flex items-center justify-center text-8xl font-bold text-green-400">
                {currentLetter}
              </div>
            </div>
            <p className="text-center text-gray-400 mt-4">
              Make this sign with your hand
            </p>
          </div>

          {/* Current Stats */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Your Stats</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-400">
                  {progress.letters.length}
                </div>
                <div className="text-gray-400">Mastered</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-400">
                  {progress.streak}
                </div>
                <div className="text-gray-400">Day Streak</div>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex gap-4">
            <button
              onClick={handlePrevious}
              className="flex-1 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold"
            >
              ← Previous
            </button>
            <button
              onClick={handleSkip}
              className="flex-1 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold"
            >
              Skip →
            </button>
          </div>

          {/* Suggest next */}
          {unmasteredLetters.length > 0 && currentLetter !== nextLetter && (
            <button
              onClick={() => {
                setCurrentLetter(nextLetter)
                setAttempts(0)
                setFeedback('')
              }}
              className="w-full py-3 bg-green-600 hover:bg-green-500 rounded-lg font-semibold"
            >
              Next: Practice '{nextLetter}' ({unmasteredLetters.length} remaining)
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
