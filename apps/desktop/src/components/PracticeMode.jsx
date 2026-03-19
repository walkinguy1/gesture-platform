import { useState, useEffect } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

export default function PracticeMode({ onBack }) {
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

  useEffect(() => {
    if (prediction === currentLetter && confidence > settings.confidenceThreshold) {
      const newAttempts = attempts + 1
      setAttempts(newAttempts)

      if (newAttempts >= maxAttempts) {
        setShowSuccess(true)
        setFeedback('Perfect! ✓')
        updateProgress(currentLetter)

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
  }, [prediction, confidence, currentLetter, attempts, settings.confidenceThreshold])

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

  const unmasteredLetters = ALPHABET.filter(l => !progress.letters.includes(l))

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Practice Mode</h2>
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          Back
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <CameraView />

          <div className="bg-gray-800 rounded-xl p-6">
            <div className="text-center">
              <div className="text-6xl font-bold mb-4">{currentLetter}</div>

              <div className="flex justify-center gap-2 mb-4">
                {Array.from({ length: maxAttempts }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-full ${
                      i < attempts ? 'bg-green-500' : 'bg-gray-600'
                    }`}
                  />
                ))}
              </div>

              {feedback && (
                <div className={`text-lg ${
                  showSuccess ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  {feedback}
                </div>
              )}
            </div>

            <div className="flex gap-4 mt-6">
              <button
                onClick={handlePrevious}
                className="flex-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
              >
                Previous
              </button>
              <button
                onClick={handleSkip}
                className="flex-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
              >
                Skip
              </button>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Progress</h3>
            <div className="text-3xl font-bold mb-2">
              {progress.letters.length} / 26
            </div>
            <div className="text-gray-400 text-sm">letters mastered</div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Alphabet</h3>
            <div className="grid grid-cols-6 gap-2">
              {ALPHABET.map(letter => (
                <button
                  key={letter}
                  onClick={() => {
                    setCurrentLetter(letter)
                    setAttempts(0)
                    setFeedback('')
                  }}
                  className={`aspect-square flex items-center justify-center rounded font-bold ${
                    progress.letters.includes(letter)
                      ? 'bg-green-600 text-white'
                      : letter === currentLetter
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {letter}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
