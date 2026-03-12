import { useState, useEffect, useRef, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/api/dialog'
import { useStore } from './store'
import CameraView from './components/CameraView'
import PracticeMode from './components/PracticeMode'
import LiveCaptionMode from './components/LiveCaptionMode'
import Settings from './components/Settings'
import Calibration from './components/Calibration'

type Mode = 'menu' | 'practice' | 'live-caption' | 'settings' | 'calibration'

function App() {
  const [mode, setMode] = useState<Mode>('menu')
  const {
    prediction,
    confidence,
    isCalibrated,
    handSize,
    progress,
    setPrediction,
    setConfidence,
    setCalibrated,
    setHandSize,
    updateProgress
  } = useStore()

  const handlePrediction = useCallback((pred: string | null, conf: number) => {
    setPrediction(pred)
    setConfidence(conf)

    // Update progress for practice mode
    if (mode === 'practice' && pred && conf > 0.9) {
      updateProgress(pred)
    }
  }, [mode, setPrediction, setConfidence, updateProgress])

  const renderContent = () => {
    switch (mode) {
      case 'practice':
        return <PracticeMode onBack={() => setMode('menu')} />
      case 'live-caption':
        return <LiveCaptionMode onBack={() => setMode('menu')} />
      case 'settings':
        return <Settings onBack={() => setMode('menu')} />
      case 'calibration':
        return <Calibration onComplete={() => {
          setCalibrated(true)
          setMode('menu')
        }} />
      default:
        return <MainMenu
          onPractice={() => setMode('practice')}
          onLiveCaption={() => setMode('live-caption')}
          onSettings={() => setMode('settings')}
          onCalibrate={() => setMode('calibration')}
          isCalibrated={isCalibrated}
          progress={progress}
        />
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-green-400">
            Gesture Platform
          </h1>
          <div className="flex items-center gap-4">
            {prediction && (
              <div className={`px-4 py-2 rounded-lg ${
                confidence > 0.9 ? 'bg-green-600' :
                confidence > 0.7 ? 'bg-yellow-600' : 'bg-red-600'
              }`}>
                <span className="text-2xl font-bold">{prediction}</span>
                <span className="ml-2 text-sm">{Math.round(confidence * 100)}%</span>
              </div>
            )}
            <button
              onClick={() => setMode('menu')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
            >
              Menu
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {renderContent()}
      </main>
    </div>
  )
}

function MainMenu({
  onPractice,
  onLiveCaption,
  onSettings,
  onCalibrate,
  isCalibrated,
  progress
}: {
  onPractice: () => void
  onLiveCaption: () => void
  onSettings: () => void
  onCalibrate: () => void
  isCalibrated: boolean
  progress: { letters: string[], words: string[] }
}) {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Card */}
      <div className="bg-gray-800 rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold mb-4">Your Progress</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-3xl font-bold text-green-400">
              {progress.letters.length}/26
            </div>
            <div className="text-gray-400">Letters Mastered</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-3xl font-bold text-blue-400">
              {progress.words.length}/100
            </div>
            <div className="text-gray-400">Words Learned</div>
          </div>
        </div>
      </div>

      {/* Calibration Status */}
      {!isCalibrated && (
        <div className="bg-yellow-900 border border-yellow-700 rounded-xl p-4 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-yellow-200">Calibration Required</h3>
              <p className="text-yellow-400 text-sm">
                Calibrate your hand size for better accuracy
              </p>
            </div>
            <button
              onClick={onCalibrate}
              className="px-6 py-2 bg-yellow-600 hover:bg-yellow-500 rounded-lg font-semibold"
            >
              Calibrate Now
            </button>
          </div>
        </div>
      )}

      {/* Main Menu Options */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <button
          onClick={onPractice}
          className="bg-gradient-to-br from-green-700 to-green-600 hover:from-green-600 hover:to-green-500 rounded-xl p-8 text-left transition-all"
        >
          <div className="text-4xl mb-4">📹</div>
          <h3 className="text-2xl font-bold mb-2">Practice Mode</h3>
          <p className="text-green-200">
            Learn ASL alphabet with real-time feedback and progress tracking
          </p>
        </button>

        <button
          onClick={onLiveCaption}
          className="bg-gradient-to-br from-blue-700 to-blue-600 hover:from-blue-600 hover:to-blue-500 rounded-xl p-8 text-left transition-all"
        >
          <div className="text-4xl mb-4">💬</div>
          <h3 className="text-2xl font-bold mb-2">Live Captions</h3>
          <p className="text-blue-200">
            Real-time translation of sign language to text
          </p>
        </button>

        <button
          onClick={onCalibrate}
          className="bg-gradient-to-br from-purple-700 to-purple-600 hover:from-purple-600 hover:to-purple-500 rounded-xl p-8 text-left transition-all"
        >
          <div className="text-4xl mb-4">⚙️</div>
          <h3 className="text-2xl font-bold mb-2">Calibration</h3>
          <p className="text-purple-200">
            Calibrate hand size for personalized accuracy
          </p>
        </button>

        <button
          onClick={onSettings}
          className="bg-gradient-to-br from-gray-700 to-gray-600 hover:from-gray-600 hover:to-gray-500 rounded-xl p-8 text-left transition-all"
        >
          <div className="text-4xl mb-4">🔧</div>
          <h3 className="text-2xl font-bold mb-2">Settings</h3>
          <p className="text-gray-300">
            Configure model, language, and preferences
          </p>
        </button>
      </div>
    </div>
  )
}

export default App
