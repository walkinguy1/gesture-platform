import { useState, useCallback, useEffect } from 'react'
import { invoke } from '@tauri-apps/api/core'
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
    settings,
    setPrediction,
    setConfidence,
    setCalibrated,
    setHandSize,
    updateProgress
  } = useStore()

  // Apply dark / light theme to the root element
  useEffect(() => {
    const root = document.documentElement
    if (settings.theme === 'light') {
      root.classList.add('light')
      root.classList.remove('dark')
    } else {
      root.classList.add('dark')
      root.classList.remove('light')
    }
  }, [settings.theme])

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore when focus is in a text/range input
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      switch (e.key) {
        case 'Escape':
          setMode('menu')
          break
        case 'p':
        case 'P':
          if (!e.altKey && !e.ctrlKey && !e.metaKey) setMode('practice')
          break
        case 'l':
        case 'L':
          if (!e.altKey && !e.ctrlKey && !e.metaKey) setMode('live-caption')
          break
        case 's':
        case 'S':
          if (!e.altKey && !e.ctrlKey && !e.metaKey) setMode('settings')
          break
        case 'c':
        case 'C':
          if (!e.altKey && !e.ctrlKey && !e.metaKey) setMode('calibration')
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

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

  const isDark = settings.theme !== 'light'

  return (
    <div className={`min-h-screen ${isDark ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'}`}>
      {/* Header */}
      <header className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-6 py-4`}>
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-green-500">
            Gesture Platform
          </h1>
          <div className="flex items-center gap-4">
            {prediction && (
              <div className={`px-4 py-2 rounded-lg ${
                confidence > 0.9 ? 'bg-green-600' :
                confidence > 0.7 ? 'bg-yellow-600' : 'bg-red-600'
              } text-white`}>
                <span className="text-2xl font-bold">{prediction}</span>
                <span className="ml-2 text-sm">{Math.round(confidence * 100)}%</span>
              </div>
            )}
            <button
              onClick={() => setMode('menu')}
              title="Back to menu (Esc)"
              className={`px-4 py-2 ${isDark ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} rounded-lg`}
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
  const { settings } = useStore()
  const isDark = settings.theme !== 'light'

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Card */}
      <div className={`${isDark ? 'bg-gray-800' : 'bg-white shadow'} rounded-xl p-6 mb-8`}>
        <h2 className="text-lg font-semibold mb-4">Your Progress</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className={`${isDark ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-4`}>
            <div className="text-3xl font-bold text-green-500">
              {progress.letters.length}/26
            </div>
            <div className={isDark ? 'text-gray-400' : 'text-gray-500'}>Letters Mastered</div>
          </div>
          <div className={`${isDark ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-4`}>
            <div className="text-3xl font-bold text-blue-500">
              {progress.words.length}/100
            </div>
            <div className={isDark ? 'text-gray-400' : 'text-gray-500'}>Words Learned</div>
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
              className="px-6 py-2 bg-yellow-600 hover:bg-yellow-500 rounded-lg font-semibold text-white"
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
          title="Practice Mode (P)"
          className="bg-gradient-to-br from-green-700 to-green-600 hover:from-green-600 hover:to-green-500 rounded-xl p-8 text-left transition-all text-white"
        >
          <div className="text-4xl mb-4">📹</div>
          <h3 className="text-2xl font-bold mb-2">Practice Mode</h3>
          <p className="text-green-200">
            Learn ASL alphabet with real-time feedback and progress tracking
          </p>
          <div className="mt-3 text-xs text-green-300 opacity-75">Shortcut: P</div>
        </button>

        <button
          onClick={onLiveCaption}
          title="Live Captions (L)"
          className="bg-gradient-to-br from-blue-700 to-blue-600 hover:from-blue-600 hover:to-blue-500 rounded-xl p-8 text-left transition-all text-white"
        >
          <div className="text-4xl mb-4">💬</div>
          <h3 className="text-2xl font-bold mb-2">Live Captions</h3>
          <p className="text-blue-200">
            Real-time translation of sign language to text
          </p>
          <div className="mt-3 text-xs text-blue-300 opacity-75">Shortcut: L</div>
        </button>

        <button
          onClick={onCalibrate}
          title="Calibration (C)"
          className="bg-gradient-to-br from-purple-700 to-purple-600 hover:from-purple-600 hover:to-purple-500 rounded-xl p-8 text-left transition-all text-white"
        >
          <div className="text-4xl mb-4">⚙️</div>
          <h3 className="text-2xl font-bold mb-2">Calibration</h3>
          <p className="text-purple-200">
            Calibrate hand size for personalized accuracy
          </p>
          <div className="mt-3 text-xs text-purple-300 opacity-75">Shortcut: C</div>
        </button>

        <button
          onClick={onSettings}
          title="Settings (S)"
          className={`bg-gradient-to-br ${isDark ? 'from-gray-700 to-gray-600 hover:from-gray-600 hover:to-gray-500' : 'from-gray-600 to-gray-500 hover:from-gray-500 hover:to-gray-400'} rounded-xl p-8 text-left transition-all text-white`}
        >
          <div className="text-4xl mb-4">🔧</div>
          <h3 className="text-2xl font-bold mb-2">Settings</h3>
          <p className="text-gray-300">
            Configure model, language, and preferences
          </p>
          <div className="mt-3 text-xs text-gray-400 opacity-75">Shortcut: S</div>
        </button>
      </div>
    </div>
  )
}

export default App
