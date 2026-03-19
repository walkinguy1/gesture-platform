import { useState, useCallback, useEffect } from 'react'
import { useStore } from './store'
import PracticeMode from './components/PracticeMode'
import LiveCaptionMode from './components/LiveCaptionMode'
import Settings from './components/Settings'
import Calibration from './components/Calibration'

function App() {
  const [mode, setMode] = useState('menu')
  const {
    prediction,
    confidence,
    isCalibrated,
    isConnected,
    progress,
    setPrediction,
    setCalibrated,
    updateProgress,
    connect,
    disconnect
  } = useStore()

  // Connect to Python backend on mount
  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  const handlePrediction = useCallback((pred, conf) => {
    setPrediction(pred, conf)

    if (mode === 'practice' && pred && conf > 0.9) {
      updateProgress(pred)
    }
  }, [mode, setPrediction, updateProgress])

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
        return (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold mb-2">Gesture Platform</h1>
              <p className="text-gray-400">Real-Time Sign Language Translation</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <button
                onClick={() => setMode(isCalibrated ? 'practice' : 'calibration')}
                className="bg-blue-600 hover:bg-blue-500 p-8 rounded-xl text-left"
              >
                <div className="text-2xl font-bold mb-2">Practice Mode</div>
                <div className="text-blue-200">
                  Learn ASL alphabet with real-time feedback
                </div>
                <div className="mt-4 text-sm text-blue-300">
                  {progress.letters.length}/26 letters mastered
                </div>
              </button>

              <button
                onClick={() => setMode('live-caption')}
                className="bg-green-600 hover:bg-green-500 p-8 rounded-xl text-left"
              >
                <div className="text-2xl font-bold mb-2">Live Caption</div>
                <div className="text-green-200">
                  Real-time sign language to text translation
                </div>
              </button>

              <button
                onClick={() => setMode('calibration')}
                className="bg-purple-600 hover:bg-purple-500 p-8 rounded-xl text-left"
              >
                <div className="text-2xl font-bold mb-2">Calibration</div>
                <div className="text-purple-200">
                  Calibrate for your hand size
                </div>
                {isCalibrated && (
                  <div className="mt-4 text-sm text-purple-300">
                    ✓ Calibrated
                  </div>
                )}
              </button>

              <button
                onClick={() => setMode('settings')}
                className="bg-gray-700 hover:bg-gray-600 p-8 rounded-xl text-left"
              >
                <div className="text-2xl font-bold mb-2">Settings</div>
                <div className="text-gray-300">
                  Configure recognition and display
                </div>
              </button>
            </div>

            {prediction && (
              <div className="bg-gray-800 rounded-xl p-6 text-center">
                <div className="text-sm text-gray-400 mb-2">Current Prediction</div>
                <div className="text-4xl font-bold">{prediction}</div>
                <div className="text-sm text-gray-400 mt-2">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </div>
              </div>
            )}

            <div className="mt-4 text-center text-sm">
              <span className={`inline-flex items-center gap-1.5 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                {isConnected ? 'Backend connected' : 'Backend offline — start the API server'}
              </span>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      {renderContent()}
    </div>
  )
}

export default App
