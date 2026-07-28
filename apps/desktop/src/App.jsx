import { useEffect, useState } from 'react'
import { useStore } from './store'
import { useBridge } from './hooks/useBridge'
import PracticeMode from './components/PracticeMode'
import LiveCaptionMode from './components/LiveCaptionMode'
import Settings from './components/Settings'
import Calibration from './components/Calibration'
import Dashboard from './components/Dashboard'
import Navigation from './components/Navigation'
import { MODES, KEYBOARD_SHORTCUTS } from './constants'

function App() {
  const [mode, setMode] = useState(MODES.DASHBOARD)
  const { realtime, calibration, progress, settings } = useStore()

  // Establishes the WebSocket connection to the Python recognizer backend
  // and keeps `realtime.*` in the store updated for every other component.
  const { sendMessage } = useBridge()

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('light', settings.theme === 'light')
    root.classList.toggle('dark', settings.theme !== 'light')
  }, [settings.theme])

  // Once the backend reports its active language, make sure it matches the
  // user's saved preference (e.g. after a restart where the backend starts
  // on its own default rather than what was last selected in this app).
  useEffect(() => {
    if (
      realtime.bridgeStatus === 'connected' &&
      realtime.activeLanguage &&
      realtime.activeLanguage !== settings.languageModel
    ) {
      sendMessage({ type: 'set_language', code: settings.languageModel })
    }
  }, [realtime.bridgeStatus, realtime.activeLanguage, settings.languageModel])

  useEffect(() => {
    const handleKeyDown = (event) => {
      const targetTag = event.target?.tagName
      if (targetTag === 'INPUT' || targetTag === 'TEXTAREA' || targetTag === 'SELECT') {
        return
      }

      const nextMode = KEYBOARD_SHORTCUTS[event.key]
      if (nextMode) {
        setMode(nextMode)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const renderContent = () => {
    if (mode === MODES.DASHBOARD) {
      return <Dashboard onNavigate={setMode} />
    }

    if (mode === MODES.PRACTICE) {
      return <PracticeMode onBack={() => setMode(MODES.DASHBOARD)} />
    }

    if (mode === MODES.LIVE_CAPTION) {
      return <LiveCaptionMode onBack={() => setMode(MODES.DASHBOARD)} />
    }

    if (mode === MODES.SETTINGS) {
      return <Settings onBack={() => setMode(MODES.DASHBOARD)} />
    }

    if (mode === MODES.CALIBRATION) {
      return <Calibration onComplete={() => setMode(MODES.DASHBOARD)} />
    }

    return null
  }

  return (
    <div className="min-h-screen bg-app text-app-text">
      <Navigation currentMode={mode} onNavigate={setMode} bridgeStatus={realtime.bridgeStatus} />
      <div className="ml-64 mx-auto flex min-h-screen w-full max-w-7xl flex-col px-5 py-5 sm:px-6 lg:px-8">
        <header className="h-12 mb-6 flex items-center justify-between">
          <div className="text-sm font-semibold uppercase tracking-wider text-app-muted">
            {MODES[mode] || 'Dashboard'}
          </div>
          <div className="text-xs text-app-muted">
            {realtime.bridgeStatus === 'connected' ? '● Connected' : '○ Disconnected'}
          </div>
        </header>
        {renderContent()}
      </div>
    </div>
  )
}

export default App
