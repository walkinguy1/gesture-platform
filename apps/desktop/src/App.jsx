import { useEffect, useMemo, useState } from 'react'
import { useStore } from './store'
import PracticeMode from './components/PracticeMode'
import LiveCaptionMode from './components/LiveCaptionMode'
import Settings from './components/Settings'
import Calibration from './components/Calibration'
import Dashboard from './components/Dashboard'
import { Navigation } from './components/Navigation'
import { MODES, FEATURE_CARDS } from './constants'

function App() {
  const [mode, setMode] = useState('dashboard')
  const { realtime, calibration, progress, settings } = useStore()

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('light', settings.theme === 'light')
    root.classList.toggle('dark', settings.theme !== 'light')
  }, [settings.theme])

  useEffect(() => {
    const handleKeyDown = (event) => {
      const targetTag = event.target?.tagName
      if (targetTag === 'INPUT' || targetTag === 'TEXTAREA' || targetTag === 'SELECT') {
        return
      }

      const shortcuts = {
        Escape: 'dashboard',
        p: 'practice',
        P: 'practice',
        l: 'live-caption',
        L: 'live-caption',
        s: 'settings',
        S: 'settings',
        c: 'calibration',
        C: 'calibration',
        d: 'dashboard',
        D: 'dashboard'
      }

      const nextMode = shortcuts[event.key]
      if (nextMode) {
        setMode(nextMode)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const completion = useMemo(() => {
    return Math.round((progress.letters.length / 26) * 100)
  }, [progress.letters.length])

  const renderContent = () => {
    if (mode === 'dashboard') {
      return <Dashboard onNavigate={setMode} onBack={() => setMode('menu')} />
    }

    if (mode === 'practice') {
      return <PracticeMode onBack={() => setMode('dashboard')} />
    }

    if (mode === 'live-caption') {
      return <LiveCaptionMode onBack={() => setMode('dashboard')} />
    }

    if (mode === 'settings') {
      return <Settings onBack={() => setMode('dashboard')} />
    }

    if (mode === 'calibration') {
      return <Calibration onComplete={() => setMode('dashboard')} />
    }

    return (
      <MainMenu
        completion={completion}
        calibration={calibration}
        progress={progress}
        onSelectMode={setMode}
      />
    )
  }

  return (
    <div className="min-h-screen bg-app text-app-text">
      <Navigation currentMode={mode} onNavigate={setMode} />
      <div className="ml-64 mx-auto flex min-h-screen w-full max-w-7xl flex-col px-5 py-5 sm:px-6 lg:px-8">
        <header className="app-panel mb-6 flex flex-col gap-5 px-5 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.32em] text-app-muted">
              Gesture Platform
            </div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
              Desktop hand-tracking workspace
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-app-muted">
              The desktop app is now cleaner about what it can do locally: camera preview,
              guided practice screens, and setup controls, with the Python recognizer still
              available through the scripts in this repo.
            </p>
          </div>

          <div className="grid gap-3 sm:min-w-[280px]">
            <StatusPill
              label="Active view"
              value={MODES[mode]}
            />
            <StatusPill
              label="Calibration"
              value={calibration.isCalibrated ? 'Ready' : 'Recommended'}
              tone={calibration.isCalibrated ? 'good' : 'warm'}
            />
            <StatusPill
              label="Latest prediction"
              value={realtime.prediction ? `${realtime.prediction} (${Math.round(realtime.confidence * 100)}%)` : 'Waiting for bridge'}
            />
          </div>
        </header>

        <div className="mb-6 grid gap-4 lg:grid-cols-3">
          <div className="app-panel px-5 py-4">
            <div className="text-sm text-app-muted">Letters mastered</div>
            <div className="mt-2 text-4xl font-semibold">{progress.letters.length}/26</div>
          </div>
          <div className="app-panel px-5 py-4">
            <div className="text-sm text-app-muted">Current streak</div>
            <div className="mt-2 text-4xl font-semibold">{progress.streak} days</div>
          </div>
          <div className="app-panel px-5 py-4">
            <div className="flex items-center justify-between text-sm text-app-muted">
              <span>Curriculum progress</span>
              <span>{completion}%</span>
            </div>
            <div className="mt-3 h-3 overflow-hidden rounded-full bg-app-track">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400"
                style={{ width: `${completion}%` }}
              />
            </div>
          </div>
        </div>

        <main className="flex-1">
          {renderContent()}
        </main>
      </div>
    </div>
  )
}

function StatusPill({ label, value, tone = 'neutral' }) {
  const toneClass =
    tone === 'good'
      ? 'border-emerald-400/25 bg-emerald-500/10 text-emerald-100'
      : tone === 'warm'
        ? 'border-amber-400/25 bg-amber-500/10 text-amber-100'
        : 'border-white/10 bg-white/5 text-app-text'

  return (
    <div className={`rounded-2xl border px-4 py-3 ${toneClass}`}>
      <div className="text-[11px] uppercase tracking-[0.28em] text-app-muted">{label}</div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
    </div>
  )
}

function MainMenu({ completion, calibration, progress, onSelectMode }) {
  return (
    <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
      <section className="grid gap-4 md:grid-cols-2">
        {FEATURE_CARDS.map((card) => (
          <button
            key={card.key}
            onClick={() => onSelectMode(card.key)}
            className="group app-panel relative overflow-hidden p-0 text-left transition duration-200 hover:-translate-y-1 hover:border-white/15"
          >
            <div className={`absolute inset-0 bg-gradient-to-br ${card.accent}`} />
            <div className="relative px-6 py-6">
              <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
                {card.eyebrow}
              </div>
              <h2 className="mt-3 text-2xl font-semibold tracking-tight">{card.title}</h2>
              <p className="mt-3 max-w-md text-sm leading-6 text-app-muted">{card.body}</p>
              <div className="mt-6 text-sm font-semibold text-emerald-200 transition group-hover:text-emerald-100">
                Open workspace
              </div>
            </div>
          </button>
        ))}
      </section>

      <aside className="grid gap-4">
        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Readiness
          </div>
          <h3 className="mt-3 text-2xl font-semibold">Session overview</h3>
          <div className="mt-5 grid gap-4">
            <OverviewRow label="Calibration" value={calibration.isCalibrated ? 'Completed' : 'Still recommended'} />
            <OverviewRow label="Practice time" value={`${progress.totalPracticeTime} min`} />
            <OverviewRow label="Words saved" value={`${progress.words.length}`} />
            <OverviewRow label="Coverage" value={`${completion}% of alphabet`} />
          </div>
        </section>

        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Keyboard
          </div>
          <div className="mt-4 grid gap-3 text-sm text-app-muted">
            <Shortcut keyName="P" action="Practice mode" />
            <Shortcut keyName="L" action="Live captions" />
            <Shortcut keyName="C" action="Calibration" />
            <Shortcut keyName="S" action="Settings" />
            <Shortcut keyName="Esc" action="Return home" />
          </div>
        </section>
      </aside>
    </div>
  )
}

function OverviewRow({ label, value }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
      <span className="text-sm text-app-muted">{label}</span>
      <span className="text-sm font-semibold text-app-text">{value}</span>
    </div>
  )
}

function Shortcut({ keyName, action }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
      <span>{action}</span>
      <kbd className="rounded-lg border border-white/10 bg-black/20 px-2 py-1 font-mono text-xs text-app-text">
        {keyName}
      </kbd>
    </div>
  )
}

export default App
