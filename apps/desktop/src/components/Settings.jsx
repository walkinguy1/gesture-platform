import { memo, useState, useEffect } from 'react'
import { useStore } from '../store'
import { Panel, StatRow, ToggleRow, Button } from './index'
import { CAMERA_CHOICES, FALLBACK_SIGN_LANGUAGES } from '../constants'

function Settings({ onBack }) {
  const [confirmReset, setConfirmReset] = useState(false)

  const {
    settings,
    progress,
    calibration,
    realtime,
    bridgeApi,
    updateSettings,
    resetProgress
  } = useStore()

  const masteredLetters = progress.letters.map(l => l.letter)

  const languages = realtime.languages.length > 0 ? realtime.languages : FALLBACK_SIGN_LANGUAGES

  const handleSelectLanguage = (code) => {
    updateSettings({ languageModel: code })
    const sent = bridgeApi.sendMessage?.({ type: 'set_language', code })
    if (!sent) {
      console.warn('Bridge not connected; language will apply once the backend reconnects.')
    }
  }

  useEffect(() => {
    if (confirmReset) {
      const timeout = setTimeout(() => setConfirmReset(false), 6000)
      return () => clearTimeout(timeout)
    }
  }, [confirmReset])

  return (
    <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[0.95fr_1.05fr]">
      <section className="grid gap-6">
        <Panel title="Appearance" eyebrow="Theme">
          <div className="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-4">
            <div>
              <div className="font-semibold">Color mode</div>
              <div className="mt-1 text-sm text-app-muted">
                Switch between the brighter canvas and the darker studio theme.
              </div>
            </div>
            <Button
              variant="secondary"
              onClick={() =>
                updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })
              }
            >
              {settings.theme === 'dark' ? 'Use light' : 'Use dark'}
            </Button>
          </div>
        </Panel>

        <Panel title="Practice profile" eyebrow="Learner">
          <StatRow label="Calibration" value={calibration.isCalibrated ? 'Completed' : 'Recommended'} />
          <StatRow label="Measured hand size" value={calibration.handSize ? calibration.handSize.toFixed(4) : 'Not stored'} />
          <StatRow label="Letters mastered" value={`${masteredLetters.length}/26`} />
          <StatRow label="Practice time" value={`${progress.totalPracticeTime} min`} />
        </Panel>

        <Panel title="Safety" eyebrow="Reset">
          {!confirmReset ? (
            <Button
              variant="danger"
              className="w-full"
              onClick={() => setConfirmReset(true)}
            >
              Reset practice progress
            </Button>
          ) : (
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-app-muted">This will erase all progress. Are you sure?</span>
              <div className="flex gap-2">
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => {
                    resetProgress()
                    setConfirmReset(false)
                  }}
                >
                  Yes, reset
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setConfirmReset(false)}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </Panel>
      </section>

      <section className="grid gap-6">
        <Panel title="Camera" eyebrow="Input">
          <div className="grid gap-3">
            <div className="text-sm text-app-muted">
              Pick the camera the recognizer should open. The backend owns the
              device and streams its view back to this preview.
            </div>
            <div className="flex flex-wrap gap-2">
              {CAMERA_CHOICES.map((cameraIndex) => (
                <Button
                  key={cameraIndex}
                  variant={settings.cameraIndex === cameraIndex ? 'primary' : 'secondary'}
                  onClick={() => updateSettings({ cameraIndex })}
                >
                  Camera {cameraIndex}
                </Button>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="Sign language" eyebrow="Vocabulary">
          <div className="grid gap-3">
            <div className="text-sm text-app-muted">
              Choose which sign language the recognizer should use. Switching
              here tells the Python backend to reload its models live.
            </div>
            <div className="grid gap-2">
              {languages.map((lang) => {
                const isSelected = settings.languageModel === lang.code
                const badges = []
                if (lang.static_ready) badges.push('Alphabet ready')
                if (lang.dynamic_ready) badges.push('Words ready')
                if (!lang.static_ready && !lang.dynamic_ready) badges.push('Needs training data')
                else if (lang.supports_dynamic && !lang.dynamic_ready) badges.push('Words: needs training data')

                return (
                  <button
                    key={lang.code}
                    onClick={() => handleSelectLanguage(lang.code)}
                    className={`flex items-center justify-between rounded-2xl border px-4 py-3 text-left transition ${
                      isSelected
                        ? 'border-emerald-400/50 bg-emerald-500/10'
                        : 'border-white/8 bg-white/5 hover:bg-white/8'
                    }`}
                  >
                    <div>
                      <div className="font-semibold">
                        {lang.name} <span className="text-app-muted">({lang.code})</span>
                      </div>
                      <div className="mt-1 text-xs text-app-muted">{badges.join(' · ')}</div>
                    </div>
                    {isSelected && (
                      <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
                        Selected
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
            {realtime.bridgeStatus !== 'connected' && (
              <p className="text-xs text-amber-300">
                Backend not connected yet -- this choice will be sent as soon as it is.
              </p>
            )}
          </div>
        </Panel>

        <Panel title="Recognition" eyebrow="Thresholds">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-app-muted">Confidence threshold</span>
              <span className="font-semibold">{Math.round(settings.confidenceThreshold * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={settings.confidenceThreshold}
              onChange={(event) =>
                updateSettings({ confidenceThreshold: Number(event.target.value) })
              }
              className="slider"
            />
            <p className="mt-2 text-sm text-app-muted">
              Lower values react faster. Higher values wait for cleaner detections.
            </p>
          </div>

          <ToggleRow
            label="Prediction smoothing"
            body="Have the recognizer favor repeated detections before reporting a sign as stable."
            checked={settings.smoothingEnabled}
            onToggle={() =>
              updateSettings({ smoothingEnabled: !settings.smoothingEnabled })
            }
          />

          <ToggleRow
            label="Hand landmarks"
            body="Draw the tracked hand skeleton and framing box on the preview."
            checked={settings.showLandmarks}
            onToggle={() =>
              updateSettings({ showLandmarks: !settings.showLandmarks })
            }
          />
        </Panel>

        <Panel title="Shortcuts and model scope" eyebrow="Reference">
          <div className="grid gap-3">
            <StatRow label="D" value="Dashboard" />
            <StatRow label="P" value="Practice mode" />
            <StatRow label="L" value="Live captions" />
            <StatRow label="C" value="Calibration" />
            <StatRow label="S" value="Settings" />
            <StatRow
              label="Active language"
              value={realtime.activeLanguage || `${settings.languageModel} (pending)`}
            />
          </div>
          <p className="mt-4 text-sm text-app-muted">
            Recognition runs in the Python backend, which the app launches on startup.
            Changes on this page are pushed to it live over the bridge.
          </p>
        </Panel>

        <div className="flex justify-start">
          <Button variant="secondary" onClick={onBack}>
            Back
          </Button>
        </div>
      </section>
    </div>
  )
}

export default memo(Settings)
