import { memo } from 'react'
import { useStore } from '../store'
import { Panel, StatRow, ToggleRow, Button } from './index'
import { CAMERA_CHOICES } from '../constants'

function Settings({ onBack }) {
  const {
    settings,
    progress,
    handSize,
    isCalibrated,
    updateSettings,
    resetProgress
  } = useStore()

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
          <StatRow label="Calibration" value={isCalibrated ? 'Completed' : 'Recommended'} />
          <StatRow label="Hand size" value={handSize ? handSize.toFixed(4) : 'Not stored'} />
          <StatRow label="Letters mastered" value={`${progress.letters.length}/26`} />
          <StatRow label="Practice time" value={`${progress.totalPracticeTime} min`} />
        </Panel>

        <Panel title="Safety" eyebrow="Reset">
          <Button
            variant="danger"
            className="w-full"
            onClick={() => {
              if (window.confirm('Reset all saved practice progress?')) {
                resetProgress()
              }
            }}
          >
            Reset practice progress
          </Button>
        </Panel>
      </section>

      <section className="grid gap-6">
        <Panel title="Camera" eyebrow="Input">
          <div className="grid gap-3">
            <div className="text-sm text-app-muted">
              Pick the camera index the preview should request.
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
            body="Favor repeated detections before the UI treats them as stable."
            checked={settings.smoothingEnabled}
            onToggle={() =>
              updateSettings({ smoothingEnabled: !settings.smoothingEnabled })
            }
          />

          <ToggleRow
            label="Guide overlay"
            body="Show the framing box on top of the camera preview."
            checked={settings.showLandmarks}
            onToggle={() =>
              updateSettings({ showLandmarks: !settings.showLandmarks })
            }
          />
        </Panel>

        <Panel title="Shortcuts and model scope" eyebrow="Reference">
          <div className="grid gap-3">
            <StatRow label="Esc" value="Return home" />
            <StatRow label="P" value="Practice mode" />
            <StatRow label="L" value="Live captions" />
            <StatRow label="C" value="Calibration" />
            <StatRow label="S" value="Settings" />
            <StatRow label="Language model" value={settings.languageModel} />
          </div>
          <p className="mt-4 text-sm text-app-muted">
            The desktop surface currently manages preview and interaction states. The Python
            scripts in this repo still handle the full recognizer pipeline.
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
