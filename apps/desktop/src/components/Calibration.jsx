import { useMemo } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'
import { StatRow, Button } from './index'

export default function Calibration({ onComplete }) {
  const { realtime, calibration, bridgeApi } = useStore()

  // Calibration runs in the backend: it samples 90 frames of real landmark
  // data and feeds the median hand size into Normalizer. This view only
  // starts/stops it and mirrors the progress the backend reports.
  const isCalibrating =
    realtime.calibrationState === 'started' || realtime.calibrationState === 'progress'
  const progress = Math.round((realtime.calibrationProgress || 0) * 100)
  const isComplete = realtime.calibrationState === 'complete' || calibration.isCalibrated
  const isConnected = realtime.bridgeStatus === 'connected'

  const startCalibration = () => {
    bridgeApi.sendMessage?.({ type: 'start_calibration' })
  }

  const cancelCalibration = () => {
    bridgeApi.sendMessage?.({ type: 'cancel_calibration' })
  }

  const secondsRemaining = useMemo(
    () => Math.max(0, Math.ceil(((100 - progress) / 100) * 3)),
    [progress]
  )

  return (
    <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
      <section className="app-panel overflow-hidden">
        <CameraView />
        <div className="px-6 py-6">
          <div className="flex items-center justify-between text-sm text-app-muted">
            <span>Calibration progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="mt-3 h-3 overflow-hidden rounded-full bg-app-track">
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber-300 to-emerald-400 transition-all duration-150"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="mt-3 text-sm text-app-muted">
            {isCalibrating
              ? `Keep your hand flat and steady for about ${secondsRemaining} more second${secondsRemaining === 1 ? '' : 's'}. Progress only advances while a hand is visible.`
              : isComplete
                ? 'Hand size measured and applied to the recognizer for this setup.'
                : isConnected
                  ? 'Start when your hand is centered inside the guide box.'
                  : 'Waiting for the Python recognizer to connect before calibration can run.'}
          </div>
        </div>
      </section>

      <aside className="grid gap-6">
        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Setup
          </div>
          <h2 className="mt-3 text-2xl font-semibold">Guided calibration</h2>
          <div className="mt-5 grid gap-4">
            <Step number="1" title="Frame one hand clearly" body="Use the overlay box to keep the hand centered and visible." />
            <Step number="2" title="Hold a neutral open palm" body="A relaxed, flat hand gives the preview the cleanest reference." />
            <Step number="3" title="Stay steady for a few seconds" body="The desktop flow stores a baseline profile for your current setup." />
          </div>
        </section>

        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Saved baseline
          </div>
          <div className="mt-4 grid gap-3">
            <StatRow
              label="Measured hand size"
              value={calibration.handSize ? calibration.handSize.toFixed(4) : 'Not saved'}
            />
            <StatRow label="Status" value={isComplete ? 'Ready' : isCalibrating ? 'Running' : 'Pending'} />
          </div>
        </section>

        <div className="flex flex-wrap gap-3">
          {isCalibrating ? (
            <Button variant="danger" onClick={cancelCalibration}>
              Cancel
            </Button>
          ) : (
            <Button variant="primary" disabled={!isConnected} onClick={startCalibration}>
              {isComplete ? 'Recalibrate' : 'Start calibration'}
            </Button>
          )}

          {isComplete && !isCalibrating && (
            <Button variant="primary" onClick={onComplete}>
              Continue
            </Button>
          )}

          <Button variant="secondary" onClick={onComplete}>
            Back
          </Button>
        </div>
      </aside>
    </div>
  )
}

function Step({ number, title, body }) {
  return (
    <div className="flex gap-4 rounded-2xl border border-white/8 bg-white/5 px-4 py-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-400/15 text-sm font-semibold text-emerald-100">
        {number}
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="mt-1 text-sm text-app-muted">{body}</div>
      </div>
    </div>
  )
}

