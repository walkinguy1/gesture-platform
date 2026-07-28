import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'
import { StatRow, Button } from './index'

export default function Calibration({ onComplete }) {
  const [isCalibrating, setIsCalibrating] = useState(false)
  const [progress, setProgress] = useState(0)

  const startedAt = useRef(0)
  const confidenceReadings = useRef([])
  const { realtime, calibration, updateCalibration } = useStore()

  useEffect(() => {
    if (!isCalibrating) {
      confidenceReadings.current = []
      return
    }

    startedAt.current = Date.now()
    confidenceReadings.current = []

    const interval = window.setInterval(() => {
      const elapsedMs = Date.now() - startedAt.current
      const nextProgress = Math.min(100, (elapsedMs / 3500) * 100)
      setProgress(nextProgress)

      // Collect confidence readings from the bridge
      if (realtime.confidence > 0) {
        confidenceReadings.current.push(realtime.confidence)
      }

      if (nextProgress >= 100) {
        // Calculate average confidence baseline from collected readings
        const readings = confidenceReadings.current
        const avgConfidence = readings.length > 0
          ? readings.reduce((sum, val) => sum + val, 0) / readings.length
          : 0.7 // fallback default

        updateCalibration({ handConfidenceBaseline: avgConfidence, isCalibrated: true })
        setIsCalibrating(false)
      }
    }, 50)

    return () => window.clearInterval(interval)
  }, [isCalibrating, realtime.confidence, updateCalibration])

  const secondsRemaining = useMemo(
    () => Math.max(0, Math.ceil(((100 - progress) / 100) * 3.5)),
    [progress]
  )

  const isComplete = progress >= 100

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
              ? `Keep your hand steady for about ${secondsRemaining} more second${secondsRemaining === 1 ? '' : 's'}. Collecting confidence readings...`
              : isComplete
                ? 'Confidence baseline profile saved for this desktop setup.'
                : realtime.bridgeStatus === 'connected'
                  ? 'Start when your hand is centered inside the guide box.'
                  : 'Connect the Python recognizer with WebSocket bridge to enable calibration.'}
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
            <StatRow label="Confidence baseline" value={calibration.handConfidenceBaseline ? calibration.handConfidenceBaseline.toFixed(4) : 'Not saved'} />
            <StatRow label="Status" value={isComplete ? 'Ready' : isCalibrating ? 'Running' : 'Pending'} />
          </div>
        </section>

        <div className="flex flex-wrap gap-3">
          {!isComplete && (
            <Button
              variant="primary"
              disabled={realtime.bridgeStatus !== 'connected'}
              onClick={() => {
                setProgress(0)
                setIsCalibrating(true)
              }}
            >
              {isCalibrating ? 'Restart calibration' : 'Start calibration'}
            </Button>
          )}

          {isComplete && (
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

