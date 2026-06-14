import { memo } from 'react'
import { useStore } from '../store'
import { Panel, StatRow, ProgressBar, Card, Button } from './index'
import { ALPHABET } from '../constants'

function Dashboard({ onNavigate }) {
  const { progress, calibration, settings } = useStore()

  const masteredPercentage = (progress.letters.length / 26) * 100
  const unmasteredCount = 26 - progress.letters.length
  const isCalibrated = calibration.isCalibrated

  return (
    <div className="space-y-6">
      <section className="grid gap-4 lg:grid-cols-3">
        <Panel title="Mastery Progress" eyebrow="LETTERS">
          <ProgressBar
            value={progress.letters.length}
            max={26}
            label="Letters learned"
            colorClass="from-emerald-500 to-emerald-400"
            showPercentage={true}
          />
        </Panel>

        <Panel title="Current Streak" eyebrow="PRACTICE">
          <StatRow
            label="Days"
            value={progress.streak}
          />
          <StatRow
            label="Total time"
            value={`${progress.totalPracticeTime} min`}
          />
        </Panel>

        <Panel title="Setup Status" eyebrow="DEVICE">
          <StatRow
            label="Calibration"
            value={isCalibrated ? '✓ Ready' : '⚠ Pending'}
            className={isCalibrated ? '' : 'text-amber-300'}
          />
          <StatRow
            label="Camera"
            value={`Camera ${settings.cameraIndex}`}
          />
          {!isCalibrated && (
            <Button
              variant="primary"
              className="mt-3 w-full"
              onClick={() => onNavigate('calibration')}
            >
              Calibrate now
            </Button>
          )}
        </Panel>
      </section>

      {/* Letter Grid */}
      <Panel title="Letter Grid" eyebrow="PROGRESS">
        <div className="mb-3 flex items-center justify-between text-sm">
          <span className="text-app-muted">Mastered: {progress.letters.length}/26</span>
          <span className="text-app-muted">Remaining: {unmasteredCount}</span>
        </div>
        <div className="grid gap-2 grid-cols-6 md:grid-cols-8 lg:grid-cols-13">
          {ALPHABET.map(letter => (
            <div
              key={letter}
              className={`aspect-square rounded-lg flex items-center justify-center font-bold text-sm transition-all hover:scale-105 cursor-pointer ${progress.letters.includes(letter)
                ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/50 shadow-[0_0_20px_rgba(16,185,129,0.2)]'
                : 'bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:border-white/20'
                }`}
              title={progress.letters.includes(letter) ? 'Mastered' : 'Not mastered'}
            >
              {letter}
            </div>
          ))}
        </div>
        {unmasteredCount > 0 && (
          <Button
            variant="primary"
            className="mt-4 w-full"
            onClick={() => onNavigate('practice')}
          >
            Practice next letter
          </Button>
        )}
      </Panel>

      {/* Quick Actions */}
      <section className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <Card
          title="Practice"
          subtitle="TRAINING"
          body={`Learn new gestures (${unmasteredCount} remaining)`}
          accent="from-emerald-500/35 via-emerald-300/10 to-transparent"
          onClick={() => onNavigate('practice')}
        />
        <Card
          title="Live Captions"
          subtitle="TRANSCRIPTION"
          body="Build phrases in real-time"
          accent="from-sky-500/35 via-sky-300/10 to-transparent"
          onClick={() => onNavigate('live-caption')}
        />
        <Card
          title="Calibration"
          subtitle="SETUP"
          body={isCalibrated ? 'Adjust baseline' : 'Complete setup first'}
          accent={isCalibrated ? 'from-emerald-400/35 via-emerald-300/10 to-transparent' : 'from-amber-400/35 via-amber-300/10 to-transparent'}
          onClick={() => onNavigate('calibration')}
        />
        <Card
          title="Settings"
          subtitle="CONFIG"
          body="Preferences & thresholds"
          accent="from-slate-300/30 via-slate-100/10 to-transparent"
          onClick={() => onNavigate('settings')}
        />
      </section>
    </div>
  )
}

export default memo(Dashboard)
