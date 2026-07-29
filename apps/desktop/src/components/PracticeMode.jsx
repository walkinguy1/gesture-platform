import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'
import { usePredictionHandler, useTimeout } from '../hooks'
import { PRACTICE_CONFIG, ALPHABET } from '../constants'
import { StatRow, Button } from './index'

export default function PracticeMode({ onBack }) {
  const [currentLetter, setCurrentLetter] = useState('A')
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('Hold the sign steadily to count a rep.')
  const [showSuccess, setShowSuccess] = useState(false)

  const sessionStartedAt = useRef(Date.now())
  const successTimeout = useRef(null)

  const {
    realtime,
    session,
    progress,
    settings,
    updateProgress,
    addPracticeTime,
    setFocusLetter
  } = useStore()

  // Live recognizer output lives under `realtime` in the store -- reading
  // `prediction`/`confidence` off the root yields undefined, which silently
  // stops the practice loop from ever advancing.
  const { prediction, confidence } = realtime

  const maxAttempts = PRACTICE_CONFIG.MAX_ATTEMPTS

  const { isValid, checkConsensus } = usePredictionHandler({
    prediction,
    confidence,
    threshold: settings.confidenceThreshold,
    debounceMs: PRACTICE_CONFIG.DEBOUNCE_MS,
    bufferSize: PRACTICE_CONFIG.BUFFER_WINDOW,
  })

  const unmasteredLetters = useMemo(
    () => ALPHABET.filter((letter) => !progress.letters.some(l => l.letter === letter)),
    [progress.letters]
  )

  // Honor a letter picked from the Dashboard grid, then clear it so returning
  // to practice later doesn't snap back to a stale choice.
  useEffect(() => {
    if (!session.focusLetter) {
      return
    }

    setCurrentLetter(session.focusLetter)
    setAttempts(0)
    setShowSuccess(false)
    setFeedback(`Focusing on ${session.focusLetter}. Hold the sign steadily to count a rep.`)
    setFocusLetter(null)
  }, [session.focusLetter, setFocusLetter])

  useEffect(() => {
    if (unmasteredLetters.length > 0 && progress.letters.some(l => l.letter === currentLetter)) {
      setCurrentLetter(unmasteredLetters[0])
    }
  }, [currentLetter, progress.letters, unmasteredLetters])

  useEffect(() => {
    return () => {
      if (successTimeout.current) {
        window.clearTimeout(successTimeout.current)
      }
      addPracticeTime((Date.now() - sessionStartedAt.current) / 1000)
    }
  }, [addPracticeTime])

  useEffect(() => {
    if (!prediction || !isValid) {
      return
    }

    if (prediction !== currentLetter) {
      setAttempts(0)
      setShowSuccess(false)
      setFeedback(`Detected ${prediction}. Reset and try ${currentLetter} again.`)
      return
    }

    const result = checkConsensus(PRACTICE_CONFIG.MIN_CONSENSUS)
    if (!result || result.prediction !== currentLetter) {
      return
    }

    setAttempts((previousAttempts) => {
      const nextAttempts = previousAttempts + 1

      if (nextAttempts >= PRACTICE_CONFIG.MAX_ATTEMPTS) {
        setShowSuccess(true)
        setFeedback(`${currentLetter} mastered. Moving to the next target.`)
        updateProgress(currentLetter)

        successTimeout.current = window.setTimeout(() => {
          const currentIndex = ALPHABET.indexOf(currentLetter)
          const nextLetter = unmasteredLetters[0] && unmasteredLetters[0] !== currentLetter
            ? unmasteredLetters[0]
            : ALPHABET[(currentIndex + 1) % ALPHABET.length]

          setCurrentLetter(nextLetter)
          setAttempts(0)
          setShowSuccess(false)
          setFeedback('Hold the sign steadily to count a rep.')
        }, 1400)

        return PRACTICE_CONFIG.MAX_ATTEMPTS
      }

      setFeedback(`Good. ${PRACTICE_CONFIG.MAX_ATTEMPTS - nextAttempts} more steady reads to master ${currentLetter}.`)
      return nextAttempts
    })
  }, [
    confidence,
    currentLetter,
    prediction,
    isValid,
    checkConsensus,
    unmasteredLetters,
    updateProgress
  ])

  const jumpToLetter = (direction) => {
    const currentIndex = ALPHABET.indexOf(currentLetter)
    const nextIndex =
      direction === 'previous'
        ? (currentIndex - 1 + ALPHABET.length) % ALPHABET.length
        : (currentIndex + 1) % ALPHABET.length

    if (successTimeout.current) {
      window.clearTimeout(successTimeout.current)
    }

    setCurrentLetter(ALPHABET[nextIndex])
    setAttempts(0)
    setShowSuccess(false)
    setFeedback('Hold the sign steadily to count a rep.')
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <section className="app-panel overflow-hidden">
        <CameraView />
        <div className="grid gap-5 px-6 py-6">
          <div className={`rounded-3xl border px-5 py-4 ${showSuccess
            ? 'border-emerald-400/30 bg-emerald-500/12'
            : 'border-white/10 bg-white/5'
            }`}>
            <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
              Feedback
            </div>
            <p className="mt-2 text-lg font-semibold">{feedback}</p>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between text-sm text-app-muted">
              <span>Steady reps</span>
              <span>{attempts}/{maxAttempts}</span>
            </div>
            <div className="flex gap-2">
              {Array.from({ length: maxAttempts }).map((_, index) => (
                <div
                  key={index}
                  className={`h-3 flex-1 rounded-full ${index < attempts ? 'bg-emerald-400' : 'bg-app-track'
                    }`}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      <aside className="grid gap-6">
        <section className="app-panel px-6 py-6">
          <div className="flex items-center justify-between">
            <Button variant="secondary" onClick={onBack}>
              Back
            </Button>
            <div className="text-sm text-app-muted">
              Mastered {progress.letters.length}/26
            </div>
          </div>

          <div className="mt-8 text-center">
            <div className="text-xs font-semibold uppercase tracking-[0.34em] text-app-muted">
              Current target
            </div>
            <div className="mx-auto mt-4 flex h-44 w-44 items-center justify-center rounded-[2rem] border border-emerald-300/20 bg-emerald-400/10 text-8xl font-semibold text-emerald-100 shadow-[0_20px_60px_rgba(16,185,129,0.18)]">
              {currentLetter}
            </div>
            <p className="mt-4 text-sm text-app-muted">
              Keep your hand inside the guide box until the recognizer sees the same sign three times.
            </p>
          </div>
        </section>

        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Session stats
          </div>
          <div className="mt-4 grid gap-3">
            <StatRow label="Day streak" value={`${progress.streak}`} />
            <StatRow label="Practice time" value={`${progress.totalPracticeTime} min`} />
            <StatRow label="Recognizer threshold" value={`${Math.round(settings.confidenceThreshold * 100)}%`} />
          </div>
        </section>

        <section className="app-panel px-6 py-6">
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => jumpToLetter('previous')} className="flex-1">
              Previous
            </Button>
            <Button variant="secondary" onClick={() => jumpToLetter('next')} className="flex-1">
              Next
            </Button>
          </div>

          {unmasteredLetters.length > 0 && currentLetter !== unmasteredLetters[0] && (
            <Button
              variant="primary"
              className="mt-4 w-full"
              onClick={() => {
                setCurrentLetter(unmasteredLetters[0])
                setAttempts(0)
                setShowSuccess(false)
                setFeedback('Jumped to your next unmastered letter.')
              }}
            >
              Jump to next unmastered letter
            </Button>
          )}
        </section>
      </aside>
    </div>
  )
}
