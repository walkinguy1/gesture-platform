import { useEffect, useMemo, useRef, useState } from 'react'
import { save } from '@tauri-apps/plugin-dialog'
import { writeTextFile } from '@tauri-apps/plugin-fs'
import { useStore } from '../store'
import CameraView from './CameraView'
import { StatRow, Button } from './index'
import { COMMON_WORDS } from '../constants'

const getSuggestions = (prefix, max = 4) => {
  if (!prefix) {
    return []
  }

  const lower = prefix.toLowerCase()
  return COMMON_WORDS.filter((word) => word.startsWith(lower) && word !== lower).slice(0, max)
}

export default function LiveCaptionMode({ onBack }) {
  const [isRecording, setIsRecording] = useState(false)
  const [captions, setCaptions] = useState([])
  const [sentence, setSentence] = useState('')
  const [showCaptions, setShowCaptions] = useState(true)
  const [copySuccess, setCopySuccess] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')

  const { realtime, settings } = useStore()

  const captionsEndRef = useRef(null)
  const predictionBuffer = useRef([])
  const lastStablePrediction = useRef('')

  useEffect(() => {
    captionsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [captions])

  useEffect(() => {
    if (!isRecording || !realtime.prediction || realtime.confidence < settings.confidenceThreshold) {
      return
    }

    predictionBuffer.current = [...predictionBuffer.current.slice(-5), realtime.prediction]

    const counts = predictionBuffer.current.reduce((map, value) => {
      map[value] = (map[value] || 0) + 1
      return map
    }, {})

    const stableCandidate = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
    if (!stableCandidate || stableCandidate[1] < 3 || stableCandidate[0] === lastStablePrediction.current) {
      return
    }

    lastStablePrediction.current = stableCandidate[0]
    setSentence((previous) => previous + stableCandidate[0])
    setCaptions((previous) => [
      ...previous.slice(-39),
      `[${new Date().toLocaleTimeString()}] ${stableCandidate[0]} (${Math.round(realtime.confidence * 100)}%)`
    ])
  }, [realtime.confidence, isRecording, realtime.prediction, settings.confidenceThreshold])

  const currentWord = useMemo(
    () => sentence.split(' ').filter(Boolean).slice(-1)[0] || '',
    [sentence]
  )
  const suggestions = useMemo(() => getSuggestions(currentWord), [currentWord])

  const handleToggleRecording = () => {
    if (!isRecording) {
      predictionBuffer.current = []
      lastStablePrediction.current = ''
      setCaptions([])
      setSentence('')
      setSaveStatus('')
    }

    setIsRecording((value) => !value)
  }

  const handleSuggestionClick = (word) => {
    const parts = sentence.trimEnd().split(' ').filter(Boolean)
    const nextParts = parts.length > 0 ? [...parts.slice(0, -1), word] : [word]
    setSentence(`${nextParts.join(' ')} `)
  }

  const handleCopy = async () => {
    const text = sentence || captions.join('\n')
    if (!text) {
      return
    }

    try {
      await navigator.clipboard.writeText(text)
      setCopySuccess(true)
      window.setTimeout(() => setCopySuccess(false), 1800)
    } catch (error) {
      console.error('Clipboard write failed', error)
    }
  }

  const handleSave = async () => {
    try {
      const filePath = await save({
        defaultPath: `captions_${Date.now()}.txt`,
        filters: [{ name: 'Text Files', extensions: ['txt'] }]
      })

      if (!filePath) {
        return
      }

      const content = sentence
        ? `Sentence:\n${sentence}\n\nRaw captions:\n${captions.join('\n')}`
        : captions.join('\n')

      await writeTextFile(filePath, content)
      setSaveStatus('Saved transcript.')
    } catch (error) {
      console.error('Error saving captions:', error)
      setSaveStatus('Save failed.')
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.12fr_0.88fr]">
      <section className="app-panel overflow-hidden">
        <CameraView />
        <div className="flex items-center justify-between px-6 py-5">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
              Capture state
            </div>
            <div className="mt-1 text-lg font-semibold">
              {isRecording ? 'Recording stable detections' : 'Paused'}
            </div>
          </div>
          <Button
            variant={isRecording ? 'danger' : 'primary'}
            onClick={handleToggleRecording}
          >
            {isRecording ? 'Pause' : 'Start'}
          </Button>
        </div>
      </section>

      <aside className="grid gap-6">
        <section className="app-panel px-6 py-6">
          <div className="flex items-center justify-between">
            <Button variant="secondary" onClick={onBack}>
              Back
            </Button>
            <Button
              variant="secondary"
              onClick={() => setShowCaptions((value) => !value)}
            >
              {showCaptions ? 'Hide raw feed' : 'Show raw feed'}
            </Button>
          </div>

          <div className="mt-6">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
                Sentence builder
              </div>
              <div className="text-sm text-app-muted">
                {sentence.trim() ? `${sentence.trim().split(/\s+/).length} words` : '0 words'}
              </div>
            </div>

            <div className="mt-3 min-h-[148px] rounded-[1.75rem] border border-white/10 bg-black/20 px-5 py-4 font-mono text-xl tracking-[0.18em] text-app-text">
              {sentence || 'Stable predictions will appear here.'}
              {isRecording && <span className="ml-1 animate-pulse text-emerald-300">|</span>}
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              <Button variant="secondary" onClick={handleCopy} disabled={!sentence && captions.length === 0}>
                {copySuccess ? 'Copied' : 'Copy'}
              </Button>
              <Button variant="secondary" onClick={handleSave} disabled={!sentence && captions.length === 0}>
                Save
              </Button>
              <Button variant="secondary" onClick={() => setSentence((value) => `${value} `)}>
                Add space
              </Button>
              <Button variant="secondary" onClick={() => setSentence((value) => value.slice(0, -1))} disabled={!sentence}>
                Delete
              </Button>
              <Button variant="danger" onClick={() => setSentence('')} disabled={!sentence}>
                Clear
              </Button>
            </div>

            {saveStatus && <div className="mt-3 text-sm text-app-muted">{saveStatus}</div>}

            {suggestions.length > 0 && (
              <div className="mt-5">
                <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
                  Suggestions
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {suggestions.map((word) => (
                    <button
                      key={word}
                      onClick={() => handleSuggestionClick(word)}
                      className="rounded-full border border-sky-300/20 bg-sky-500/10 px-3 py-1.5 text-sm font-medium text-sky-100 transition hover:bg-sky-500/20"
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {showCaptions && (
          <section className="app-panel px-6 py-6">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
                Raw feed
              </div>
              <div className="text-sm text-app-muted">
                {captions.length} entries
              </div>
            </div>

            <div className="mt-4 max-h-72 overflow-y-auto rounded-[1.5rem] border border-white/10 bg-black/20 px-4 py-4">
              {captions.length === 0 ? (
                <div className="text-sm text-app-muted">
                  {isRecording
                    ? 'Waiting for repeated high-confidence detections.'
                    : 'Start a recording session to collect captions.'}
                </div>
              ) : (
                <div className="space-y-2 text-sm text-app-text">
                  {captions.map((caption, index) => (
                    <div key={`${caption}-${index}`} className="animate-fade-in rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
                      {caption}
                    </div>
                  ))}
                  <div ref={captionsEndRef} />
                </div>
              )}
            </div>
          </section>
        )}

        <section className="app-panel px-6 py-6">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-app-muted">
            Current recognizer state
          </div>
          <div className="mt-4 grid gap-3">
            <StatRow label="Current symbol" value={realtime.prediction || 'No bridge signal'} />
            <StatRow label="Confidence gate" value={`${Math.round(settings.confidenceThreshold * 100)}%`} />
            <StatRow label="Latest confidence" value={realtime.prediction ? `${Math.round(realtime.confidence * 100)}%` : '--'} />
          </div>
        </section>
      </aside>
    </div>
  )
}

