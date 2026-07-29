import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { subscribeToFrames } from '../frameStream'

// How long without a frame before the preview reports the video as stalled.
const FRAME_TIMEOUT_MS = 2500

export default function CameraView() {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)
  const lastFrameAtRef = useRef(0)
  const frameSizeRef = useRef({ width: 640, height: 360 })

  // 'waiting' until the first frame arrives, then 'live'; 'stalled' if frames
  // stop. Updated only on transitions -- never per frame -- so incoming video
  // doesn't drive React re-renders.
  const [streamState, setStreamState] = useState('waiting')

  const bridgeStatus = useStore((state) => state.realtime.bridgeStatus)
  const showLandmarks = useStore((state) => state.settings.showLandmarks)
  const cameraIndex = useStore((state) => state.settings.cameraIndex)

  useEffect(() => {
    const unsubscribe = subscribeToFrames((frame) => {
      if (!frame) {
        setStreamState((previous) => (previous === 'waiting' ? previous : 'waiting'))
        return
      }

      lastFrameAtRef.current = frame.receivedAt
      frameSizeRef.current = { width: frame.width, height: frame.height }

      // Assigning .src directly keeps decoding off the React path entirely.
      if (imgRef.current) {
        imgRef.current.src = frame.url
      }

      setStreamState((previous) => (previous === 'live' ? previous : 'live'))
    })

    const stallCheck = window.setInterval(() => {
      if (!lastFrameAtRef.current) {
        return
      }
      const stalled = Date.now() - lastFrameAtRef.current > FRAME_TIMEOUT_MS
      setStreamState((previous) => {
        const next = stalled ? 'stalled' : 'live'
        return previous === next ? previous : next
      })
    }, 1000)

    return () => {
      unsubscribe()
      window.clearInterval(stallCheck)
    }
  }, [])

  // Guide box overlay. Redrawn only when the frame size or toggle changes --
  // it's static geometry, so there's no reason to animate it per frame.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    const draw = () => {
      const { width, height } = frameSizeRef.current
      canvas.width = width
      canvas.height = height
      context.clearRect(0, 0, width, height)

      if (!showLandmarks || streamState !== 'live') {
        return
      }

      const guideWidth = width * 0.42
      const guideHeight = height * 0.62
      const guideX = (width - guideWidth) / 2
      const guideY = (height - guideHeight) / 2

      context.strokeStyle = 'rgba(110, 231, 183, 0.85)'
      context.lineWidth = 2
      context.setLineDash([12, 9])
      context.strokeRect(guideX, guideY, guideWidth, guideHeight)

      context.setLineDash([])
      context.fillStyle = 'rgba(8, 15, 28, 0.4)'
      context.fillRect(guideX, guideY, guideWidth, 22)

      context.fillStyle = '#d1fae5'
      context.font = '600 13px "Segoe UI Variable", "Segoe UI", sans-serif'
      context.fillText('Keep one hand centered in the guide box', guideX + 10, guideY + 16)
    }

    draw()
    // Frame dimensions are only known after the first frame lands, so redraw
    // once the image reports its natural size.
    const image = imgRef.current
    image?.addEventListener('load', draw)
    return () => image?.removeEventListener('load', draw)
  }, [showLandmarks, streamState])

  if (bridgeStatus !== 'connected' || streamState !== 'live') {
    const isBackendDown = bridgeStatus !== 'connected'
    const title = isBackendDown
      ? 'Waiting for the recognizer'
      : streamState === 'stalled'
        ? 'Video stalled'
        : 'Starting camera stream...'
    const body = isBackendDown
      ? 'The Python backend owns the camera and streams the preview. It starts automatically with the app; if you launched the UI on its own, run: python scripts/realtime_demo.py --headless'
      : streamState === 'stalled'
        ? `No frames for a few seconds. Camera ${cameraIndex} may have been disconnected or claimed by another app.`
        : 'Connected to the recognizer. Waiting for the first camera frame.'

    return (
      <div className="camera-shell flex min-h-[320px] items-center justify-center">
        <div className="max-w-md space-y-3 px-6 py-8 text-center">
          {!isBackendDown && streamState !== 'stalled' && (
            <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-emerald-200/25 border-t-emerald-300" />
          )}
          <div
            className={`text-sm font-semibold uppercase tracking-[0.3em] ${
              streamState === 'stalled' ? 'text-rose-300' : 'text-app-muted'
            }`}
          >
            {title}
          </div>
          <p className="text-sm text-slate-300">{body}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="camera-shell relative overflow-hidden">
      {/* Frames arrive already mirrored and annotated from the backend, so no
          CSS flip here -- that would un-mirror the preview. */}
      <img
        ref={imgRef}
        alt="Live recognizer view"
        className="h-full w-full bg-slate-950 object-cover"
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      <div className="absolute left-4 top-4 rounded-full border border-white/10 bg-slate-950/70 px-3 py-1.5 text-xs font-medium text-slate-100 backdrop-blur">
        Recognizer view
      </div>
      <div className="absolute right-4 top-4 rounded-full border border-emerald-400/30 bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-100 backdrop-blur">
        Camera {cameraIndex}
      </div>
      <div className="absolute bottom-4 left-4 rounded-2xl border border-white/10 bg-slate-950/75 px-4 py-3 text-xs text-slate-200 backdrop-blur">
        <div className="font-semibold text-white">Preview tips</div>
        <div className="mt-1">
          Good lighting, one hand in frame, and a calm background will improve recognition.
        </div>
      </div>
    </div>
  )
}
