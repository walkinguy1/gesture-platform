import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'

export default function CameraView() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const retryCountRef = useRef(0)

  const [cameraState, setCameraState] = useState('idle')
  const [error, setError] = useState(null)
  const [deviceLabel, setDeviceLabel] = useState('Default camera')

  const { settings } = useStore()

  const initCamera = async () => {
    let isDisposed = false
    let activeStream = null

    const stopStream = () => {
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop())
        activeStream = null
      }
    }

    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Camera access is not available in this environment.')
        setCameraState('error')
        return
      }

      try {
        setCameraState('loading')
        setError(null)

        const devices = await navigator.mediaDevices.enumerateDevices()
        const videoInputs = devices.filter((device) => device.kind === 'videoinput')
        const selectedDevice = videoInputs[settings.cameraIndex]

        activeStream = await navigator.mediaDevices.getUserMedia({
          video: selectedDevice?.deviceId
            ? {
              deviceId: { exact: selectedDevice.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 }
            }
            : {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: 'user'
            }
        })

        if (isDisposed) {
          stopStream()
          return
        }

        if (videoRef.current) {
          videoRef.current.srcObject = activeStream
          await videoRef.current.play()
        }

        setDeviceLabel(selectedDevice?.label || `Camera ${settings.cameraIndex}`)
        setCameraState('ready')
        retryCountRef.current = 0
      } catch (err) {
        console.error('Camera error:', err)
        setError('Could not access the selected camera. Check permissions and device selection.')
        setCameraState('error')
      }
    }

    await startCamera()

    return () => {
      isDisposed = true
      stopStream()
    }
  }

  useEffect(() => {
    let cleanup = null
    initCamera().then((fn) => { cleanup = fn })
    return () => { if (cleanup) cleanup() }
  }, [settings.cameraIndex])

  useEffect(() => {
    if (cameraState !== 'ready' || !canvasRef.current || !videoRef.current) {
      return
    }

    const canvas = canvasRef.current
    const video = videoRef.current
    const context = canvas.getContext('2d')

    if (!context) {
      return
    }

    let animationId = 0

    const syncCanvasSize = () => {
      canvas.width = video.videoWidth || 1280
      canvas.height = video.videoHeight || 720
    }

    const drawGuideOverlay = () => {
      syncCanvasSize()
      context.setTransform(1, 0, 0, 1, 0, 0)
      context.clearRect(0, 0, canvas.width, canvas.height)

      if (!settings.showLandmarks) {
        animationId = requestAnimationFrame(drawGuideOverlay)
        return
      }

      const guideWidth = canvas.width * 0.42
      const guideHeight = canvas.height * 0.62
      const guideX = (canvas.width - guideWidth) / 2
      const guideY = (canvas.height - guideHeight) / 2

      context.strokeStyle = 'rgba(110, 231, 183, 0.85)'
      context.lineWidth = 3
      context.setLineDash([14, 10])
      context.strokeRect(guideX, guideY, guideWidth, guideHeight)

      context.setLineDash([])
      context.fillStyle = 'rgba(8, 15, 28, 0.35)'
      context.fillRect(guideX, guideY, guideWidth, 28)

      context.fillStyle = '#d1fae5'
      context.font = '600 18px "Segoe UI Variable", "Segoe UI", sans-serif'
      context.fillText('Keep one hand centered in the guide box', guideX + 14, guideY + 20)

      animationId = requestAnimationFrame(drawGuideOverlay)
    }

    if (video.readyState >= 1) {
      syncCanvasSize()
    }

    video.addEventListener('loadedmetadata', syncCanvasSize)
    drawGuideOverlay()

    return () => {
      video.removeEventListener('loadedmetadata', syncCanvasSize)
      cancelAnimationFrame(animationId)
      context.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [cameraState, settings.showLandmarks])

  if (cameraState === 'error') {
    return (
      <div className="camera-shell flex min-h-[320px] items-center justify-center">
        <div className="max-w-sm space-y-3 px-6 py-8 text-center">
          <div className="text-sm font-semibold uppercase tracking-[0.3em] text-rose-300">
            Camera unavailable
          </div>
          <p className="text-sm text-slate-300">{error}</p>
          <button
            onClick={() => {
              retryCountRef.current += 1
              initCamera()
            }}
            className="rounded-full bg-rose-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-rose-400"
          >
            Retry camera
          </button>
        </div>
      </div>
    )
  }

  if (cameraState !== 'ready') {
    return (
      <div className="camera-shell flex min-h-[320px] items-center justify-center">
        <div className="space-y-3 text-center">
          <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-emerald-200/25 border-t-emerald-300" />
          <p className="text-sm text-slate-300">Starting camera preview...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="camera-shell relative overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="h-full w-full bg-slate-950 object-cover"
        style={{ transform: 'scaleX(-1)' }}
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      <div className="absolute left-4 top-4 rounded-full border border-white/10 bg-slate-950/70 px-3 py-1.5 text-xs font-medium text-slate-100 backdrop-blur">
        Live preview
      </div>
      <div className="absolute right-4 top-4 rounded-full border border-emerald-400/30 bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-100 backdrop-blur">
        {deviceLabel}
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
