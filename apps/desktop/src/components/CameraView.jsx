import { useEffect, useRef, useState, useCallback } from 'react'
import { useStore } from '../store'

// MediaPipe hand skeleton connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [0,9],[9,10],[10,11],[11,12],  // middle
  [0,13],[13,14],[14,15],[15,16],// ring
  [0,17],[17,18],[18,19],[19,20],// pinky
  [5,9],[9,13],[13,17],          // palm
]

function drawLandmarks(ctx, landmarks, width, height) {
  if (!landmarks || landmarks.length < 21) return

  const pts = landmarks.map(([x, y]) => ({
    x: (1 - x) * width, // mirror to match mirrored video
    y: y * height,
  }))

  // Connections
  ctx.strokeStyle = 'rgba(0, 255, 128, 0.85)'
  ctx.lineWidth = 2
  ctx.lineJoin = 'round'
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath()
    ctx.moveTo(pts[a].x, pts[a].y)
    ctx.lineTo(pts[b].x, pts[b].y)
    ctx.stroke()
  }

  // Landmark dots
  for (const { x, y } of pts) {
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.fill()
    ctx.strokeStyle = 'rgba(0, 200, 100, 1)'
    ctx.lineWidth = 1.5
    ctx.stroke()
  }
}

export default function CameraView({ active = true }) {
  const videoRef = useRef(null)
  const captureCanvasRef = useRef(null)  // hidden, for JPEG encoding
  const overlayCanvasRef = useRef(null)  // visible, for landmark drawing
  const [hasCamera, setHasCamera] = useState(false)
  const [error, setError] = useState(null)
  const frameLoopRef = useRef(null)
  const overlayLoopRef = useRef(null)
  const fpsRef = useRef({ count: 0, last: Date.now(), display: 0 })

  const { sendFrame, isConnected, settings } = useStore()
  // Pull prediction/landmarks from store for overlay
  const { landmarks, prediction, confidence } = useStore()

  // Capture a frame from the video and send it via WebSocket
  const captureAndSend = useCallback(() => {
    if (!active || !isConnected) return
    const video = videoRef.current
    const canvas = captureCanvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const reader = new FileReader()
        reader.onloadend = () => {
          const base64 = reader.result.split(',')[1]
          sendFrame(base64)
          // FPS tracking
          fpsRef.current.count++
          const now = Date.now()
          if (now - fpsRef.current.last >= 1000) {
            fpsRef.current.display = fpsRef.current.count
            fpsRef.current.count = 0
            fpsRef.current.last = now
          }
        }
        reader.readAsDataURL(blob)
      },
      'image/jpeg',
      0.7,
    )
  }, [active, isConnected, sendFrame])

  // Overlay draw loop (runs every animation frame)
  const drawOverlay = useCallback(() => {
    const video = videoRef.current
    const canvas = overlayCanvasRef.current
    if (!video || !canvas) return

    const w = video.offsetWidth || video.videoWidth || 640
    const h = video.offsetHeight || video.videoHeight || 360
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, w, h)

    if (settings.showLandmarks && landmarks && landmarks.length > 0) {
      drawLandmarks(ctx, landmarks, w, h)
    }

    if (settings.showFPS) {
      ctx.font = 'bold 14px monospace'
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fillRect(6, 6, 72, 22)
      ctx.fillStyle = '#00ff80'
      ctx.fillText(`${fpsRef.current.display} fps`, 10, 22)
    }

    if (prediction && confidence > 0) {
      const conf = Math.round(confidence * 100)
      const label = `${prediction}  ${conf}%`
      ctx.font = 'bold 28px sans-serif'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = 'rgba(0,0,0,0.55)'
      ctx.fillRect(w / 2 - tw / 2 - 8, h - 52, tw + 16, 38)
      ctx.fillStyle = confidence > 0.8 ? '#4ade80' : '#facc15'
      ctx.textAlign = 'center'
      ctx.fillText(label, w / 2, h - 22)
      ctx.textAlign = 'left'
    }
  }, [settings.showLandmarks, settings.showFPS, landmarks, prediction, confidence])

  // Camera init
  useEffect(() => {
    let stream = null

    const initCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setHasCamera(true)
        }
      } catch (err) {
        console.error('Camera error:', err)
        setError('Failed to access camera. Please allow camera access.')
      }
    }

    initCamera()

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Frame capture loop (~30 fps)
  useEffect(() => {
    if (!hasCamera || !active) return

    const interval = setInterval(captureAndSend, 33)
    frameLoopRef.current = interval

    return () => clearInterval(interval)
  }, [hasCamera, active, captureAndSend])

  // Overlay draw loop (rAF)
  useEffect(() => {
    let animId
    const loop = () => {
      drawOverlay()
      animId = requestAnimationFrame(loop)
    }
    animId = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animId)
  }, [drawOverlay])

  return (
    <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-red-400">{error}</div>
        </div>
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Visible overlay canvas for landmarks + HUD */}
      <canvas
        ref={overlayCanvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Hidden canvas used only for JPEG frame capture */}
      <canvas ref={captureCanvasRef} className="hidden" />

      {!isConnected && hasCamera && (
        <div className="absolute top-2 right-2 bg-red-600/80 text-xs px-2 py-1 rounded">
          Backend offline
        </div>
      )}
    </div>
  )
}
