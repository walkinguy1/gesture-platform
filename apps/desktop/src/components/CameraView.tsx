import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'

export default function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hasCamera, setHasCamera] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { setPrediction, setConfidence, settings } = useStore()

  useEffect(() => {
    let stream: MediaStream | null = null

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
        setError('Could not access camera. Please grant permission.')
        setHasCamera(false)
      }
    }

    initCamera()

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Draw loop for landmarks overlay
  useEffect(() => {
    if (!hasCamera || !canvasRef.current || !videoRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationId: number

    const draw = () => {
      // Mirror the canvas
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)

      // Draw video frame
      if (videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      }

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animationId)
    }
  }, [hasCamera])

  if (error) {
    return (
      <div className="aspect-video bg-gray-900 flex items-center justify-center">
        <div className="text-center p-6">
          <div className="text-4xl mb-4">📷</div>
          <p className="text-red-400">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!hasCamera) {
    return (
      <div className="aspect-video bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-pulse text-4xl mb-4">⏳</div>
          <p className="text-gray-400">Loading camera...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full aspect-video bg-gray-900"
        style={{ transform: 'scaleX(-1)' }}
      />
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        className="absolute top-0 left-0 w-full aspect-video pointer-events-none"
      />

      {/* Camera overlay */}
      <div className="absolute top-4 left-4 bg-black/50 px-3 py-1 rounded text-sm">
        <span className="text-green-400">●</span> Camera Active
      </div>

      {settings.showLandmarks && (
        <div className="absolute bottom-4 right-4 bg-black/50 px-3 py-1 rounded text-sm">
          Hand Tracking: ON
        </div>
      )}
    </div>
  )
}
