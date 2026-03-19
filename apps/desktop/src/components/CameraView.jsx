import { useEffect, useRef, useState, useCallback } from 'react'
import { useStore } from '../store'

export default function CameraView({ active = true }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [hasCamera, setHasCamera] = useState(false)
  const [error, setError] = useState(null)
  const frameLoopRef = useRef(null)

  const { sendFrame, isConnected, settings } = useStore()

  // Capture a frame from the video and send it via WebSocket
  const captureAndSend = useCallback(() => {
    if (!active || !isConnected) return
    const video = videoRef.current
    const canvas = canvasRef.current
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
          // reader.result is "data:image/jpeg;base64,XXXX"
          const base64 = reader.result.split(',')[1]
          sendFrame(base64)
        }
        reader.readAsDataURL(blob)
      },
      'image/jpeg',
      0.7,
    )
  }, [active, isConnected, sendFrame])

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

  // Frame capture loop (~15 fps to keep bandwidth reasonable)
  useEffect(() => {
    if (!hasCamera || !active) return

    const interval = setInterval(captureAndSend, 66)
    frameLoopRef.current = interval

    return () => clearInterval(interval)
  }, [hasCamera, active, captureAndSend])

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
        className="w-full h-full object-cover mirror"
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Hidden canvas used for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {!isConnected && hasCamera && (
        <div className="absolute top-2 right-2 bg-red-600/80 text-xs px-2 py-1 rounded">
          Backend offline
        </div>
      )}
    </div>
  )
}
