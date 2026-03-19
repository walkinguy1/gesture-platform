import { useState, useEffect } from 'react'
import { useStore } from '../store'
import CameraView from './CameraView'

export default function Calibration({ onComplete }) {
  const [calibrationComplete, setCalibrationComplete] = useState(false)
  const [localProgress, setLocalProgress] = useState(0)
  const [localHandSize, setLocalHandSize] = useState(null)

  const {
    sendCalibrationStart,
    sendCalibrationStop,
    isConnected,
    isCalibrated,
    handSize,
  } = useStore()

  // Listen to store changes pushed by WebSocket
  useEffect(() => {
    const unsub = useStore.subscribe((state, prev) => {
      if (state.isCalibrated && !prev.isCalibrated) {
        setCalibrationComplete(true)
        setLocalHandSize(state.handSize)
        setLocalProgress(100)
      }
    })
    return unsub
  }, [])

  // Also listen for calibration progress messages
  useEffect(() => {
    const handler = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'calibration' && data.status === 'progress') {
          setLocalProgress(data.progress)
        }
      } catch { /* ignore */ }
    }

    const { ws } = useStore.getState()
    if (ws) ws.addEventListener('message', handler)
    return () => {
      if (ws) ws.removeEventListener('message', handler)
    }
  }, [isConnected])

  const startCalibration = () => {
    setLocalProgress(0)
    sendCalibrationStart()
  }

  const isCalibrating = localProgress > 0 && localProgress < 100 && !calibrationComplete

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Hand Calibration</h2>
      </div>

      {!calibrationComplete ? (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Instructions</h3>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center font-bold">1</div>
                <div>
                  <div className="font-semibold">Hold your hand flat</div>
                  <div className="text-gray-400 text-sm">
                    Extend your hand with palm facing the camera
                  </div>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center font-bold">2</div>
                <div>
                  <div className="font-semibold">Keep your hand steady</div>
                  <div className="text-gray-400 text-sm">
                    Hold still for 3 seconds during calibration
                  </div>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center font-bold">3</div>
                <div>
                  <div className="font-semibold">Calibration helps accuracy</div>
                  <div className="text-gray-400 text-sm">
                    Your hand size is used to normalize signs
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl overflow-hidden">
            <CameraView active={true} />

            {isCalibrating && (
              <div className="p-6">
                <div className="text-center mb-4">
                  <div className="text-xl font-semibold text-yellow-400">
                    Calibrating...
                  </div>
                  <div className="text-gray-400">
                    Hold your hand steady
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">Progress</span>
                    <span>{Math.round(localProgress)}%</span>
                  </div>
                  <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-yellow-500 transition-all duration-100"
                      style={{ width: `${localProgress}%` }}
                    />
                  </div>
                </div>

                <div className="text-center text-sm text-gray-500">
                  {Math.max(0, Math.round((100 - localProgress) / 100 * 3))} seconds remaining
                </div>
              </div>
            )}
          </div>

          {!isCalibrating && (
            <button
              onClick={startCalibration}
              disabled={!isConnected}
              className="w-full py-4 bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-semibold text-lg"
            >
              {isConnected ? 'Start Calibration' : 'Waiting for backend...'}
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-xl p-8 text-center">
            <div className="text-6xl mb-4">&#x2713;</div>
            <h3 className="text-2xl font-bold text-green-400 mb-2">
              Calibration Complete!
            </h3>
            <p className="text-gray-400 mb-6">
              Your hand has been calibrated for optimal recognition
            </p>

            {localHandSize && (
              <div className="bg-gray-700 rounded-lg p-4 inline-block">
                <div className="text-sm text-gray-400">Calibrated Hand Size</div>
                <div className="text-2xl font-bold">{localHandSize.toFixed(4)}</div>
              </div>
            )}
          </div>

          <button
            onClick={onComplete}
            className="w-full py-4 bg-green-600 hover:bg-green-500 rounded-xl font-semibold text-lg"
          >
            Continue
          </button>
        </div>
      )}

      <div className="mt-6 bg-gray-800 rounded-xl p-4">
        <h4 className="font-semibold mb-2">Why calibrate?</h4>
        <ul className="list-disc list-inside text-gray-400 text-sm space-y-1">
          <li>Different hand sizes can affect recognition accuracy</li>
          <li>Calibration normalizes your hand to the model</li>
          <li>Re-calibrate if recognition feels inaccurate</li>
        </ul>
      </div>
    </div>
  )
}
