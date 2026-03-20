import { useStore } from '../store'

interface SettingsProps {
  onBack: () => void
}

export default function Settings({ onBack }: SettingsProps) {
  const { settings, updateSettings, progress, resetProgress, handSize, isCalibrated } = useStore()

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          ← Back
        </button>

        <h2 className="text-2xl font-bold">
          Settings
        </h2>

        <div className="w-20" />  {/* Spacer */}
      </div>

      <div className="space-y-6">
        {/* User Profile */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">User Profile</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Calibration Status:</span>
              <span className={isCalibrated ? 'text-green-400' : 'text-yellow-400'}>
                {isCalibrated ? '✓ Calibrated' : 'Not Calibrated'}
              </span>
            </div>

            {handSize && (
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Hand Size:</span>
                <span className="text-white">{handSize.toFixed(4)}</span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <span className="text-gray-400">Letters Mastered:</span>
              <span className="text-white">{progress.letters.length}/26</span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-gray-400">Practice Time:</span>
              <span className="text-white">{progress.totalPracticeTime} minutes</span>
            </div>
          </div>
        </div>

        {/* Recognition Settings */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Recognition Settings</h3>

          <div className="space-y-4">
            {/* Confidence Threshold */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400">Confidence Threshold:</span>
                <span className="text-white">{Math.round(settings.confidenceThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={settings.confidenceThreshold}
                onChange={(e) => updateSettings({
                  confidenceThreshold: parseFloat(e.target.value)
                })}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-sm text-gray-500 mt-1">
                Lower = more predictions, Higher = more accurate
              </p>
            </div>

            {/* Smoothing */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-white">Temporal Smoothing</span>
                <p className="text-sm text-gray-500">
                  Reduces prediction jitter
                </p>
              </div>
              <button
                onClick={() => updateSettings({
                  smoothingEnabled: !settings.smoothingEnabled
                })}
                className={`w-14 h-8 rounded-full transition-colors ${
                  settings.smoothingEnabled ? 'bg-green-600' : 'bg-gray-600'
                }`}
              >
                <div className={`w-6 h-6 bg-white rounded-full transition-transform ${
                  settings.smoothingEnabled ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            {/* Show Landmarks */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-white">Show Hand Landmarks</span>
                <p className="text-sm text-gray-500">
                  Display skeleton overlay
                </p>
              </div>
              <button
                onClick={() => updateSettings({
                  showLandmarks: !settings.showLandmarks
                })}
                className={`w-14 h-8 rounded-full transition-colors ${
                  settings.showLandmarks ? 'bg-green-600' : 'bg-gray-600'
                }`}
              >
                <div className={`w-6 h-6 bg-white rounded-full transition-transform ${
                  settings.showLandmarks ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
          </div>
        </div>

        {/* Language Model */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Language Model</h3>

          <div className="space-y-2">
            <button
              onClick={() => updateSettings({ languageModel: 'ASL' })}
              className={`w-full p-4 rounded-lg text-left transition-colors ${
                settings.languageModel === 'ASL'
                  ? 'bg-green-600 border-2 border-green-400'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <div className="font-semibold">ASL (American Sign Language)</div>
              <div className="text-sm text-gray-400">26 letters + 10 numbers</div>
            </button>

            <button
              onClick={() => updateSettings({ languageModel: 'BSL' })}
              disabled
              className="w-full p-4 rounded-lg text-left bg-gray-700/50 opacity-50 cursor-not-allowed"
            >
              <div className="font-semibold">BSL (British Sign Language)</div>
              <div className="text-sm text-gray-400">Coming Soon</div>
            </button>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="bg-gray-800 rounded-xl p-6 border border-red-800">
          <h3 className="text-lg font-semibold mb-4 text-red-400">Danger Zone</h3>

          <button
            onClick={() => {
              if (confirm('Are you sure you want to reset all progress?')) {
                resetProgress()
              }
            }}
            className="w-full py-3 bg-red-600 hover:bg-red-500 rounded-lg font-semibold"
          >
            Reset All Progress
          </button>
        </div>

        {/* About */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">About</h3>

          <div className="space-y-2 text-gray-400">
            <p>Gesture Platform v2.0.0</p>
            <p>Real-time Sign Language Translation</p>
            <p className="text-sm">
              Built with MediaPipe, TensorFlow, and Tauri
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
