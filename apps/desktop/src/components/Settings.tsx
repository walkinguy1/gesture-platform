import { useStore } from '../store'

interface SettingsProps {
  onBack: () => void
}

export default function Settings({ onBack }: SettingsProps) {
  const { settings, updateSettings, progress, resetProgress, handSize, isCalibrated } = useStore()

  const isDark = settings.theme !== 'light'
  const card = isDark ? 'bg-gray-800' : 'bg-white shadow'
  const label = isDark ? 'text-gray-400' : 'text-gray-500'
  const value = isDark ? 'text-white' : 'text-gray-900'
  const subtext = isDark ? 'text-gray-500' : 'text-gray-400'

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBack}
          className={`px-4 py-2 ${isDark ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} rounded-lg`}
        >
          ← Back
        </button>

        <h2 className="text-2xl font-bold">
          Settings
        </h2>

        <div className="w-20" />  {/* Spacer */}
      </div>

      <div className="space-y-6">
        {/* Appearance */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">Appearance</h3>

          <div className="space-y-4">
            {/* Theme toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className={value}>Theme</span>
                <p className={`text-sm ${subtext}`}>
                  Switch between dark and light mode
                </p>
              </div>
              <button
                onClick={() =>
                  updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })
                }
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isDark
                    ? 'bg-gray-600 hover:bg-gray-500 text-white'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                }`}
                aria-label="Toggle theme"
              >
                {isDark ? '☀ Light' : '🌙 Dark'}
              </button>
            </div>
          </div>
        </div>

        {/* Camera Settings */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">Camera</h3>

          <div className="space-y-4">
            {/* Camera index selector */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className={label}>Camera Device</span>
                <span className={value}>Camera {settings.cameraIndex}</span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {[0, 1, 2, 3].map((idx) => (
                  <button
                    key={idx}
                    onClick={() => updateSettings({ cameraIndex: idx })}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      settings.cameraIndex === idx
                        ? 'bg-green-600 text-white'
                        : isDark
                          ? 'bg-gray-700 hover:bg-gray-600 text-white'
                          : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                    }`}
                  >
                    Camera {idx}
                  </button>
                ))}
              </div>
              <p className={`text-sm ${subtext} mt-1`}>
                Select your camera if the default one is not correct
              </p>
            </div>
          </div>
        </div>

        {/* User Profile */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">User Profile</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className={label}>Calibration Status:</span>
              <span className={isCalibrated ? 'text-green-500' : 'text-yellow-500'}>
                {isCalibrated ? '✓ Calibrated' : 'Not Calibrated'}
              </span>
            </div>

            {handSize && (
              <div className="flex items-center justify-between">
                <span className={label}>Hand Size:</span>
                <span className={value}>{handSize.toFixed(4)}</span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <span className={label}>Letters Mastered:</span>
              <span className={value}>{progress.letters.length}/26</span>
            </div>

            <div className="flex items-center justify-between">
              <span className={label}>Practice Time:</span>
              <span className={value}>{progress.totalPracticeTime} minutes</span>
            </div>
          </div>
        </div>

        {/* Recognition Settings */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">Recognition Settings</h3>

          <div className="space-y-4">
            {/* Confidence Threshold */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className={label}>Confidence Threshold:</span>
                <span className={value}>{Math.round(settings.confidenceThreshold * 100)}%</span>
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
              <p className={`text-sm ${subtext} mt-1`}>
                Lower = more predictions, Higher = more accurate
              </p>
            </div>

            {/* Smoothing */}
            <div className="flex items-center justify-between">
              <div>
                <span className={value}>Temporal Smoothing</span>
                <p className={`text-sm ${subtext}`}>
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
                <span className={value}>Show Hand Landmarks</span>
                <p className={`text-sm ${subtext}`}>
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
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">Language Model</h3>

          <div className="space-y-2">
            <button
              onClick={() => updateSettings({ languageModel: 'ASL' })}
              className={`w-full p-4 rounded-lg text-left transition-colors ${
                settings.languageModel === 'ASL'
                  ? 'bg-green-600 border-2 border-green-400 text-white'
                  : isDark
                    ? 'bg-gray-700 hover:bg-gray-600 text-white'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
              }`}
            >
              <div className="font-semibold">ASL (American Sign Language)</div>
              <div className={`text-sm ${label}`}>26 letters + 10 numbers</div>
            </button>

            <button
              onClick={() => updateSettings({ languageModel: 'BSL' })}
              disabled
              className={`w-full p-4 rounded-lg text-left opacity-50 cursor-not-allowed ${
                isDark ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-900'
              }`}
            >
              <div className="font-semibold">BSL (British Sign Language)</div>
              <div className={`text-sm ${label}`}>Coming Soon</div>
            </button>
          </div>
        </div>

        {/* Keyboard Shortcuts */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
          <div className="space-y-2">
            {[
              { key: 'Esc', action: 'Return to menu' },
              { key: 'P', action: 'Open Practice Mode' },
              { key: 'L', action: 'Open Live Captions' },
              { key: 'S', action: 'Open Settings' },
              { key: 'C', action: 'Open Calibration' },
            ].map(({ key, action }) => (
              <div key={key} className="flex items-center justify-between">
                <span className={label}>{action}</span>
                <kbd className={`px-2 py-1 rounded text-sm font-mono ${isDark ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700'}`}>
                  {key}
                </kbd>
              </div>
            ))}
          </div>
        </div>

        {/* Danger Zone */}
        <div className={`${card} rounded-xl p-6 border border-red-800`}>
          <h3 className="text-lg font-semibold mb-4 text-red-400">Danger Zone</h3>

          <button
            onClick={() => {
              if (confirm('Are you sure you want to reset all progress?')) {
                resetProgress()
              }
            }}
            className="w-full py-3 bg-red-600 hover:bg-red-500 rounded-lg font-semibold text-white"
          >
            Reset All Progress
          </button>
        </div>

        {/* About */}
        <div className={`${card} rounded-xl p-6`}>
          <h3 className="text-lg font-semibold mb-4">About</h3>

          <div className={`space-y-2 ${label}`}>
            <p>Gesture Platform v2.0.0</p>
            <p>Real-time Sign Language Translation</p>
            <p className="text-sm">
              Built with MediaPipe, scikit-learn, and Tauri
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
