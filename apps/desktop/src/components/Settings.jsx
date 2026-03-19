import { useStore } from '../store'

export default function Settings({ onBack }) {
  const { settings, updateSettings, progress, resetProgress, handSize, isCalibrated } = useStore()

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          ← Back
        </button>

        <h2 className="text-2xl font-bold">Settings</h2>

        <div className="w-20" />
      </div>

      <div className="space-y-6">
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
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Recognition Settings</h3>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-gray-400">Confidence Threshold</label>
                <span className="text-white">{(settings.confidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={settings.confidenceThreshold}
                onChange={(e) => updateSettings({ confidenceThreshold: parseFloat(e.target.value) })}
                className="w-full"
              />
              <div className="text-sm text-gray-500 mt-1">
                Higher = more accurate but slower
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <label className="text-gray-400">Smoothing Window</label>
                <span className="text-white">{settings.smoothingWindow} frames</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={settings.smoothingWindow}
                onChange={(e) => updateSettings({ smoothingWindow: parseInt(e.target.value) })}
                className="w-full"
              />
              <div className="text-sm text-gray-500 mt-1">
                Higher = smoother but more lag
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">Show Landmarks</div>
                <div className="text-sm text-gray-500">Display hand skeleton overlay</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.showLandmarks}
                  onChange={(e) => updateSettings({ showLandmarks: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-700 peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">Show FPS</div>
                <div className="text-sm text-gray-500">Display frames per second</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.showFPS}
                  onChange={(e) => updateSettings({ showFPS: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-700 peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 text-red-400">Danger Zone</h3>

          <button
            onClick={resetProgress}
            className="w-full py-3 bg-red-600 hover:bg-red-500 rounded-lg font-semibold"
          >
            Reset All Progress
          </button>

          <div className="text-sm text-gray-500 mt-2">
            This will clear all mastered letters and calibration data
          </div>
        </div>
      </div>
    </div>
  )
}
