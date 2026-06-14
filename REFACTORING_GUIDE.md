# Quick-Start Refactoring Guide

Execute these steps in order to modernize the codebase.

## PHASE 1: Quick Wins (30 min)

### Step 1.1: Update Dependencies
```bash
# Use the new, leaner requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Only when developing
```

**Impact:** Faster pip installs, cleaner environment (~500MB saved)

---

### Step 1.2: Use New Components in Settings.jsx

**File:** `apps/desktop/src/components/Settings.jsx`

Replace hardcoded imports with the new components:

```jsx
// OLD (currently using Panel and StatRow inline)
import { useStore } from '../store'

// NEW - Import component library
import { Button, Panel, StatRow, ProgressBar } from './index'
import { CAMERA_CHOICES } from '../constants'

// Then replace CAMERA_CHOICES hardcoded array:
const CAMERA_CHOICES = [0, 1, 2, 3]  // DELETE this
// Use from constants instead:
import { CAMERA_CHOICES } from '../constants'
```

**Files to update:**
1. `apps/desktop/src/components/Settings.jsx` - import from `./index`
2. `apps/desktop/src/components/Calibration.jsx` - same
3. `apps/desktop/src/App.jsx` - import `FEATURE_CARDS`, `MODES`, `KEYBOARD_SHORTCUTS` from constants

---

### Step 1.3: Use Custom Hooks in PracticeMode.jsx

**File:** `apps/desktop/src/components/PracticeMode.jsx`

Replace inline debouncing logic with the custom hook:

```jsx
// OLD (lines ~55-85)
useEffect(() => {
    if (!prediction || confidence < settings.confidenceThreshold) {
      return
    }
    if (prediction !== currentLetter) {
      setAttempts(0)
      setShowSuccess(false)
      setFeedback(`Detected ${prediction}. Reset...`)
      return
    }

    const now = Date.now()
    if (now - lastAcceptedAt.current < 900) {
      return  // ← Debouncing
    }
    lastAcceptedAt.current = now
    // ... rest of logic
})

// NEW - Use hook
import { usePredictionHandler, useInterval } from '../hooks'

const { isValid, checkConsensus } = usePredictionHandler({
  prediction,
  confidence,
  threshold: settings.confidenceThreshold,
  debounceMs: PRACTICE_CONFIG.DEBOUNCE_MS,
  bufferSize: PRACTICE_CONFIG.BUFFER_WINDOW,
  onValid: (result) => {
    // Handle valid prediction
    setAttempts(prev => prev + 1)
  }
})

useEffect(() => {
  if (isValid) {
    const result = checkConsensus(PRACTICE_CONFIG.MAX_ATTEMPTS)
    // ... use result
  }
}, [isValid, prediction, confidence, checkConsensus])
```

---

### Step 1.4: Clean Up Exception Classes

**File:** `gesture_platform/exceptions.py`

Delete all unused exception classes, keeping only these 5:
1. `GesturePlatformError` (base)
2. `ModelNotLoadedError`
3. `PredictionError`
4. `InputValidationError`
5. `ConfigurationError`

```python
# DELETE everything else - these 5 handle all error cases
```

---

## PHASE 2: Structural Refactoring (2-3 hours)

### Step 2.1: Consolidate Store State

**File:** `apps/desktop/src/store.jsx`

Simplify the store to have a single, unified shape:

```jsx
const useStore = create(
  persist(
    (set) => ({
      // ──────── SETTINGS (User Preferences) ────────
      settings: {
        theme: 'dark',
        cameraIndex: 0,
        confidenceThreshold: 0.7,
        smoothingEnabled: true,
        showLandmarks: true
      },

      // ──────── CALIBRATION (Device Profile) ────────
      calibration: {
        isCalibrated: false,
        handSize: null
      },

      // ──────── PROGRESS (Learner Stats) ────────
      progress: {
        letters: [],
        words: [],
        totalPracticeTime: 0,
        streak: 0,
        lastPracticeDate: null
      },

      // ──────── REALTIME (Current Session) ────────
      realtime: {
        prediction: null,
        confidence: 0
      },

      // ──────── ACTIONS ────────
      updateSettings: (updates) => set(state => ({
        settings: { ...state.settings, ...updates }
      })),

      updateCalibration: (updates) => set(state => ({
        calibration: { ...state.calibration, ...updates }
      })),

      updateProgress: (letter) => set(state => ({
        progress: {
          ...state.progress,
          letters: state.progress.letters.includes(letter)
            ? state.progress.letters
            : [...state.progress.letters, letter]
        }
      })),

      setPrediction: (pred, conf) => set({
        realtime: { prediction: pred, confidence: conf }
      }),

      reset: () => set({
        progress: emptyProgress(),
        realtime: { prediction: null, confidence: 0 }
      })
    }),
    { name: 'gesture-platform', version: 3 }
  )
)
```

Then update all components to use `store.settings`, `store.calibration`, etc.

---

### Step 2.2: Create a Dashboard Component

**File:** `apps/desktop/src/components/Dashboard.jsx` (NEW)

```jsx
import { useStore } from '../store'
import { Panel, StatRow, ProgressBar, Card } from './index'

export default function Dashboard({ onNavigate }) {
  const { progress, calibration, settings } = useStore()

  const masteredPercentage = (progress.letters.length / 26) * 100

  return (
    <div className="space-y-6">
      <section className="grid gap-4 lg:grid-cols-3">
        <Panel title="Mastery Progress" eyebrow="LETTERS">
          <ProgressBar
            value={progress.letters.length}
            max={26}
            label="Letters learned"
            colorClass="from-emerald-500 to-emerald-400"
            showPercentage={true}
          />
        </Panel>

        <Panel title="Current Streak" eyebrow="PRACTICE">
          <StatRow
            label="Days"
            value={progress.streak}
          />
          <StatRow
            label="Total time"
            value={`${progress.totalPracticeTime} min`}
          />
        </Panel>

        <Panel title="Setup Status" eyebrow="DEVICE">
          <StatRow
            label="Calibration"
            value={calibration.isCalibrated ? '✓ Ready' : 'Pending'}
          />
          <StatRow
            label="Camera"
            value={`Camera ${settings.cameraIndex}`}
          />
        </Panel>
      </section>

      {/* Letter Grid */}
      <Panel title="Letter Grid" eyebrow="PROGRESS">
        <div className="grid gap-2 grid-cols-6 md:grid-cols-8 lg:grid-cols-13">
          {Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ').map(letter => (
            <div
              key={letter}
              className={`aspect-square rounded-lg flex items-center justify-center font-bold text-sm ${
                progress.letters.includes(letter)
                  ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/50'
                  : 'bg-white/5 text-white/50 border border-white/10'
              }`}
            >
              {letter}
            </div>
          ))}
        </div>
      </Panel>

      {/* Quick Actions */}
      <section className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <Card
          title="Practice"
          subtitle="TRAINING"
          body="Learn new gestures"
          accent="from-emerald-500/35 via-emerald-300/10 to-transparent"
          onClick={() => onNavigate('practice')}
        />
        <Card
          title="Live Captions"
          subtitle="TRANSCRIPTION"
          body="Build phrases"
          accent="from-sky-500/35 via-sky-300/10 to-transparent"
          onClick={() => onNavigate('live-caption')}
        />
        <Card
          title="Calibration"
          subtitle="SETUP"
          body="Adjust baseline"
          accent="from-amber-400/35 via-amber-300/10 to-transparent"
          onClick={() => onNavigate('calibration')}
        />
        <Card
          title="Settings"
          subtitle="CONFIG"
          body="Preferences"
          accent="from-slate-300/30 via-slate-100/10 to-transparent"
          onClick={() => onNavigate('settings')}
        />
      </section>
    </div>
  )
}
```

---

### Step 2.3: Simplify Configuration

**File:** `gesture_platform/config.py`

Replace complex nested dataclasses with simple config:

```python
from dataclasses import dataclass

@dataclass
class Config:
    """Simplified configuration - production-only settings"""

    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Recognition
    model_path: str = "models/asl_alphabet.pkl"
    confidence_threshold: float = 0.70
    smoothing_window: int = 5
    use_smoothing: bool = True

    # Tracking
    max_num_hands: int = 1
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.50

    # Output
    show_landmarks: bool = True

    @classmethod
    def from_env(cls):
        """Load from environment variables with defaults"""
        import os
        return cls(
            camera_index=int(os.getenv('CAMERA_INDEX', 0)),
            model_path=os.getenv('MODEL_PATH', 'models/asl_alphabet.pkl'),
            confidence_threshold=float(os.getenv('CONFIDENCE', 0.70))
        )
```

Delete everything else from config.py (HandTrackerConfig, RecognizerConfig, etc.).

---

### Step 2.4: Use PredictionSmoother

**Files:** `asl_recognizer.py`, `mlp_model.py`

Replace duplicate buffering logic with the smoother:

```python
# OLD (30+ lines of duplicate code per model)
from collections import deque
# ... 30 lines of prediction_buffer logic ...

# NEW (3 lines)
from .prediction_smoother import PredictionSmoother

class ASLRecognizer:
    def __init__(self, ..., smoothing_window=5, use_smoothing=True):
        self._smoother = PredictionSmoother(window_size=smoothing_window)
        self.use_smoothing = use_smoothing

    def predict_with_smoothing(self, features):
        pred, conf = self.predict(features)
        if self.use_smoothing:
            return self._smoother.get_smoothed()  # ← Single line
        return pred, conf

    def reset_smoothing(self):
        self._smoother.reset()
```

---

## PHASE 3: Add Dashboard & Improve Navigation (2 hours)

### Step 3.1: Update App.jsx to Include Dashboard

```jsx
// Add to MODES
const MODES = {
  dashboard: 'Dashboard',
  menu: 'Home',
  practice: 'Practice',
  // ... rest
}

// Add to renderContent()
const renderContent = () => {
  if (mode === 'dashboard') {
    return <Dashboard onNavigate={setMode} onBack={() => setMode('menu')} />
  }
  // ... rest
}

// Update initial state
const [mode, setMode] = useState('dashboard')  // Start with dashboard instead of menu
```

---

### Step 3.2: Add Sidebar Navigation

Create `Navigation.jsx`:

```jsx
export function Navigation({ currentMode, onNavigate }) {
  const items = [
    { key: 'dashboard', label: 'Dashboard', icon: '📊' },
    { key: 'practice', label: 'Practice', icon: '✋' },
    { key: 'live-caption', label: 'Live Captions', icon: '💬' },
    { key: 'calibration', label: 'Calibration', icon: '⚙️' },
    { key: 'settings', label: 'Settings', icon: '🔧' }
  ]

  return (
    <nav className="fixed left-0 top-0 h-full w-64 bg-black/40 border-r border-white/10 p-4 space-y-2">
      {items.map(item => (
        <button
          key={item.key}
          onClick={() => onNavigate(item.key)}
          className={`w-full text-left px-4 py-3 rounded-lg transition ${
            currentMode === item.key
              ? 'bg-blue-500 text-white'
              : 'text-gray-400 hover:bg-white/10'
          }`}
        >
          {item.icon} {item.label}
        </button>
      ))}
    </nav>
  )
}
```

---

## VERIFICATION CHECKLIST

After completing the above steps, verify:

- [ ] `pip install -r requirements.txt` works (no TensorFlow errors)
- [ ] All `.jsx` files import from `components/index` instead of defining Button/Panel inline
- [ ] `constants.js` has all hardcoded values
- [ ] Store has unified shape (settings, calibration, progress, realtime)
- [ ] Custom hooks are used in PracticeMode and LiveCaptionMode
- [ ] Dashboard displays all 26 letters in a grid
- [ ] Navigation sidebar appears in App
- [ ] No console errors in DevTools
- [ ] Predictions still work during practice mode
- [ ] Settings persist across sessions

---

## NEXT STEPS (Lower Priority)

- [ ] Error boundaries for graceful fallbacks
- [ ] Camera preview with hand detection overlay
- [ ] Mobile responsiveness improvements
- [ ] Dark/light theme refinements
- [ ] Performance profiling
- [ ] Unit tests for custom hooks

---

## File Summary

**New files created:**
- `apps/desktop/src/components/Button.jsx`
- `apps/desktop/src/components/index.jsx` (Panel, StatRow, etc.)
- `apps/desktop/src/components/Dashboard.jsx`
- `apps/desktop/src/components/Navigation.jsx`
- `apps/desktop/src/hooks/index.js`
- `apps/desktop/src/constants.js`
- `gesture_platform/prediction_smoother.py`
- `requirements-dev.txt`

**Files to modify:**
- `apps/desktop/src/App.jsx` - import constants, add dashboard
- `apps/desktop/src/store.jsx` - consolidate state shape
- `apps/desktop/src/components/Settings.jsx` - use components, constants
- `apps/desktop/src/components/PracticeMode.jsx` - use hooks
- `apps/desktop/src/components/LiveCaptionMode.jsx` - use hooks
- `apps/desktop/src/components/Calibration.jsx` - use components
- `gesture_platform/config.py` - simplify to single Config
- `gesture_platform/exceptions.py` - delete 12 unused classes
- `gesture_platform/asl_recognizer.py` - use PredictionSmoother
- `gesture_platform/mlp_model.py` - use PredictionSmoother

**Files to delete:**
- None (but can deprecate HandTrackerConfig, RecognizerConfig, etc.)

---

Estimated completion time: **5-6 hours** for full modernization
