import { memo } from 'react'
import { LayoutDashboard, Hand, Captions, SlidersHorizontal, Settings } from 'lucide-react'
import { MODES } from '../constants'

function Navigation({ currentMode, onNavigate, bridgeStatus }) {
  const items = [
    { key: MODES.DASHBOARD, label: 'Dashboard', icon: LayoutDashboard, shortcut: 'D' },
    { key: MODES.PRACTICE, label: 'Practice', icon: Hand, shortcut: 'P' },
    { key: MODES.LIVE_CAPTION, label: 'Live Captions', icon: Captions, shortcut: 'L' },
    { key: MODES.CALIBRATION, label: 'Calibration', icon: SlidersHorizontal, shortcut: 'C' },
    { key: MODES.SETTINGS, label: 'Settings', icon: Settings, shortcut: 'S' }
  ]

  const getStatusColor = () => {
    switch (bridgeStatus) {
      case 'connected': return 'bg-emerald-400'
      case 'connecting': return 'bg-amber-400'
      default: return 'bg-rose-400'
    }
  }

  return (
    <nav className="fixed left-0 top-0 h-full w-64 bg-black/40 border-r border-white/10 p-4 flex flex-col" aria-label="Main navigation">
      <div className="mb-6 pb-4 border-b border-white/10">
        <div className="text-lg font-semibold text-app-text">Gesture</div>
        <div className="text-sm text-app-muted">Platform</div>
      </div>

      <div className="space-y-1 flex-1">
        {items.map(item => {
          const Icon = item.icon
          const isActive = currentMode === item.key
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center justify-between ${isActive
                  ? 'bg-white/8 border-l-2 border-emerald-400 text-white'
                  : 'text-app-muted hover:bg-white/5 hover:text-app-text'
                }`}
              aria-current={isActive ? 'page' : undefined}
              title={`${item.label} (Press ${item.shortcut})`}
            >
              <span className="flex items-center gap-3">
                <Icon size={18} aria-hidden="true" />
                {item.label}
              </span>
              <span className="text-xs opacity-60">{item.shortcut}</span>
            </button>
          )
        })}
      </div>

      <div className="pt-4 border-t border-white/10 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
        <span className="text-xs text-app-muted">Recognizer</span>
      </div>
    </nav>
  )
}

export default memo(Navigation)
