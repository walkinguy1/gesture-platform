import { memo } from 'react'

function Navigation({ currentMode, onNavigate }) {
  const items = [
    { key: 'dashboard', label: 'Dashboard', icon: '📊', shortcut: 'D' },
    { key: 'practice', label: 'Practice', icon: '✋', shortcut: 'P' },
    { key: 'live-caption', label: 'Live Captions', icon: '💬', shortcut: 'L' },
    { key: 'calibration', label: 'Calibration', icon: '⚙️', shortcut: 'C' },
    { key: 'settings', label: 'Settings', icon: '🔧', shortcut: 'S' }
  ]

  return (
    <nav className="fixed left-0 top-0 h-full w-64 bg-black/40 border-r border-white/10 p-4 space-y-2" aria-label="Main navigation">
      {items.map(item => (
        <button
          key={item.key}
          onClick={() => onNavigate(item.key)}
          className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center justify-between ${currentMode === item.key
            ? 'bg-blue-500 text-white'
            : 'text-gray-400 hover:bg-white/10'
            }`}
          aria-current={currentMode === item.key ? 'page' : undefined}
          title={`${item.label} (Press ${item.shortcut})`}
        >
          <span className="flex items-center gap-2">
            {item.icon} {item.label}
          </span>
          <span className="text-xs opacity-60">{item.shortcut}</span>
        </button>
      ))}
    </nav>
  )
}

export default memo(Navigation)
