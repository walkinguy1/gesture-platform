/**
 * Button Component
 * Consistent button styling with variants
 */
export function Button({ variant = 'primary', children, className = '', ...props }) {
    const variants = {
        primary: 'bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-400 transition',
        secondary: 'bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition',
        danger: 'bg-rose-500 text-white px-4 py-2 rounded-lg hover:bg-rose-400 transition',
        ghost: 'text-white px-4 py-2 rounded-lg hover:bg-white/10 transition'
    }
    return (
        <button className={`${variants[variant]} ${className}`} {...props}>
            {children}
        </button>
    )
}

export { ErrorBoundary } from './ErrorBoundary'

/**
 * Panel Component
 * Reusable container for grouped content sections
 */

export function Panel({ title, eyebrow, children, className = '' }) {
    return (
        <section className={`rounded-2xl border border-white/8 bg-white/5 p-6 ${className}`}>
            {eyebrow && (
                <div className="text-xs font-semibold uppercase tracking-wider text-app-muted">
                    {eyebrow}
                </div>
            )}
            {title && (
                <h3 className={`text-lg font-semibold ${eyebrow ? 'mt-2' : ''}`}>
                    {title}
                </h3>
            )}
            <div className={title || eyebrow ? 'mt-4' : ''}>
                {children}
            </div>
        </section>
    )
}

/**
 * StatRow Component
 * Display a label-value pair (commonly used in settings/stats)
 */
export function StatRow({ label, value, className = '' }) {
    return (
        <div className={`flex items-center justify-between rounded-lg border border-white/5 bg-white/3 px-4 py-3 text-sm ${className}`}>
            <span className="text-app-muted">{label}</span>
            <span className="font-semibold">{value}</span>
        </div>
    )
}

/**
 * Card Component
 * Used for feature highlights and main navigation options
 */
export function Card({
    title,
    subtitle,
    body,
    accent = 'from-blue-500/35 via-blue-300/10 to-transparent',
    onClick,
    className = ''
}) {
    return (
        <button
            onClick={onClick}
            className={`group rounded-2xl border border-white/8 bg-gradient-to-b ${accent} p-6 text-left transition-all hover:border-white/15 hover:bg-white/8 ${className}`}
        >
            {subtitle && (
                <div className="text-xs font-semibold uppercase tracking-wider text-app-muted">
                    {subtitle}
                </div>
            )}
            {title && (
                <h3 className={`text-xl font-semibold ${subtitle ? 'mt-2' : ''}`}>
                    {title}
                </h3>
            )}
            {body && (
                <p className="mt-3 text-sm text-app-muted">
                    {body}
                </p>
            )}
        </button>
    )
}

/**
 * ProgressBar Component
 * Visual progress indicator
 */
export function ProgressBar({
    value,
    max = 100,
    label,
    colorClass = 'from-blue-500 to-blue-400',
    showPercentage = true
}) {
    const percentage = (value / max) * 100

    return (
        <div>
            {(label || showPercentage) && (
                <div className="mb-2 flex items-center justify-between text-sm">
                    {label && <span className="text-app-muted">{label}</span>}
                    {showPercentage && <span className="font-semibold">{Math.round(percentage)}%</span>}
                </div>
            )}
            <div className="h-3 overflow-hidden rounded-full bg-app-track">
                <div
                    className={`h-full rounded-full bg-gradient-to-r ${colorClass} transition-all duration-150`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    )
}

/**
 * ToggleRow Component
 * Toggle switch with label and description
 */
export function ToggleRow({ label, body, checked, onToggle }) {
    return (
        <div className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/5 px-4 py-4">
            <div>
                <div className="font-semibold">{label}</div>
                <div className="mt-1 text-sm text-app-muted">{body}</div>
            </div>
            <button
                onClick={onToggle}
                className={`relative h-8 w-16 rounded-full transition ${checked ? 'bg-emerald-400/80' : 'bg-white/15'
                    }`}
            >
                <span
                    className={`absolute top-1 h-6 w-6 rounded-full bg-white transition ${checked ? 'left-9' : 'left-1'
                        }`}
                />
            </button>
        </div>
    )
}
