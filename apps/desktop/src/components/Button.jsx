/**
 * Reusable Button Component
 * Eliminates repeated Tailwind class strings across the app
 */

export function Button({
  variant = 'primary',
  size = 'md',
  disabled = false,
  className = '',
  children,
  ...props
}) {
  const variants = {
    primary: 'bg-blue-500 hover:bg-blue-600 text-white',
    secondary: 'bg-slate-700 hover:bg-slate-600 text-white border border-white/10',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    ghost: 'hover:bg-white/10 text-white'
  }

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  }

  const baseStyle = 'rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const variantStyle = variants[variant] || variants.primary
  const sizeStyle = sizes[size] || sizes.md

  return (
    <button
      disabled={disabled}
      className={`${baseStyle} ${variantStyle} ${sizeStyle} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
