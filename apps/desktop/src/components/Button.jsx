/**
 * Canonical Button Component
 * Consistent button styling with variants and sizes
 */

export function Button({ variant = 'primary', size = 'md', disabled, className = '', children, ...props }) {
  const variants = {
    primary: 'primary-button',
    secondary: 'secondary-button',
    danger: 'danger-button',
    ghost: 'hover:bg-white/10 text-white'
  }

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base'
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
