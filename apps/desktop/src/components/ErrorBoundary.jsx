import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    })
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-app text-app-text">
          <div className="max-w-md space-y-4 px-6 py-8 text-center">
            <div className="text-6xl">⚠️</div>
            <h1 className="text-2xl font-semibold">Something went wrong</h1>
            <p className="text-sm text-app-muted">
              An unexpected error occurred. Please refresh the page or try again.
            </p>
            {this.state.error && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm font-semibold text-app-muted">
                  Error details
                </summary>
                <pre className="mt-2 overflow-auto rounded-lg bg-white/5 p-4 text-xs">
                  {this.state.error.toString()}
                  {this.state.errorInfo && this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
            <button
              onClick={() => window.location.reload()}
              className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white transition hover:bg-blue-400"
            >
              Refresh page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
