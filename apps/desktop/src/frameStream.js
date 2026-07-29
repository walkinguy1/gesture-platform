/**
 * Camera frame transport.
 *
 * The Python backend owns the webcam exclusively -- a second consumer can open
 * the device but cannot read from it -- so the UI can't run its own
 * getUserMedia preview. Instead the backend broadcasts annotated JPEG frames
 * over the WebSocket bridge and CameraView renders them.
 *
 * Frames deliberately bypass the Zustand store: at ~15fps a base64 payload in
 * store state would re-render every subscribed component several times a
 * second. This module is a plain pub/sub so only CameraView reacts, and it
 * writes straight to an <img> element without going through React state.
 */

const listeners = new Set()
let latestFrame = null

/**
 * @param {{url: string, width: number, height: number, receivedAt: number} | null} frame
 */
export function publishFrame(frame) {
  latestFrame = frame
  listeners.forEach((listener) => {
    try {
      listener(frame)
    } catch (error) {
      console.error('Frame listener failed:', error)
    }
  })
}

/**
 * Subscribe to incoming frames. The most recent frame (if any) is delivered
 * immediately so a newly mounted view doesn't wait for the next broadcast.
 *
 * @returns {() => void} unsubscribe
 */
export function subscribeToFrames(listener) {
  listeners.add(listener)
  if (latestFrame) {
    listener(latestFrame)
  }
  return () => listeners.delete(listener)
}

export function getLatestFrame() {
  return latestFrame
}

/** Drop the retained frame so a reconnect doesn't show stale video. */
export function clearFrames() {
  publishFrame(null)
}
