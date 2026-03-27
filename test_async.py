"""
Test AsyncPipeline - Should give 25-30 FPS
"""
from gesture_platform.pipeline import AsyncPipeline
import cv2
import time

def main():
    print("Creating AsyncPipeline...")

    pipeline = AsyncPipeline(
        model_path="models/asl_alphabet.pkl",
        camera_index=0,
        frame_width=1280,
        frame_height=720,
        show_landmarks=True,
        confidence_threshold=0.70,
        use_smoothing=True
    )

    print("Pipeline created! Starting camera...")
    print("Press 'q' to quit\n")

    fps_history = []

    with pipeline:
        start = time.time()
        frame_count = 0

        while frame_count < 300:  # Run for ~10 seconds
            result = pipeline.get_result(timeout=0.1)

            if result is not None:
                frame_count += 1
                fps_history.append(result.fps)

                # Print every 30 frames
                if frame_count % 30 == 0:
                    avg_fps = sum(fps_history[-30:]) / min(30, len(fps_history))
                    print(f"Frame {frame_count}: FPS = {avg_fps:.1f}, Prediction = {result.prediction}")

                # Display frame
                frame = result.frame

                # Draw prediction
                if result.prediction:
                    color = (0, 255, 0) if result.confidence > 0.85 else (0, 255, 255)
                    text = f"{result.prediction} ({result.confidence:.0%})"
                    cv2.putText(frame, text, (30, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

                # Draw FPS
                cv2.putText(frame, f"FPS: {result.fps:.1f}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('AsyncPipeline Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = time.time() - start
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        print(f"\n{'='*50}")
        print(f"Test Complete!")
        print(f"{'='*50}")
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Min FPS: {min(fps_history):.1f}")
        print(f"Max FPS: {max(fps_history):.1f}")

        if avg_fps > 20:
            print(f"\n✅ EXCELLENT! Much better than 4 FPS!")
        elif avg_fps > 15:
            print(f"\n✅ GOOD! Better than before!")
        else:
            print(f"\n⚠️  Still slow, might need more optimization")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
