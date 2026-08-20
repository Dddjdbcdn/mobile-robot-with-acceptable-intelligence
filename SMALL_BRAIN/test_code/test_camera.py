import base64
import time
from pathlib import Path

import cv2


def capture_snapshot(
    camera_index: int = 0,
    output_path: str = "camera_snapshot.jpg",
    width: int = 640,
    height: int = 480,
    jpeg_quality: int = 70,
) -> bytes:
    """
    Capture one USB-camera frame, save it, and return JPEG-encoded bytes.

    Returns:
        JPEG image data as bytes.

    Raises:
        RuntimeError: If the camera cannot be opened or a frame cannot be read.
    """
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try camera_index=1 or check whether another program is using it."
        )

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Request MJPEG from many common USB cameras.
        # Some cameras may ignore this setting, which is okay.
        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        # Reduce the chance of receiving an old buffered frame.
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Discard initial frames while auto-exposure and white balance settle.
        frame = None

        for _ in range(10):
            success, candidate = camera.read()

            if success and candidate is not None:
                frame = candidate

            time.sleep(0.05)

        if frame is None:
            raise RuntimeError("Camera opened, but no valid frame was received.")

        # Encode the OpenCV frame into JPEG in memory.
        encode_success, encoded_image = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        if not encode_success:
            raise RuntimeError("OpenCV could not encode the frame as JPEG.")

        jpeg_bytes = encoded_image.tobytes()

        # Save exactly the same JPEG bytes that will be sent to the LLM.
        path = Path(output_path)
        path.write_bytes(jpeg_bytes)

        print(f"Snapshot saved to: {path.resolve()}")
        print(f"JPEG size: {len(jpeg_bytes):,} bytes")
        print(f"Requested resolution: {width}x{height}")
        print(f"Actual frame shape: {frame.shape}")

        return jpeg_bytes

    finally:
        camera.release()


def main() -> None:
    try:
        jpeg_bytes = capture_snapshot()

        # Useful when an API expects the image inside JSON.
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode("ascii")

        # Many image APIs accept a data URL in this format.
        image_data_url = f"data:image/jpeg;base64,{jpeg_base64}"

        print(f"Base64 length: {len(jpeg_base64):,} characters")
        print(f"Data URL begins with: {image_data_url[:50]}...")

    except RuntimeError as error:
        print(f"Camera error: {error}")


if __name__ == "__main__":
    main()